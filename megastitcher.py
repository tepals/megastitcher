#!/usr/bin/env python3
"""
Created on 20 Oct 2026

Copyright © 2026 Wilco Zuidema, Thera Pals

The megastitcher is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License version 2 as published by the Free Software
Foundation.

The megastitcher is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.

"""

import os
import sys
import glob
import re
import time
import shutil
import threading
import subprocess
import webbrowser
import socket
import numpy as np
import cv2
import tifffile
import pyvips
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import base64
from concurrent.futures import ProcessPoolExecutor
from http.server import HTTPServer, SimpleHTTPRequestHandler

# --- OPTIONAL IMPORTS ---
try:
    import zarr
    import dask.array as da
    from dask_image.imread import imread as dask_imread
    from ome_zarr.writer import write_image
    from ome_zarr.io import parse_url
    from ome_zarr.scale import Scaler
    HAS_OME_ZARR = True
except ImportError:
    HAS_OME_ZARR = False

try:
    import imagecodecs
    HAS_IMAGECODECS = True
except ImportError:
    HAS_IMAGECODECS = False

# --- CONFIGURATION (TWEAK THESE FOR ALIGNMENT) ---
OUTPUT_BASE = os.path.join(os.path.expanduser("~"), "Pictures", "output_megafields")
TEMP_WORK_DIR = os.path.join(os.path.expanduser("~"), "Pictures", "megastitch_temp")

# 1. Global Settings
SEARCH_WIDTH = 400         
DOWNSAMPLE = 0.5           
PIXEL_SIZE_NM = 4.0        
FLAT_FIELD_NAME = "avarage_field_pattern.tiff" 

# 2. Matcher Strictness
MIN_MATCHES = 5            
MATCH_RATIO = 0.75         
RANSAC_THRESH = 5.0        

# 3. Image Pre-Processing (Feature Detection)
CLAHE_CLIP = 4.0           
SIFT_CONTRAST = 0.04       
SIFT_EDGE = 10             

# --- PROCESSING ENGINE ---
def get_strip_downsampled(path, side):
    try:
        img = tifffile.imread(path)
        dir_name = os.path.dirname(path)
        ff_path = os.path.join(dir_name, FLAT_FIELD_NAME)
        
        flat_field = None
        if os.path.exists(ff_path):
            try:
                flat_field = tifffile.imread(ff_path)
                if img.ndim == 2 and flat_field.ndim == 3:
                    flat_field = cv2.cvtColor(flat_field, cv2.COLOR_RGB2GRAY)
                elif img.ndim == 3 and flat_field.ndim == 2:
                    flat_field = cv2.cvtColor(flat_field, cv2.COLOR_GRAY2RGB)
            except:
                flat_field = None

        h, w = img.shape[:2]
        if side == 'right':   x, y, cw, ch = max(0, w - SEARCH_WIDTH), 0, SEARCH_WIDTH, h
        elif side == 'left':  x, y, cw, ch = 0, 0, min(w, SEARCH_WIDTH), h
        elif side == 'bottom':x, y, cw, ch = 0, max(0, h - SEARCH_WIDTH), w, SEARCH_WIDTH
        elif side == 'top':   x, y, cw, ch = 0, 0, w, min(h, SEARCH_WIDTH)

        strip = img[y:y+ch, x:x+cw]
        
        if flat_field is not None:
            fh, fw = flat_field.shape[:2]
            if y+ch <= fh and x+cw <= fw:
                flat_strip = flat_field[y:y+ch, x:x+cw]
                if strip.shape != flat_strip.shape:
                    flat_strip = cv2.resize(flat_strip, (strip.shape[1], strip.shape[0]))
                
                strip_f = strip.astype(np.float32)
                flat_f = flat_strip.astype(np.float32)
                flat_mean = np.mean(flat_f)
                strip = (strip_f / (flat_f + 1e-5)) * flat_mean

        strip_8bit = cv2.normalize(strip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        small = cv2.resize(strip_8bit, (0,0), fx=DOWNSAMPLE, fy=DOWNSAMPLE, interpolation=cv2.INTER_NEAREST)
        if small.ndim == 3: small = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(4,4))
        return clahe.apply(small), x, y
    except Exception as e:
        print(f"Error loading strip {path}: {e}")
        return None, 0, 0

def match_pair_worker(args):
    path_ref, path_curr, direction = args
    if direction == 'horizontal':
        res_ref = get_strip_downsampled(path_ref, 'right')
        res_curr = get_strip_downsampled(path_curr, 'left')
    else:
        res_ref = get_strip_downsampled(path_ref, 'bottom')
        res_curr = get_strip_downsampled(path_curr, 'top')
        
    if res_ref[0] is None or res_curr[0] is None: return None
    img_ref, rx, ry = res_ref
    img_curr, cx, cy = res_curr

    sift = cv2.SIFT_create(contrastThreshold=SIFT_CONTRAST, edgeThreshold=SIFT_EDGE)
    kp1, des1 = sift.detectAndCompute(img_ref, None)
    kp2, des2 = sift.detectAndCompute(img_curr, None)

    if des1 is None or des2 is None or len(kp1) < MIN_MATCHES or len(kp2) < MIN_MATCHES: return None
    
    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < MATCH_RATIO * n.distance]
    if len(good) < MIN_MATCHES: return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESH)
    if M is None: return None

    good_filtered = [good[i] for i in range(len(good)) if mask.ravel()[i]]
    if len(good_filtered) < MIN_MATCHES: return None
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_filtered])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_filtered])
    
    deltas = pts1 - pts2
    scale = 1.0 / DOWNSAMPLE
    return ((rx + np.median(deltas[:,0])*scale) - cx, (ry + np.median(deltas[:,1])*scale) - cy)

# --- GUI ---
class StitcherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MegaStitcher V28 (Optimized Chunking)")
        self.root.geometry("700x750")
        
        tk.Label(root, text="Source Images (Network/USB):", font=("Arial", 10, "bold")).pack(pady=(10, 5))
        self.path_frame = tk.Frame(root)
        self.path_frame.pack()
        self.entry_path = tk.Entry(self.path_frame, width=50); self.entry_path.pack(side=tk.LEFT, padx=5)
        tk.Button(self.path_frame, text="Browse", command=self.browse_folder).pack(side=tk.LEFT)

        tk.Label(root, text="Selection (Leave blank for ALL shapes/files):", font=("Arial", 10, "bold")).pack(pady=(15, 5))
        f_grid = tk.Frame(root)
        f_grid.pack()
        tk.Label(f_grid, text="Start:").grid(row=0, column=0, padx=5)
        self.entry_tl = tk.Entry(f_grid, width=15); self.entry_tl.grid(row=0, column=1, padx=5)
        tk.Label(f_grid, text="End:").grid(row=1, column=0, padx=5)
        self.entry_br = tk.Entry(f_grid, width=15); self.entry_br.grid(row=1, column=1, padx=5)

        # Export Options Frame
        export_frame = tk.LabelFrame(root, text="Export Formats", font=("Arial", 9, "bold"), padx=10, pady=5)
        export_frame.pack(pady=15, fill="x", padx=100)
        
        self.use_bigtiff = tk.BooleanVar(value=True)
        self.chk_bigtiff = tk.Checkbutton(export_frame, text="Export 16-bit BigTIFF", variable=self.use_bigtiff, font=("Arial", 9))
        self.chk_bigtiff.pack(anchor="w")

        self.use_zarr = tk.BooleanVar(value=True)
        self.chk_zarr = tk.Checkbutton(export_frame, text="Export OME-Zarr (Multiscale)", variable=self.use_zarr, font=("Arial", 9))
        self.chk_zarr.pack(anchor="w")
        
        tk.Label(export_frame, text="* 8-bit DZI web preview is always generated.", font=("Arial", 8, "italic"), fg="gray").pack(anchor="w", pady=(5,0))

        status_txt = "Ready."
        if not HAS_OME_ZARR:
            status_txt = "Missing 'ome-zarr' or 'dask'. Zarr disabled."
            self.chk_zarr.config(state="disabled")
            self.use_zarr.set(False)
        elif not HAS_IMAGECODECS:
             status_txt = "Warning: 'imagecodecs' missing. BigTIFF will be uncompressed."
        
        tk.Label(root, text=status_txt, fg="red" if "Missing" in status_txt else "orange" if "Warning" in status_txt else "green").pack()

        tk.Label(root, text="Progress:", font=("Arial", 9)).pack(pady=(5, 0))
        self.progress = ttk.Progressbar(root, orient="horizontal", length=600, mode="determinate")
        self.progress.pack(pady=5)

        self.log_text = scrolledtext.ScrolledText(root, height=10, width=80, state='disabled', bg="#f0f0f0", font=("Consolas", 9))
        self.log_text.pack(pady=5, padx=10)

        self.btn_run = tk.Button(root, text="COPY & STITCH", bg="#2E7D32", fg="white", 
                                font=("Arial", 12, "bold"), height=2, width=20, command=self.run_process)
        self.btn_run.pack(pady=10)

    def log(self, msg):
        print(msg)
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update_idletasks()

    def browse_folder(self):
        d = filedialog.askdirectory()
        if d: self.entry_path.delete(0, tk.END); self.entry_path.insert(0, d)

    def run_process(self):
        src_dir = self.entry_path.get()
        tl = self.entry_tl.get().strip()
        br = self.entry_br.get().strip()
        if not os.path.isdir(src_dir): return messagebox.showerror("Error", "Invalid Directory")
        
        use_bounds = False
        tl_r = tl_c = br_r = br_c = 0
        if tl or br:
            try:
                tl_r, tl_c = map(int, tl.split('_')[:2])
                br_r, br_c = map(int, br.split('_')[:2])
                use_bounds = True
            except: 
                return messagebox.showerror("Error", "Format: 000_000 or leave blank for all files")

        self.btn_run.config(state=tk.DISABLED, text="Running...")
        self.log_text.config(state='normal'); self.log_text.delete(1.0, tk.END); self.log_text.config(state='disabled')
        threading.Thread(target=self.stitch_logic, args=(src_dir, use_bounds, tl_r, tl_c, br_r, br_c)).start()

    def convert_to_ome_zarr(self, tiff_path, zarr_path):
        self.log("Starting OME-Zarr conversion...")
        try:
            dask_arr = dask_imread(tiff_path)
            
            if dask_arr.ndim == 2: dask_arr = dask_arr[np.newaxis, np.newaxis, ...] 
            elif dask_arr.ndim == 3: dask_arr = dask_arr[np.newaxis, ...]

            # --- FORCE LARGE CHUNKS TO REDUCE FILE COUNT ---
            self.log("Rechunking to 4096x4096 to reduce file count...")
            dask_arr = dask_arr.rechunk((1, 1, 4096, 4096))

            self.log("Calculating display contrast...")
            try:
                d, c, h, w = dask_arr.shape
                cy, cx = h // 2, w // 2
                sample = dask_arr[0, 0, cy-1000:cy+1000, cx-1000:cx+1000].compute()
                vmin, vmax = np.percentile(sample, (1, 99))
                if vmax < 255: vmax = 65535 
                self.log(f"Auto-Contrast: {int(vmin)} - {int(vmax)}")
            except:
                vmin, vmax = 0, 65535
                self.log("Auto-Contrast failed, using default 0-65535.")

            omero_meta = {
                "id": 1,
                "channels": [{
                    "active": True,
                    "coefficient": 1,
                    "color": "FFFFFF",
                    "family": "linear",
                    "inverted": False,
                    "label": "Stitched",
                    "window": {
                        "end": float(vmax),
                        "max": 65535,
                        "min": 0,
                        "start": float(vmin)
                    }
                }],
                "rdefs": {"defaultT": 0, "defaultZ": 0, "model": "color"}
            }

            store = parse_url(zarr_path, mode="w").store
            root = zarr.group(store=store)

            # --- USE MATCHING LARGE CHUNKS DURING WRITE ---
            write_image(
                image=dask_arr, group=root, axes="tcyx",
                storage_options=dict(chunks=(1, 1, 4096, 4096)), scaler=Scaler(method='nearest')
            )
            
            root.attrs['omero'] = omero_meta
            self.log("OME-Zarr write & metadata injection complete.")
            return True
            
        except Exception as e:
            self.log(f"Zarr Conversion Failed: {e}")
            return False

    def stitch_logic(self, src_dir, use_bounds, tl_r, tl_c, br_r, br_c):
        try:
            self.log(f"--- PHASE 1: SCANNING SHAPE ---")
            
            files = []
            for ext in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
                files.extend(glob.glob(os.path.join(src_dir, ext)))
            
            flat_field_src = os.path.join(src_dir, FLAT_FIELD_NAME)
            has_ff = os.path.exists(flat_field_src)
            if has_ff: self.log(f"Flat Field Detected: {FLAT_FIELD_NAME}")
            else: self.log("WARNING: No Flat Field found.")

            pattern = re.compile(r"(\d{3})_(\d{3})_(\d+)\.") 
            candidates = {} 
            for f in files:
                if FLAT_FIELD_NAME in f: continue
                m = pattern.search(os.path.basename(f))
                if m:
                    r, c = int(m.group(1)), int(m.group(2))
                    idx = int(m.group(3))
                    
                    if use_bounds:
                        if not (tl_r <= r <= br_r and tl_c <= c <= br_c): continue
                        
                    if (r,c) not in candidates: candidates[(r,c)] = []
                    candidates[(r,c)].append((idx, f))

            network_grid = {}
            for (r,c), variants in candidates.items():
                variants.sort(key=lambda x: x[0])
                for idx, path in variants:
                    if idx == 0:
                        network_grid[(r,c)] = path; break
                
            if not network_grid:
                self.log("Error: No '_0.tiff' files found."); self.btn_run.config(state=tk.NORMAL, text="COPY & STITCH"); return
            self.log(f"Found {len(network_grid)} tiles mapping an arbitrary shape.")

            self.log(f"--- PHASE 2: LOCAL COPY ---")
            if os.path.exists(TEMP_WORK_DIR): shutil.rmtree(TEMP_WORK_DIR)
            os.makedirs(TEMP_WORK_DIR, exist_ok=True)
            self.progress["maximum"] = len(network_grid); self.progress["value"] = 0
            
            if has_ff: shutil.copy2(flat_field_src, os.path.join(TEMP_WORK_DIR, FLAT_FIELD_NAME))

            local_grid = {}
            count = 0
            for (r,c), net_path in network_grid.items():
                count += 1
                fname = os.path.basename(net_path)
                local_path = os.path.join(TEMP_WORK_DIR, fname)
                shutil.copy2(net_path, local_path)
                local_grid[(r,c)] = local_path
                self.progress["value"] = count
                if count % 10 == 0: self.root.update_idletasks()
            self.log("Files copied to SSD.")

            self.log(f"--- PHASE 3: ALIGNING GRAPH ---")
            tasks = []
            task_ids = []
            
            for (r, c) in local_grid.keys():
                if (r, c+1) in local_grid:
                    tasks.append((local_grid[(r, c)], local_grid[(r, c+1)], 'horizontal'))
                    task_ids.append(((r, c), (r, c+1)))
                if (r+1, c) in local_grid:
                    tasks.append((local_grid[(r, c)], local_grid[(r+1, c)], 'vertical'))
                    task_ids.append(((r, c), (r+1, c)))

            results_map = {}
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(match_pair_worker, tasks))
            
            success = 0
            graph = {node: [] for node in local_grid.keys()}
            for idx, res in enumerate(results):
                if res: 
                    u, v = task_ids[idx]
                    dx, dy = res
                    graph[u].append((v, dx, dy))
                    graph[v].append((u, -dx, -dy))
                    success += 1
            self.log(f"Matches found: {success}/{len(tasks)}")

            coords = {}
            visited = set()
            components = []
            unvisited = set(local_grid.keys())

            while unvisited:
                start_node = min(unvisited)
                comp = []
                queue = [start_node]
                coords[start_node] = (0.0, 0.0)
                visited.add(start_node)
                unvisited.remove(start_node)

                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    cx, cy = coords[curr]
                    
                    for neighbor, dx, dy in graph[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            unvisited.remove(neighbor)
                            coords[neighbor] = (cx + dx, cy + dy)
                            queue.append(neighbor)
                components.append(comp)
            
            self.log(f"Tiles formed {len(components)} connected island(s).")

            self.log(f"--- PHASE 4: ASSEMBLY ---")
            sample_img = pyvips.Image.new_from_file(list(local_grid.values())[0])
            is_16bit = (sample_img.format == 'ushort' or sample_img.format == 'short')
            mosaic_fmt = 'ushort' if is_16bit else 'uchar'
            
            tile_w, tile_h = sample_img.width, sample_img.height
            min_r = min(r for r, c in local_grid.keys())
            min_c = min(c for r, c in local_grid.keys())
            
            naive_coords = {}
            stride_x, stride_y = tile_w * 0.9, tile_h * 0.9
            for (r,c) in local_grid.keys():
                naive_coords[(r,c)] = ((c - min_c) * stride_x, (r - min_r) * stride_y)
                
            final_coords = {}
            for i, comp in enumerate(components):
                anchor = comp[0]
                anchor_x, anchor_y = coords[anchor]
                offset_x = naive_coords[anchor][0] - anchor_x
                offset_y = naive_coords[anchor][1] - anchor_y
                
                for node in comp:
                    cx, cy = coords[node]
                    final_coords[node] = (cx + offset_x, cy + offset_y)
            coords = final_coords

            parts = src_dir.strip(os.sep).split(os.sep)
            folder_name = "_".join(parts[-3:]) if len(parts) >= 3 else "_".join(parts)
            out_dir = os.path.join(OUTPUT_BASE, folder_name)
            os.makedirs(out_dir, exist_ok=True)
            
            min_x = min(v[0] for v in coords.values()); min_y = min(v[1] for v in coords.values())
            
            vips_ff = None
            ff_mean = 1.0
            if has_ff:
                ff_path_local = os.path.join(TEMP_WORK_DIR, FLAT_FIELD_NAME)
                vips_ff = pyvips.Image.new_from_file(ff_path_local, access='random')
                vips_ff = vips_ff.cast("float")
                ff_mean = vips_ff.avg() 

            max_w, max_h = 0, 0
            for (r,c), f in local_grid.items():
                x, y = coords[(r,c)]
                v = pyvips.Image.new_from_file(f, access='random')
                max_w = max(max_w, (x - min_x) + v.width); max_h = max(max_h, (y - min_y) + v.height)

            mosaic = pyvips.Image.black(int(max_w)+100, int(max_h)+100, bands=1).cast(mosaic_fmt)
            self.progress["maximum"] = len(coords); self.progress["value"] = 0
            
            for i, (key) in enumerate(sorted(local_grid.keys())):
                r, c = key
                if i % 10 == 0: self.log(f"Stitching {i}/{len(coords)}...")
                
                x, y = coords[(r,c)]
                tile = pyvips.Image.new_from_file(local_grid[(r,c)], access='random')
                
                if vips_ff is not None:
                    tile = tile.cast("float")
                    tile = (tile / vips_ff) * ff_mean
                
                tile = tile.cast(mosaic_fmt)
                mosaic = mosaic.insert(tile, int(x-min_x), int(y-min_y))
                self.progress["value"] = i+1; self.root.update_idletasks()

            # Logic to handle conditional TIFF and ZARR exporting
            needs_tiff = self.use_bigtiff.get() or self.use_zarr.get()
            raw_path = os.path.join(out_dir, "stitched_raw_16bit.tiff")

            if needs_tiff:
                self.log("Saving RAW 16-bit BigTIFF (Wait...)...")
                comp_method = "lzw" if HAS_IMAGECODECS else "none"
                if not HAS_IMAGECODECS: self.log("Note: Saving Uncompressed (imagecodecs missing)")
                mosaic.write_to_file(raw_path, compression=comp_method, bigtiff=True)
                self.log(f"Saved Raw Data: {raw_path}")

            if self.use_zarr.get() and HAS_OME_ZARR:
                zarr_name = os.path.join(out_dir, "stitched.ome.zarr")
                self.log(f"Converting to OME-Zarr: {zarr_name}")
                self.convert_to_ome_zarr(raw_path, zarr_name)

            # Cleanup temporary TIFF if it was only needed for Zarr creation
            if needs_tiff and not self.use_bigtiff.get():
                self.log("Removing temporary BigTIFF...")
                try:
                    os.remove(raw_path)
                except Exception as e:
                    self.log(f"Could not remove temporary BigTIFF: {e}")
            
            # Unconditional DZI Generation
            self.log("Generating 8-bit Web Preview...")
            if is_16bit:
                mosaic_8bit = mosaic.cast("float") * (255.0 / 65535.0)
                mosaic_8bit = mosaic_8bit.cast("uchar")
            else:
                mosaic_8bit = mosaic.cast("uchar")
                
            dzi_name = "preview"
            mosaic_8bit.dzsave(os.path.join(out_dir, dzi_name), suffix='.jpg')
            
            self.write_html_offline(out_dir, dzi_name)
            self.write_launchers(out_dir)
            
            self.log("Done! Launching Preview..."); self.launch_browser(out_dir)
            self.btn_run.config(state=tk.NORMAL, text="COPY & STITCH")

        except Exception as e:
            self.log(f"Error: {e}"); print(e); self.btn_run.config(state=tk.NORMAL, text="COPY & STITCH")

    def write_html_offline(self, folder, name):
        html_content = f"""<!DOCTYPE html><html><head>
<meta charset="utf-8">
<title>MegaField Result</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"></script>
<style>
  body{{margin:0;background:#111;font-family:sans-serif;}}
  #viewer{{width:100vw;height:100vh;}}
  #scalebar {{
    position: absolute; bottom: 15px; left: 15px; z-index: 9999;
    background: rgba(255,255,255,0.9); padding: 4px 8px; border-radius: 4px;
    display: flex; flex-direction: column; align-items: center; pointer-events: none;
    box-shadow: 0 0 5px rgba(0,0,0,0.5);
  }}
  #sb-line {{ height: 4px; background: black; width: 100px; margin-top: 2px; }}
  #sb-text {{ color: black; font-weight: bold; font-size: 14px; font-family: monospace;}}
</style>
</head><body>
<div id="viewer"></div>
<div id="scalebar"><div id="sb-text">...</div><div id="sb-line"></div></div>
<script>
    var viewer = OpenSeadragon({{
        id: "viewer", prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
        tileSources: "{name}.dzi", showNavigator: true, animationTime: 0.5, blendTime: 0.1 
    }});

    var PIXEL_SIZE_NM = {PIXEL_SIZE_NM}; 
    var sbText = document.getElementById("sb-text");
    var sbLine = document.getElementById("sb-line");

    function updateScalebar() {{
        var zoom = viewer.viewport.getZoom(true);
        var imageZoom = viewer.viewport.viewportToImageZoom(zoom);
        var targetPixels = 200; 
        var nmPerPixel = PIXEL_SIZE_NM / imageZoom;
        var totalNm = targetPixels * nmPerPixel;

        var units = ["nm", "&mu;m", "mm"];
        var unitIndex = 0;
        
        if(totalNm > 1000) {{ totalNm /= 1000; unitIndex=1; }}
        if(totalNm > 1000) {{ totalNm /= 1000; unitIndex=2; }}
        
        var magnitude = Math.pow(10, Math.floor(Math.log10(totalNm)));
        var mantissa = totalNm / magnitude;
        var snap = 1;
        if (mantissa >= 2) snap = 2;
        if (mantissa >= 5) snap = 5;
        if (mantissa >= 10) snap = 10;
        
        var niceVal = snap * magnitude;
        
        var realNm = niceVal;
        if(unitIndex==1) realNm *= 1000;
        if(unitIndex==2) realNm *= 1000000;
        
        var finalPixels = realNm / nmPerPixel;
        
        sbLine.style.width = finalPixels + "px";
        sbText.innerHTML = niceVal + " " + units[unitIndex];
    }}

    viewer.addHandler('open', updateScalebar);
    viewer.addHandler('animation', updateScalebar);
    viewer.addHandler('resize', updateScalebar);
</script></body></html>"""
        
        with open(os.path.join(folder, "index.html"), "w", encoding="utf-8") as f:
            f.write(html_content)

    def write_launchers(self, folder):
        sh_path = os.path.join(folder, "view_result.sh")
        with open(sh_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write('cd "$(dirname "$0")"\n')
            f.write('python3 -c "\n')
            f.write("import http.server, socket, socketserver, webbrowser, threading, time, sys\n")
            f.write("def start_server():\n")
            f.write("    with socketserver.TCPServer(('localhost', 0), http.server.SimpleHTTPRequestHandler) as httpd:\n")
            f.write("        port = httpd.server_address[1]\n")
            f.write("        print(f'Serving on port {port}')\n")
            f.write("        def open_browser():\n")
            f.write("            time.sleep(1)\n")
            f.write("            webbrowser.open(f'http://localhost:{port}/index.html')\n")
            f.write("        threading.Thread(target=open_browser).start()\n")
            f.write("        try: httpd.serve_forever()\n")
            f.write("        except KeyboardInterrupt: sys.exit(0)\n")
            f.write("start_server()\n")
            f.write('"\n')
        
        os.chmod(sh_path, 0o755)

        dt_path = os.path.join(folder, "View_Mosaic.desktop")
        with open(dt_path, "w") as f:
            f.write("[Desktop Entry]\n")
            f.write("Type=Application\n")
            f.write("Name=View Mosaic\n")
            f.write(f"Exec={sh_path}\n")
            f.write("Icon=image-x-generic\n")
            f.write("Terminal=true\n")
        os.chmod(dt_path, 0o755)

        bat_path = os.path.join(folder, "view_result.bat")
        with open(bat_path, "w") as f:
            f.write("@echo off\n")
            f.write('cd /d "%~dp0"\n')
            f.write('python -c "import http.server, socket, webbrowser, threading, time, sys; server = http.server.HTTPServer((\'localhost\', 0), http.server.SimpleHTTPRequestHandler); port = server.server_address[1]; print(f\'Serving on port {port}...\'); threading.Timer(1, lambda: webbrowser.open(f\'http://localhost:{port}/index.html\')).start(); server.serve_forever()"\n')
            f.write("pause\n")
        
        try:
            subprocess.run(["gio", "set", dt_path, "metadata::trusted", "yes"], 
                           stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        except:
            pass
        
        self.log(f"Created launchers in {folder}")

    def launch_browser(self, folder):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
        
        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=folder, **kwargs)
        
        def run_server():
            httpd = HTTPServer(('localhost', port), Handler)
            httpd.serve_forever()
            
        t = threading.Thread(target=run_server, daemon=True)
        t.start()
        time.sleep(1)
        webbrowser.open(f"http://localhost:{port}/index.html")

APP_ICON_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAF10lEQVR4AeXBoYGFMBRE0Xn6"
    "+/SAQWPQGBwN0EY6SBs0gMOgMWgMPeCxs7cH4vacMPSPhaF/LAwh9Yu+eo45DKGdNtVwrWMY"
    "wq/JquG9SxhCGELqF331HHMYQjttquFaxzCEX5NVw3uXMIQwhNQv+uo55jCEdtpUw7WOYQi/"
    "JquG9y5hCGEIqV/01XPMYQjttKmGax3DEH5NVg3vXcIQwhBSv+ir55jDENppUw3XOoYh/Jqs"
    "Gt67hCGEIaR+0VfPMYchtNOmGq51DEP4NVk1vHcJQwhDSP2ir55jDkNop001XOsYhvBrsmp4"
    "7xKGEIaQ+kVfPccchtBOm2q41jEM4ddk1fDeJQwhDCH1i756jjkMoZ021XCtYxjCr8mq4b1L"
    "GEIYQuoXffUccxhCO22q4VrHMIRfk1XDe5cwhDCE1C/66jnmMIR22lTDtY5hCL8mq4b3LmEI"
    "YQipX/TVc8xhCO20qYZrHcMQfk1WDe9dwhDCEFK/6KvnmMMQ2mlTDdc6hiH8mqwa3ruEIYQh"
    "pH7RV88xhyG006YarnUMQ/g1WTW8dwlDCENI/aKvnmMOQ2inTTVc6xiG8GuyanjvEoYQhpD6"
    "RV89xxyG0E6barjWMQzh12TV8N4lDCEMIfWLvnqOOQyhnTbVcK1jGMKvyarhvUsYQhhCQBUY"
    "QkCVGEJAlRhCGPrHwtA/FoYw5FNf7aULQ0j9ohqeYw5DaKdNNVzrGIYQhjDkU1/tpQtDSP2i"
    "Gp5jDkNop001XOsYhhCGMORTX+2lC0NI/aIanmMOQ2inTTVc6xiGEIYw5FNf7aULQ0j9ohqe"
    "Yw5DaKdNNVzrGIYQhjDkU1/tpQtDSP2iGp5jDkNop001XOsYhhCGMORTX+2lC0NI/aIanmMO"
    "Q2inTTVc6xiGEIYw5FNf7aULQ0j9ohqeYw5DaKdNNVzrGIYQhjDkU1/tpQtDSP2iGp5jDkNo"
    "p001XOsYhhCGMORTX+2lC0NI/aIanmMOQ2inTTVc6xiGEIYw5FNf7aULQ0j9ohqeYw5DaKdN"
    "NVzrGIYQhjDkU1/tpQtDSP2iGp5jDkNop001XOsYhhCGMORTX+2lC0NI/aIanmMOQ2inTTVc"
    "6xiGEIYw5FNf7aULQ0j9ohqeYw5DaKdNNVzrGIYQhjDkU1/tpQtDSP2iGp5jDkNop001XOsY"
    "hhCGMORTX+2lC0NI/aIanmMOQ2inTTVc6xiGEIYw5FNf7aULQ0j9ohqeYw5DaKdNNVzrGIYQ"
    "hjDkU1/tpQtDSP2iGp5jDkNop001XOsYhhCGMORTX+2lC0NI/aIanmMOQ2inTTVc6xiGEIYQ"
    "UAWGEFAlhhBQJYYQhv6xMPSPhSH8mqyv3ruEIQz5VA176cIQUr+ohueYwxDCEH5N1lfvXcIQ"
    "hnyqhr10YQipX1TDc8xhCGEIvybrq/cuYQhDPlXDXrowhNQvquE55jCEMIRfk/XVe5cwhCGf"
    "qmEvXRhC6hfV8BxzGEIYwq/J+uq9SxjCkE/VsJcuDCH1i2p4jjkMIQzh12R99d4lDGHIp2r"
    "YSxeGkPpFNTzHHIYQhvBrsr567xKGMORTNeylC0NI/aIanmMOQwhD+DVZX713CUMY8qka9tKF"
    "IaR+UQ3PMYchhCH8mqyv3ruEIQz5VA176cIQUr+ohueYwxDCEH5N1lfvXcIQhnyqhr10YQip"
    "X1TDc8xhCGEIvybrq/cuYQhDPlXDXrowhNQvquE55jCEMIRfk/XVe5cwhCGfqmEvXRhC6hfV"
    "8BxzGEIYwq/J+uq9SxjCkE/VsJcuDCH1i2p4jjkMIQzh12R99d4lDGHIp2rYSxeGkPpFNTz"
    "HHIYQhvBrsr567xKGMORTNeylC0NI/aIanmMOQwhD+DVZX713CUMY8qka9tKFIaR+UQ3PMYch"
    "hCH8mqyv3ruEIQz5VA176cIQUr+ohueYwxDCEH5N1lfvXcIQhnyqhr10YQipX1TDc8xhCGEI"
    "AVVgCAFVYggBVWIIYegfC0P/2B9hH3yMgYZnwwAAAABJRU5ErkJggg=="
)

if __name__ == "__main__":
    root = tk.Tk()
    try:
        icon_data = base64.b64decode(APP_ICON_B64)
        icon_img = tk.PhotoImage(data=icon_data)
        root.iconphoto(True, icon_img)
    except Exception:
        pass
    app = StitcherApp(root)
    root.mainloop()
