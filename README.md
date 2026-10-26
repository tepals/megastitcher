# MegaField Stitcher & Viewer

Installation guide for the **MegaField Stitcher** (processes your images) and the **MegaViewer** (views the results with a 4nm scale bar).

Follow the steps **in order** and don't skip any. Copy each command exactly as written (one line = one command), then press Enter.

## Prerequisites

1. A Linux computer (Ubuntu/Debian recommended).
2. Python 3 installed. Check with:

   ```bash
   python3 --version
   ```

   You should see something like `Python 3.12.3`. If you get an error, run `sudo apt-get install python3 -y`.
3. Internet access (for setup only).
4. These files together in **one folder** (called the "megastitcher folder" below):
   - `megastitcher.py`
   - `megastitcher.sh`
   - `README.md` (this file)

## Step 0: Open a terminal inside the megastitcher folder

Every command below must be run from inside the megastitcher folder.

**Easiest way:** open the folder in the Files app, right-click an empty area, and choose **Open in Terminal**.

**Or** type this (replace the path with where *your* folder is):

```bash
cd ~/development/megastitcher
```

Check you are in the right place:

```bash
ls
```

You must see `megastitcher.py` and `megastitcher.sh` in the list. If you don't, **stop**: you are in the wrong folder or a file is missing.

## Step 1: Install system libraries (one time only)

```bash
sudo apt-get update
sudo apt-get install python3-venv python3-tk libvips libvips-dev -y
```

> It will ask for your password. Nothing shows while you type it. That is normal. Type it and press Enter.

## Step 2: Create the Python environment (one time only)

Run these one at a time, still inside the megastitcher folder:

```bash
python3 -m venv stitcher_env
source stitcher_env/bin/activate
pip install --upgrade pip
pip install numpy opencv-python tifffile pyvips zarr dask dask-image ome-zarr imagecodecs
```

- After `source ...activate`, your prompt should start with `(stitcher_env)`. If it doesn't, something went wrong, so redo this step.
- The last command can take several minutes. Wait until you get your prompt back.
- The environment **must** be called `stitcher_env` and live in the megastitcher folder, because `megastitcher.sh` looks for it there.

## Step 3: Make the files runnable (one time only)

```bash
chmod +x megastitcher.py megastitcher.sh
```

No `sudo` is needed.

**Optional, for double-clicking `megastitcher.sh`:** open Files → Preferences → Behavior → Executable Text Files and choose **Ask what to do** (or **Run**). Then double-click `megastitcher.sh` and choose **Run**.

## How to use

**Option A (recommended, always works):** from a terminal inside the megastitcher folder, run:

```bash
./megastitcher.sh
```

**Option B:** double-click `megastitcher.sh` (see the optional note in Step 3).

Then in the program window:

**Browse Folder** → **Enter Coords** → **Copy & Stitch**

When it finishes, the terminal says `Closing in 5 seconds...`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Permission denied` | Redo Step 3. |
| `No such file or directory: stitcher_env/bin/activate` | You are in the wrong folder (Step 0) or skipped Step 2. |
| `can't open file ... megastitcher.py` | `megastitcher.py` is not in the same folder as `megastitcher.sh`. |
| `ModuleNotFoundError: No module named ...` | Redo the `pip install` line in Step 2, with `(stitcher_env)` shown in the prompt. |
| Double-click does nothing | Use Option A instead. |
