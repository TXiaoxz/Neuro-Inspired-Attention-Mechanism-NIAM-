# Migration Guide

## Migrating from Server to Laptop

This project has been optimized with a **hybrid path management approach**:
- **Project file paths**: Auto-adaptive (using relative paths)
- **Dataset paths**: Manual configuration (one-time setup)

---

## Quick Migration Steps

### 1. Clone Project to Your Laptop

```bash
# On your laptop
git clone <your-github-repo-url>
cd MLSP_Project
```

### 2. Update Dataset Path

**Edit line 27 in `config.py`:**

```python
# Change this line:
DATA_DIR = '/home/dlwlx05/project/NIAM/data'

# To your laptop's dataset path, for example:
# Windows:
DATA_DIR = 'D:/Datasets/NIAM_data'

# Mac:
DATA_DIR = '/Users/YourName/Datasets/NIAM_data'

# Linux:
DATA_DIR = '/home/yourname/datasets/NIAM_data'
```

### 3. Verify Configuration

```bash
python config.py
```

You'll see all configured paths. Verify they're correct.

### 4. Run Demo

```bash
python demo_app.py
```

Then open `http://localhost:7860` in your browser.

---

## Project Structure

```
MLSP_Project/
├── config.py                 # Central configuration file
├── demo_app.py              # Web demo (using relative paths)
├── train_niam_v2.py         # Training script (using config)
├── inference_niam_v2.py     # Inference script (using config)
├── results/                 # Results directory (follows project)
│   └── niam_v2_refined/
│       ├── checkpoints/     # Model files
│       ├── test_samples/    # Test audio samples
│       └── demo_noise_30s.wav
└── [External Dataset]       # Specified by DATA_DIR in config.py
```

---

## Files Required for Demo

**Good news: All demo files are included in the project!**

Included in project (tracked by Git):
- `results/niam_v2_refined/checkpoints/transformer_niam_v2_refined_best.pt` (24MB)
- `results/niam_v2_refined/demo_noise_30s.wav`
- `results/niam_v2_refined/test_samples/` (example audio files)

**If you only want to run the demo (no training)**, you don't need to download the full dataset!

---

## Configuration File (`config.py`)

### Auto-updated paths (no modification needed)
- `PROJECT_ROOT` - Project root directory (auto-detected)
- `RESULTS_DIR` - Results directory
- `CHECKPOINTS_DIR` - Model checkpoint directory
- `NIAM_V2_REFINED_MODEL` - Model file path

### Manually configured paths
- `DATA_DIR` - Dataset root directory **Change this when migrating!**
- `CLEAN_SA_CACHE` - Clean SA dataset cache
- `HF_CACHE_DIR` - HuggingFace cache directory

---

## Path Format for Different OS

### Windows
```python
DATA_DIR = 'D:/Datasets/NIAM_data'  # Recommended: forward slashes
# or
DATA_DIR = 'D:\\Datasets\\NIAM_data'  # Backslashes need escaping
```

### Mac / Linux
```python
DATA_DIR = '/Users/yourname/Datasets/NIAM_data'
```

---

## Files Updated with Relative Paths

The following files have been updated and don't require manual modification:

1. `config.py` - Central configuration file (newly created)
2. `demo_app.py` - Using config.py
3. `train_niam_v2.py` - Using config.py
4. `inference_niam_v2.py` - Using config.py

---

## Other Files

The following files still contain hard-coded paths but **don't affect demo**:

- `train_transformer.py`, `train_rnn.py` - Other training scripts
- `evaluate*.py` - Evaluation scripts
- `compare_*.py` - Comparison scripts
- `*.md` files - Documentation only

If you need to run these scripts, you can modify them following the `config.py` pattern.

---

## FAQ

### Q: Do I need to download the dataset to run the demo?
**A:** No! All files required for the demo are included in the project.

### Q: What if I want to train the model?
**A:** You'll need to download the MLCommons People's Speech dataset and configure the `DATA_DIR` path in `config.py`.

### Q: Should I use `/` or `\` in paths?
**A:** Use forward slashes `/` - Python handles OS differences automatically.

### Q: Can I store model files elsewhere?
**A:** Yes! Modify `CHECKPOINTS_DIR` in `config.py` to point to your model storage location.

---

## Migration Checklist

- [ ] Git clone project to laptop
- [ ] Edit `config.py` to update `DATA_DIR` (if training needed)
- [ ] Run `python config.py` to verify configuration
- [ ] Run `python demo_app.py` to test demo
- [ ] Visit `http://localhost:7860` to confirm web interface works

---

## Done!

Your project is now portable across different machines!

For issues, check the comments in `config.py`.
