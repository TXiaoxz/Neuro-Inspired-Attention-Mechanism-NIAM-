# Project Structure

**Last Updated**: 2025-11-05

## 📁 Directory Layout

```
MLSP_Project/
├── train_fast.py                    # ⭐ Main training script
├── README.md                        # Project overview
├── requirements.txt                 # Python dependencies
├── setup_env.sh                     # Environment setup script
│
├── src/                             # 💾 Core source code
│   ├── data/
│   │   └── cocktail_augmentor.py   # Cocktail party augmentation
│   ├── models/                      # Model architectures
│   ├── training/                    # Training utilities
│   └── utils/                       # Helper functions
│
├── experiments/                     # 📓 Experimental notebooks
│   ├── project.ipynb               # Main experiment notebook
│   └── audio_enhancement_baseline.ipynb
│
├── tests/                           # 🧪 Test scripts
│   └── test_cocktail_augmentor.py  # Augmentor tests & visualization
│
├── scripts/                         # 🔧 Utility scripts
│   └── debug_training.py           # Training pipeline debugger
│
├── docs/                            # 📄 Documentation
│   └── COCKTAIL_AUGMENTATION.md    # Cocktail party augmentation docs
│
├── data/                            # 💾 Dataset storage
│   ├── cache/                      # Hugging Face dataset cache (6.5GB)
│   ├── metadata/                   # Dataset metadata
│   ├── noisy/                      # Noisy audio samples
│   ├── processed/                  # Preprocessed data
│   └── raw/                        # Raw audio files
│
├── checkpoints/                     # 💾 Model checkpoints
│   └── *.pt                        # Saved model weights
│
├── results/                         # 📊 Experimental results
│   ├── cocktail_test_clean.wav     # Test audio samples
│   ├── cocktail_test_noisy.wav
│   ├── cocktail_comparison.png     # Visualization plots
│   └── snr_distribution.png
│
├── configs/                         # ⚙️ Configuration files
└── notebooks/                       # 📓 Additional notebooks (empty)
```

## 🎯 Key Files

### Training & Inference
- **`train_fast.py`** - Main training script with cocktail party augmentation
  - Set `USE_COCKTAIL_PARTY = True/False` to toggle augmentation mode
  - Supports both traditional noise and multi-speaker augmentation

### Core Modules
- **`src/data/cocktail_augmentor.py`** - Multi-speaker noise augmentation
  - `CocktailPartyAugmentor`: 20% noise pool, 5 interferers, 20-50% volume
  - `HybridAugmentor`: 70% cocktail + 30% traditional noise

### Testing & Debugging
- **`tests/test_cocktail_augmentor.py`** - Test augmentor functionality
  - Run from tests directory: `cd tests && python test_cocktail_augmentor.py`
  - Generates audio samples and visualizations in `../results/`

- **`scripts/debug_training.py`** - Debug training pipeline
  - Tests each component: data loading, audio processing, model, training loop

### Documentation
- **`README.md`** - Project overview and goals
- **`docs/COCKTAIL_AUGMENTATION.md`** - Detailed augmentation documentation
- **`PROJECT_STRUCTURE.md`** - This file

## 🚀 Quick Start

### Training
```bash
# From project root
conda activate mlsp_project
python train_fast.py
```

### Testing Augmentor
```bash
# From tests directory
cd tests
python test_cocktail_augmentor.py
cd ..
```

### Debugging
```bash
# From scripts directory
cd scripts
python debug_training.py
cd ..
```

## 📝 Notes

- **Data loading**: On-the-fly augmentation (no preprocessing required)
- **Storage**: ~6.5GB for dataset cache, checkpoints vary by model size
- **Import paths**: All scripts use absolute paths for cross-directory imports
- **Results**: Generated outputs saved in `results/` directory

## 🔄 Recent Changes (2025-11-05)

- ✅ Reorganized project structure for better maintainability
- ✅ Moved notebooks to `experiments/`
- ✅ Moved test scripts to `tests/`
- ✅ Moved utility scripts to `scripts/`
- ✅ Moved documentation to `docs/`
- ✅ Updated all import paths and file references
- ✅ Verified all scripts still work correctly
