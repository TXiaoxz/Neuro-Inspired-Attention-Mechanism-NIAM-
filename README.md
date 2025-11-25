# Neuro-Inspired Attention for Real-Time Audio Enhancement

**Course**: Machine Learning for Signal Processing (MLSP)

**Author**: Xupeng Zhang, Yuxi Zheng, Yunqi Liu, Xinyao Ye

**Institution**: Johns Hopkins University

---

##  Project Overview

### Problem Statement
In noisy environments (e.g., crowded streets, cafes, multiple speakers), it's difficult to focus on a target speaker's voice. While the human auditory system naturally implements selective attention (the "cocktail party effect"), current audio processing systems struggle with this task.

### Solution
We propose a **Neuro-Inspired Attention Mechanism (NIAM)** that enables models to selectively enhance target audio while suppressing background noise and interfering speakers - not by completely removing them, but by **adjusting their relative volumes** to maintain natural spatial awareness.

### Goals
1. Build CNN/RNN/Transformer baseline models for audio enhancement
2. Implement NIAM-integrated Transformer for selective audio enhancement
3. Evaluate enhancement quality on noisy speech data (SNR improvements)
4. Implement a real-time audio processing demo

---

##  Key Achievements

**Four Models Trained & Evaluated**
- CNN (12.5M params): +5.44 dB SI-SNR improvement
- RNN (3.3M params): +8.78 dB SI-SNR improvement
- Transformer (15M params): +10.54 dB SI-SNR improvement
- **NIAM v2 Refined (15M params): -17.50 dB SI-SNR best validation loss**

**NIAM v2 Implementation**
- Neuro-inspired attention with 4 specialized modules (Selective, Frequency, Temporal, Noise Adaptation)
- Learnable module weights: [0.307, 0.044, 0.003, 0.500]
- Residual refinement mode with soft thresholding
- Trained on 50,000 clean speech samples with cocktail party noise

**Speech Recognition Performance**
- Whisper ASR evaluation: 5.6% WER improvement (53.3% to 50.3%)
- Comprehensive evaluation on cocktail party scenarios
- Significant perceptual quality improvements

**Interactive Web Demo**
- Real-time audio enhancement with Gradio interface
- Upload audio or record from microphone
- Add synthetic cocktail party noise
- Visual waveform comparison

**Cocktail Party Augmentation System**
- Realistic multi-speaker noise simulation (5 interferers)
- Hybrid augmentation: 70% cocktail party + 30% traditional noise
- Significantly more challenging than standard denoising

**Key Insights**
- NIAM modules learn specialized roles (Noise Adaptation dominates at 50%)
- Residual refinement provides temporal smoothness
- Self-attention with neuro-inspired mechanisms excels at source separation

---

##  Research Approach

### NIAM Module - Four Biological Hearing Mechanisms

1. **Selective Attention** - Enhance specific sound sources, suppress others
2. **Frequency Tuning** - Dynamic attention to specific frequency bands
3. **Temporal Focus** - Adaptive attention window adjustment
4. **Noise Adaptation** - Environment-aware noise suppression

NIAM integrates into Transformer's Multi-Head Attention module to learn which audio regions to enhance vs. suppress.

---

##  Experimental Architecture

### 1. Baseline Models
- **CNN**: Extract local time-frequency features
- **RNN (GRU)**: Model temporal dependencies
- **Transformer**: Standard self-attention mechanism

### 2. NIAM-Transformer
NIAM module integrated into Transformer encoder attention layers.

### 3. Task: Audio Enhancement
```
Input:  Noisy audio (target speaker + background noise)
Output: Enhanced audio (target speaker amplified, noise suppressed)
```

### 4. Dataset
**People's Speech** (MLCommons) - Large-scale English speech dataset
- Clean speech for training
- Synthesized noisy variants at different SNR levels

---

##  Project Structure

```
MLSP_Project/
├── README.md                        # This file
├── MIGRATION_GUIDE.md               # Guide for migrating project to different machines
├── config.py                        # Central configuration (paths management)
├── requirements.txt                 # Python dependencies
├── PROJECT_STRUCTURE.md             # Detailed structure documentation
│
├── demo_app.py                      # Interactive web demo (Gradio)
├── train_niam_v2.py                 # NIAM v2 training script
├── train_fast.py                    # CNN training script
├── train_rnn.py                     # RNN (GRU) training script
├── train_transformer.py             # Transformer training script
├── train_transformer_v2.py          # Transformer v2 training script
│
├── inference_niam_v2.py             # NIAM v2 inference
├── inference.py                     # Audio enhancement inference
├── evaluate.py                      # Single model evaluation
├── evaluate_comprehensive.py        # Comprehensive evaluation with Whisper ASR
├── evaluate_whisper.py              # Whisper ASR evaluation
├── compare_models.py                # Compare all models with visualizations
│
├── niam_v2.py                       # NIAM v2 module implementation
├── generate_demo_noise.py           # Generate demo noise files
├── create_audio_comparison.py       # Create audio comparison pages
│
├── src/                             # Core source code
│   ├── data/
│   │   └── cocktail_augmentor.py   # Cocktail party augmentation
│   ├── models/                      # Model architectures
│   ├── training/                    # Training utilities
│   └── utils/                       # Helper functions
│       ├── audio_utils.py          # Audio processing utilities
│       └── noise_generation.py     # Noise generation utilities
│
├── data/                            # Dataset storage (configured in config.py)
│   └── [External - not in repo]    # Specified by DATA_DIR in config.py
│
├── checkpoints/                     # Trained model weights
│   ├── cnn_enhancer_best.pt        # CNN model (12.5M params)
│   ├── rnn_enhancer_best.pt        # RNN model (3.3M params)
│   └── transformer_enhancer_best.pt # Transformer model (~15M params)
│
└── results/                         # Experimental results & visualizations
    ├── niam_v2_refined/
    │   ├── checkpoints/
    │   │   └── transformer_niam_v2_refined_best.pt  # Best NIAM v2 model
    │   ├── test_samples/            # Generated test audio samples
    │   ├── demo_noise_30s.wav       # Fixed demo noise file
    │   └── comprehensive_eval/      # Comprehensive evaluation results
    ├── si_snr_comparison_cocktail.png
    └── all_metrics_comparison_cocktail.png
```

---

##  Audio Enhancement Pipeline

### 1. Feature Extraction
- **Mel-Spectrogram**: 80 mel bins, 25ms window, 10ms hop
- Represents time-frequency structure of audio

### 2. Noise Augmentation
- **Noise Types**: White noise, pink noise, brown noise
- **SNR Levels**: -5dB, 0dB, 5dB, 10dB, 15dB, 20dB, Clean
- Simulates real-world noisy conditions

### 3. Model Processing
```
Noisy Mel-Spec → CNN/RNN/Transformer → Attention Weights → Enhanced Mel-Spec
```

### 4. Audio Reconstruction
- **Griffin-Lim Algorithm**: Reconstruct audio from enhanced mel-spectrogram
- Future: Neural vocoder (HiFi-GAN, MelGAN) for higher quality

### 5. Evaluation Metrics
- **SI-SNR** (Scale-Invariant Signal-to-Noise Ratio): Audio quality improvement
- **PESQ** (Perceptual Evaluation of Speech Quality): Subjective quality
- **Listening Tests**: Subjective evaluation

---

##  Getting Started

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd MLSP_Project

# Create and activate environment
conda create -n mlsp_project python=3.10
conda activate mlsp_project

# Install dependencies
pip install -r requirements.txt
```

### Configuration

The project uses a central configuration file (`config.py`) for path management:

```bash
# Verify current configuration
python config.py
```

**Important**: If migrating to a new machine, edit `config.py` line 27 to update `DATA_DIR`:
```python
DATA_DIR = '/path/to/your/dataset'  # Update this for your machine
```

For detailed migration instructions, see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

### Quick Start: Run Demo

The easiest way to try the project is to run the interactive demo:

```bash
# Launch web demo (no dataset needed - all files included)
python demo_app.py
```

Then open `http://localhost:7860` in your browser to:
- Upload audio files or record from microphone
- Add synthetic cocktail party noise
- Enhance audio with NIAM v2 model
- Compare original vs enhanced audio

### Training Models

**Note**: Training requires the People's Speech dataset. Configure `DATA_DIR` in `config.py` first.

```bash
# Train NIAM v2 (our best model)
python train_niam_v2.py

# Train baseline models
# Train CNN (fastest: ~1.5 it/s, ~52 min for 15 epochs)
python train_fast.py

# Train RNN (fast: ~14 it/s, ~5.5 min for 15 epochs)
python train_rnn.py

# Train Transformer (very fast: ~60 it/s, ~1.3 min for 15 epochs)
python train_transformer.py
```

Each script supports cocktail party or traditional noise augmentation via `USE_COCKTAIL_PARTY` flag.

### Evaluation

```bash
# Comprehensive evaluation with Whisper ASR
python evaluate_comprehensive.py

# Evaluate single baseline model
python evaluate.py -t cnn --cocktail -n 500
python evaluate.py -t rnn --cocktail -n 500
python evaluate.py -t transformer --cocktail -n 500

# Compare all baseline models with visualizations
python compare_models.py --cocktail -n 500
```

### Inference

```bash
# NIAM v2 inference (generates test samples)
python inference_niam_v2.py -m results/niam_v2_refined/checkpoints/transformer_niam_v2_refined_best.pt -n 10

# Baseline model inference
python inference.py input.wav -o output_enhanced.wav -m checkpoints/transformer_enhancer_best.pt
```

---

##  Experimental Results

### Model Performance Comparison

All models evaluated on **Cocktail Party** scenario (5 interfering speakers, 20-50% volume each).

#### SI-SNR Performance

| Model | Parameters | Training Speed | SI-SNR (Validation) | WER (Whisper) |
|-------|-----------|----------------|---------------------|---------------|
| **CNN** | 12.5M | 1.5 it/s | +5.44 dB | N/A |
| **RNN (GRU)** | 3.3M | 14.2 it/s | +8.78 dB | N/A |
| **Transformer** | ~15M | 60 it/s | +10.54 dB | N/A |
| **NIAM v2 Refined** | ~15M | ~60 it/s | **-17.50 dB (best)** | **50.3% (improved from 53.3%)** |

#### Key Findings

1. **NIAM v2 Refined achieves best performance**
   - Best validation loss: -17.50 dB SI-SNR
   - 5.6% Word Error Rate improvement (53.3% to 50.3%) on Whisper ASR
   - Learnable module weights show specialization: [0.307, 0.044, 0.003, 0.500]
   - Noise Adaptation module dominates (50%), showing learned noise suppression priority

2. **Baseline Transformer performs well**
   - 94% better than CNN baseline (+10.54 dB vs +5.44 dB)
   - 20% better than RNN (+10.54 dB vs +8.78 dB)
   - Self-attention excels at cocktail party source separation

3. **Training Efficiency**
   - Transformer & NIAM v2: Fastest training (~60 it/s)
   - RNN: Fast (14.2 it/s)
   - CNN: Slowest (1.5 it/s)
   - Parallelization advantage of Transformer over sequential RNN

4. **Architecture Insights**
   - **CNN**: Good at local time-frequency features, struggles with global context
   - **RNN**: Temporal modeling helps, but sequential dependency limits speed
   - **Transformer**: Global attention + full parallelization
   - **NIAM v2**: Adds neuro-inspired attention for specialized audio processing

### NIAM v2 Module Weights

The learned weights show specialization of different attention mechanisms:
- Selective Attention: 30.7%
- Frequency Tuning: 4.4%
- Temporal Focus: 0.3%
- Noise Adaptation: 50.0%

This indicates the model learns to prioritize noise suppression over other mechanisms.

---

## Current Progress

### Phase 1: Baseline Models  **COMPLETED**
- [x] Data loading and preprocessing
- [x] Noise augmentation pipeline (traditional + cocktail party)
- [x] Feature extraction (STFT/Mel-Spectrogram)
- [x] Cocktail party augmentation system (5 interferers, hybrid mode)
- [x] CNN baseline model training (+5.44 dB SI-SNR)
- [x] RNN (Bi-GRU) baseline training (+8.78 dB SI-SNR)
- [x] Transformer baseline training (+10.54 dB SI-SNR)
- [x] Comprehensive evaluation framework
- [x] Multi-model comparison with visualizations

### Phase 2: NIAM Development  **COMPLETED**
- [x] Design NIAM attention mechanism
  - [x] Selective attention module
  - [x] Frequency tuning layer
  - [x] Temporal focus mechanism
  - [x] Noise adaptation controller
- [x] Integrate NIAM into Transformer
- [x] Implement learnable module weights with soft thresholding
- [x] Add residual refinement mode
- [x] Train NIAM v2 on 50,000 samples
- [x] Achieve -17.50 dB SI-SNR best validation loss
- [x] Comprehensive evaluation with Whisper ASR (5.6% WER improvement)

### Phase 3: Interactive Demo  **COMPLETED**
- [x] Build Gradio web interface
- [x] Implement real-time audio enhancement
- [x] Add microphone recording support
- [x] Generate demo noise files
- [x] Create audio comparison visualizations
- [x] Deploy demo application

### Phase 4: Project Portability  **COMPLETED**
- [x] Create central configuration system (config.py)
- [x] Update all scripts to use relative paths
- [x] Write comprehensive migration guide
- [x] Package demo files with project

---

##  Technical Stack

- **Deep Learning**: PyTorch 2.0+, torchaudio
- **Audio Processing**: librosa, soundfile, STFT/iSTFT
- **Data**: Hugging Face Datasets (People's Speech - MLCommons)
- **Evaluation**: SI-SNR, SNR, MSE, Whisper ASR (WER)
- **Demo**: Gradio web interface
- **Visualization**: matplotlib, seaborn
- **Optimization**: Mixed precision training (AMP), gradient clipping
- **Configuration**: Central path management (config.py)

---

##  Implementation Details

### Model Architectures

#### CNN Enhancer (UNet-style)
- 4-layer encoder-decoder with skip connections
- 2D convolutions on mel-spectrograms
- Batch normalization + ReLU activation
- Parameters: 12.5M

#### RNN Enhancer (Bi-GRU)
- 3-layer bidirectional GRU
- Layer normalization
- Temporal sequence modeling
- Parameters: 3.3M

#### Transformer Enhancer
- 4-layer encoder with 8 attention heads
- Positional encoding for temporal information
- GELU activation, pre-layer normalization
- Feed-forward dimension: 1024
- Parameters: ~15M

#### NIAM v2 Transformer Enhancer
- Based on Transformer architecture
- Integrated NIAM module with 4 attention mechanisms:
  - Selective Attention (channel-wise)
  - Frequency Tuning (spectral weighting)
  - Temporal Focus (time-domain attention)
  - Noise Adaptation (learned noise suppression)
- Learnable module weights with soft thresholding
- Residual refinement mode (alpha=0.2)
- Parameters: ~15M

### Training Configuration

#### Baseline Models (CNN, RNN, Transformer)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **Scheduler**: CosineAnnealingLR (T_max=15)
- **Loss**: Combined SI-SNR + 0.1×MSE
- **Batch size**: 16
- **Epochs**: 15
- **Data**: 5,000 train, 500 validation samples
- **Augmentation**: 70% cocktail party + 30% traditional noise

#### NIAM v2 Model
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **Scheduler**: CosineAnnealingLR (T_max=15)
- **Loss**: SI-SNR (primary) + 0.1×MSE (auxiliary)
- **Batch size**: 16
- **Epochs**: 15
- **Data**: 50,000 train, 5,000 validation samples (clean_sa split)
- **Augmentation**: Cocktail party noise (5 interferers, SNR=8dB)

### Cocktail Party Augmentation
- Noise pool: 20% of dataset (3,724 samples)
- Interferers per sample: 5 speakers
- Volume range: 20-50% per interferer
- Simulates realistic multi-speaker environments

---

##  References

1. Vaswani et al. (2017) - Attention is All You Need
2. Luo & Mesgarani (2019) - Conv-TasNet for Speech Separation
3. Pandey & Wang (2019) - TCNN for Monaural Speech Enhancement
4. People's Speech Dataset - MLCommons
5. Choi et al. (2019) - Phase-aware Speech Enhancement with Deep Complex U-Net

---

##  Project Files

### Configuration
- `config.py` - Central path configuration (edit DATA_DIR when migrating)
- `MIGRATION_GUIDE.md` - Guide for migrating project across machines

### Demo
- `demo_app.py` - Interactive web demo (Gradio)
- `generate_demo_noise.py` - Generate demo noise files

### Training Scripts
- `train_niam_v2.py` - NIAM v2 model training
- `train_fast.py` - CNN model training
- `train_rnn.py` - RNN (GRU) model training
- `train_transformer.py` - Transformer model training
- `train_transformer_v2.py` - Transformer v2 model training

### Evaluation Scripts
- `evaluate_comprehensive.py` - Comprehensive evaluation with Whisper ASR
- `evaluate_whisper.py` - Whisper ASR evaluation
- `evaluate.py` - Evaluate single baseline model
- `compare_models.py` - Compare all baseline models with visualizations

### Inference Scripts
- `inference_niam_v2.py` - NIAM v2 inference (generate test samples)
- `inference.py` - Baseline model audio enhancement
- `inference_transformer.py` - Transformer inference

### Core Modules
- `niam_v2.py` - NIAM v2 module implementation
- `src/data/cocktail_augmentor.py` - Multi-speaker augmentation
- `src/utils/audio_utils.py` - Audio processing utilities
- `src/utils/noise_generation.py` - Noise generation utilities

---

##  Academic Context

This project was developed as part of the Machine Learning for Signal Processing (MLSP) course at Johns Hopkins University. The work demonstrates the application of modern deep learning architectures (CNN, RNN, Transformer) to the classic "cocktail party problem" in audio signal processing.

**Author**: Xupeng Zhang, Yuxi Zheng, Yunqi Liu, Xinyao Ye

**Institution**: Johns Hopkins University

**Course**: Machine Learning for Signal Processing (MLSP)

**Last Updated**: November 25, 2024
