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

**Three Baseline Models Trained & Evaluated**
- CNN (12.5M params): +5.44 dB SI-SNR improvement
- RNN (3.3M params): +8.78 dB SI-SNR improvement
- Transformer (15M params): **+10.54 dB SI-SNR improvement** 

**Cocktail Party Augmentation System**
- Realistic multi-speaker noise simulation (5 interferers)
- Hybrid augmentation: 70% cocktail party + 30% traditional noise
- Significantly more challenging than standard denoising

**Comprehensive Evaluation Framework**
- Multi-model comparison with visualizations
- SI-SNR, SNR, and MSE metrics
- Automated performance analysis

**Key Insights**
- Transformer outperforms CNN by **94%** on cocktail party task
- Training speed: Transformer (60 it/s) >> RNN (14.2 it/s) >> CNN (1.5 it/s)
- Self-attention mechanism excels at source separation

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
├── requirements.txt                 # Python dependencies
├── PROJECT_STRUCTURE.md             # Detailed structure documentation
│
├── train_fast.py                    # CNN training script
├── train_rnn.py                     # RNN (GRU) training script
├── train_transformer.py             # Transformer training script
├── evaluate.py                      # Single model evaluation
├── compare_models.py                # Compare all models with visualizations
├── inference.py                     # Audio enhancement inference
│
├── src/                             # Core source code
│   ├── data/
│   │   └── cocktail_augmentor.py   # Cocktail party augmentation
│   ├── models/                      # Model architectures
│   ├── training/                    # Training utilities
│   └── utils/                       # Helper functions
│
├── data/                            # Dataset storage
│   └── cache/                       # Hugging Face dataset cache (6.5GB)
│
├── checkpoints/                     # Trained model weights
│   ├── cnn_enhancer_best.pt        # CNN model (12.5M params)
│   ├── rnn_enhancer_best.pt        # RNN model (3.3M params)
│   └── transformer_enhancer_best.pt # Transformer model (~15M params)
│
└── results/                         # Experimental results & visualizations
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
cd /home/dlwlx05/JHU_Course/MLSP/MLSP_Project

# Activate environment
conda activate mlsp_project

# Install dependencies
pip install torch torchaudio datasets soundfile matplotlib seaborn
```

### Training Models

```bash
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
# Evaluate single model
python evaluate.py -t cnn --cocktail -n 500
python evaluate.py -t rnn --cocktail -n 500
python evaluate.py -t transformer --cocktail -n 500

# Compare all models with visualizations
python compare_models.py --cocktail -n 500
```

### Inference

```bash
# Enhance audio file
python inference.py input.wav -o output_enhanced.wav -m checkpoints/transformer_enhancer_best.pt
```

---

##  Experimental Results

### Baseline Model Performance

All models evaluated on **Cocktail Party** scenario (5 interfering speakers, 20-50% volume each).

#### SI-SNR Improvement (Primary Metric)

| Model | Parameters | Training Speed | SI-SNR Improvement | SNR Improvement | MSE |
|-------|-----------|----------------|-------------------|-----------------|-----|
| **CNN** | 12.5M | 1.5 it/s | +5.44 dB | +4.00 dB | 46.38 |
| **RNN (GRU)** | 3.3M | 14.2 it/s | +8.78 dB | +1.52 dB | 77.01 |
| **Transformer**  | ~15M | 60 it/s | **+10.54 dB**  | +2.50 dB | 67.19 |

#### Key Findings

1. **Transformer achieves best SI-SNR improvement (+10.54 dB)**
   - 94% better than CNN baseline
   - 20% better than RNN
   - Self-attention excels at cocktail party source separation

2. **Training Efficiency**
   - Transformer: Fastest training (60 it/s, ~1.3 min)
   - RNN: Fast (14.2 it/s, ~5.5 min)
   - CNN: Slowest (1.5 it/s, ~52 min)
   - Parallelization advantage of Transformer over sequential RNN

3. **Architecture Insights**
   - **CNN**: Good at local time-frequency features, struggles with global context
   - **RNN**: Temporal modeling helps, but sequential dependency limits speed
   - **Transformer**: Best of both worlds - global attention + full parallelization

### Traditional Noise Performance

| Model | SI-SNR Improvement |
|-------|-------------------|
| CNN | +10.77 dB |
| RNN | TBD |
| Transformer | TBD |

*Traditional noise (white/pink/brown) is easier than cocktail party separation.*

### Next Steps: NIAM-Transformer

Target: **+12-15 dB SI-SNR improvement** on cocktail party scenario by adding:
- Selective Attention (enhance target, suppress interferers)
- Frequency Tuning (dynamic spectral weighting)
- Temporal Focus (adaptive time window)
- Noise Adaptation (environment-aware processing)

---

## 📝 Current Progress

### Phase 1: Baseline Models  **COMPLETED**
- [x] Data loading and preprocessing
- [x] Noise augmentation pipeline (traditional + cocktail party)
- [x] Feature extraction (Mel-Spectrogram)
- [x] Cocktail party augmentation system (5 interferers, hybrid mode)
- [x] CNN baseline model training (+5.44 dB SI-SNR)
- [x] RNN (Bi-GRU) baseline training (+8.78 dB SI-SNR)
- [x] Transformer baseline training (+10.54 dB SI-SNR) ⭐
- [x] Comprehensive evaluation framework
- [x] Multi-model comparison with visualizations

### Phase 2: NIAM Development  **IN PROGRESS**
- [ ] Design NIAM attention mechanism
  - [ ] Selective attention module
  - [ ] Frequency tuning layer
  - [ ] Temporal focus mechanism
  - [ ] Noise adaptation controller
- [ ] Integrate NIAM into Transformer
- [ ] Train and evaluate NIAM model
- [ ] Target: +12-15 dB SI-SNR improvement

### Phase 3: Real-time Demo  (Planned)
- [ ] Implement streaming interface
- [ ] Optimize inference speed
- [ ] Build interactive demo

---

##  Technical Stack

- **Deep Learning**: PyTorch 2.0+, torchaudio
- **Audio Processing**: librosa, soundfile, Griffin-Lim
- **Data**: Hugging Face Datasets (People's Speech)
- **Evaluation**: SI-SNR, SNR, MSE metrics
- **Visualization**: matplotlib, seaborn
- **Optimization**: Mixed precision training (AMP), gradient clipping

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

### Training Configuration
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **Scheduler**: CosineAnnealingLR (T_max=15)
- **Loss**: Combined SI-SNR + 0.1×MSE
- **Batch size**: 16
- **Epochs**: 15
- **Data**: 5000 train, 500 validation samples
- **Augmentation**: 70% cocktail party + 30% traditional noise

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

### Training Scripts
- `train_fast.py` - CNN model training
- `train_rnn.py` - RNN (GRU) model training
- `train_transformer.py` - Transformer model training

### Evaluation Scripts
- `evaluate.py` - Evaluate single model (supports CNN/RNN/Transformer)
- `compare_models.py` - Compare all models with visualizations
- `inference.py` - Audio enhancement on single files

### Core Modules
- `src/data/cocktail_augmentor.py` - Multi-speaker augmentation
- `src/utils/audio_utils.py` - Audio processing utilities

---

##  Academic Context

This project was developed as part of the Machine Learning for Signal Processing (MLSP) course at Johns Hopkins University. The work demonstrates the application of modern deep learning architectures (CNN, RNN, Transformer) to the classic "cocktail party problem" in audio signal processing.

**Author**: Xupeng Zhang, Yuxi Zheng, Yunqi Liu, Xinyao Ye

**Institution**: Johns Hopkins University

**Course**: Machine Learning for Signal Processing (MLSP)

**Last Updated**: November 9, 2025
