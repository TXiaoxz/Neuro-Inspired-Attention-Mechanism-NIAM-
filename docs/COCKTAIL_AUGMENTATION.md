# Cocktail Party Effect Data Augmentation

## Overview

Implements a realistic cocktail party effect data augmentation method for training audio enhancement models. Compared to traditional white/pink/brown noise, this approach uses real multi-speaker mixing to simulate actual cocktail party scenarios.

## Implementation Details

### Core Logic

1. **Noise Pool (20% of Dataset)**
   - Randomly select 20% of samples from the entire dataset as noise sources
   - For 18,622 samples, the noise pool contains 3,724 samples

2. **Mixing Strategy**
   - Each training sample randomly selects 5 interfering speakers
   - Each interferer is mixed at 20-50% random volume into the target audio
   - Avoids self-mixing (samples are not mixed with themselves)

3. **Hybrid Mode (HybridAugmentor)**
   - 70% probability: use cocktail party noise (multi-speaker)
   - 30% probability: use traditional noise (white/pink/brown)
   - Provides more diverse training data

### SNR Characteristics

Based on statistics from 50 samples:
```
Mean SNR:   -9.53 dB
Std Dev:     7.71 dB
Range:      -22.66 to +11.63 dB
Median:     -11.08 dB
```

This is more challenging than traditional noise augmentation because the model needs to separate similar speech signals rather than simple noise.

## File Structure

```
MLSP_Project/
├── src/data/
│   └── cocktail_augmentor.py          # Core implementation
├── train_fast.py                      # Integrated with new augmentation
├── tests/test_cocktail_augmentor.py   # Testing and visualization script
└── results/
    ├── cocktail_test_clean.wav        # Test audio (clean)
    ├── cocktail_test_noisy.wav        # Test audio (mixed)
    ├── cocktail_comparison.png        # Waveform comparison plot
    └── snr_distribution.png           # SNR distribution histogram
```

## Usage

### 1. Enable Cocktail Party Augmentation During Training

Edit line 222 in `train_fast.py`:

```python
USE_COCKTAIL_PARTY = True  # Enable cocktail party augmentation
```

Or disable it (use traditional noise):

```python
USE_COCKTAIL_PARTY = False  # Use traditional white/pink/brown noise
```

### 2. Run Training

```bash
cd /home/dlwlx05/JHU_Course/MLSP/MLSP_Project
conda activate mlsp_project
python train_fast.py
```

Example output:
```
COCKTAIL PARTY AUGMENTATION ENABLED
======================================================================
[CocktailPartyAugmentor] Initialized:
  - Total dataset size: 18622
  - Noise pool size: 3724 (20%)
  - Interferers per sample: 5
  - Volume range: 20%-50%
[HybridAugmentor] Cocktail probability: 70%
```

### 3. Test Augmentation Effects

Run the test script to view visualizations and statistics:

```bash
cd tests
python test_cocktail_augmentor.py
```

This generates:
- Audio samples (clean vs noisy)
- Waveform comparison plots
- SNR distribution statistics

## Configurable Parameters

Modify `CocktailPartyAugmentor` initialization in `train_fast.py`:

```python
cocktail_aug = CocktailPartyAugmentor(
    dataset=dataset,
    noise_pool_ratio=0.2,      # Noise pool ratio (0.1-0.3)
    num_interferers=5,          # Number of interfering speakers (3-7)
    volume_range=(0.2, 0.5),   # Volume range (0.1-0.6)
    seed=42                     # Random seed
)
```

Modify `HybridAugmentor` initialization:

```python
noise_aug = HybridAugmentor(
    dataset=dataset,
    cocktail_augmentor=cocktail_aug,
    traditional_noise_types=['white', 'pink', 'brown'],
    cocktail_prob=0.7          # Cocktail party noise probability (0.0-1.0)
)
```

## Comparison of Advantages

### Traditional Noise Augmentation (White/Pink/Brown Noise)
- ✅ Simple and fast
- ✅ Low computational overhead
- ❌ Not realistic enough
- ❌ Cannot simulate multi-speaker scenarios

### Cocktail Party Augmentation (Multi-speaker Mixing)
- ✅ More realistic training scenarios
- ✅ Aligns with project goals ("multiple speakers")
- ✅ More challenging, better model learning
- ✅ Preserves spectral characteristics of speech
- ⚠️ Slightly increases data loading time

## Expected Effects

Models trained with cocktail party augmentation should be able to:

1. **Better separate multi-speaker audio** - Exposed to realistic multi-person mixing during training
2. **Stronger generalization** - Better performance in real-world scenarios
3. **Maintain spatial awareness** - Adjust volume ratios rather than completely removing background
4. **Better align with paper objectives** - NIAM for "cocktail party effect"

## Implementation Classes

### `CocktailPartyAugmentor`
- Core augmentor that implements multi-speaker mixing
- Methods:
  - `add_cocktail_noise()`: Add cocktail party noise
  - `compute_snr()`: Compute signal-to-noise ratio
  - `get_noise_sample()`: Get noise samples

### `HybridAugmentor`
- Hybrid augmentor combining traditional and cocktail party noise
- Methods:
  - `augment()`: Randomly select augmentation type
  - `add_traditional_noise()`: Add traditional noise

## Next Steps

- [ ] Comparative experiments: Train two models (traditional vs cocktail party), compare results
- [ ] Parameter tuning: Try different numbers of interferers and volume ranges
- [ ] Evaluation metrics: Calculate SI-SNR, PESQ improvements
- [ ] Visualization: Compare attention maps (see if model learns selective attention)

---

**Created**: 2025-11-05
**Status**: ✅ Completed and Integrated
