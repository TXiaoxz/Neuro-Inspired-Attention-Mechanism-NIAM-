"""
Test script for Cocktail Party Augmentor

This script tests the cocktail party augmentation functionality:
1. Loads a small subset of the dataset
2. Creates cocktail party noise
3. Visualizes and saves sample outputs
4. Computes SNR statistics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, Audio
import soundfile as sf
import sys
import os

# Add src to path
sys.path.insert(0, '/home/dlwlx05/JHU_Course/MLSP/MLSP_Project')

from src.data.cocktail_augmentor import CocktailPartyAugmentor, HybridAugmentor
from src.utils.audio_utils import decode_audio


def plot_comparison(clean_audio, noisy_audio, snr, save_path='test_cocktail.png'):
    """Plot clean vs noisy audio waveforms."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Clean audio
    axes[0].plot(clean_audio.numpy())
    axes[0].set_title('Clean Audio (Target Speaker)', fontsize=12)
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # Noisy audio
    axes[1].plot(noisy_audio.numpy())
    axes[1].set_title(f'Noisy Audio (Target + 5 Interferers) - SNR: {snr:.2f} dB', fontsize=12)
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    # Noise (difference)
    noise = noisy_audio - clean_audio
    axes[2].plot(noise.numpy())
    axes[2].set_title('Interfering Speakers (Noise)', fontsize=12)
    axes[2].set_xlabel('Sample')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {save_path}")
    plt.close()


def save_audio_samples(clean_audio, noisy_audio, sample_rate=16000, prefix='test_sample'):
    """Save clean and noisy audio as WAV files."""
    sf.write(f'{prefix}_clean.wav', clean_audio.numpy(), sample_rate)
    sf.write(f'{prefix}_noisy.wav', noisy_audio.numpy(), sample_rate)
    print(f"✓ Saved audio samples: {prefix}_clean.wav, {prefix}_noisy.wav")


def test_cocktail_augmentor():
    """Test CocktailPartyAugmentor functionality."""
    print("="*70)
    print("Testing Cocktail Party Augmentor")
    print("="*70)

    # Load small dataset
    print("\n[1/5] Loading dataset...")
    ds_val = load_dataset(
        "MLCommons/peoples_speech",
        "validation",
        cache_dir="./data/cache"
    )
    ds_val = ds_val.cast_column("audio", Audio(decode=False))
    dataset = ds_val['validation']

    # Create augmentor
    print("\n[2/5] Creating CocktailPartyAugmentor...")
    augmentor = CocktailPartyAugmentor(
        dataset=dataset,
        noise_pool_ratio=0.2,
        num_interferers=5,
        volume_range=(0.2, 0.5),
        seed=42
    )

    # Test on a sample
    print("\n[3/5] Testing augmentation on sample audio...")
    test_idx = 100
    clean_audio = augmentor._load_audio(test_idx)

    # Ensure fixed length for testing
    max_length = 160000  # 10 seconds at 16kHz
    if len(clean_audio) > max_length:
        clean_audio = clean_audio[:max_length]
    else:
        clean_audio = torch.nn.functional.pad(clean_audio, (0, max_length - len(clean_audio)))

    print(f"  Clean audio shape: {clean_audio.shape}")
    print(f"  Clean audio duration: {len(clean_audio)/16000:.2f} seconds")

    # Add cocktail noise
    noisy_audio, interferer_indices, interferer_volumes = augmentor.add_cocktail_noise(
        clean_audio,
        target_idx=test_idx
    )

    print(f"\n  Interferer details:")
    for i, (idx, vol) in enumerate(zip(interferer_indices, interferer_volumes)):
        print(f"    Interferer {i+1}: Index={idx}, Volume={vol*100:.1f}%")

    # Compute SNR
    snr = augmentor.compute_snr(clean_audio, noisy_audio)
    print(f"\n  Resulting SNR: {snr:.2f} dB")

    # Save samples
    print("\n[4/5] Saving audio samples...")
    save_audio_samples(clean_audio, noisy_audio, prefix='../results/cocktail_test')

    # Plot comparison
    print("\n[5/5] Creating visualization...")
    plot_comparison(clean_audio, noisy_audio, snr, save_path='../results/cocktail_comparison.png')

    print("\n" + "="*70)
    print("Test completed successfully!")
    print("="*70)


def test_hybrid_augmentor():
    """Test HybridAugmentor functionality."""
    print("\n\n")
    print("="*70)
    print("Testing Hybrid Augmentor")
    print("="*70)

    # Load dataset
    print("\n[1/3] Loading dataset...")
    ds_val = load_dataset(
        "MLCommons/peoples_speech",
        "validation",
        cache_dir="./data/cache"
    )
    ds_val = ds_val.cast_column("audio", Audio(decode=False))
    dataset = ds_val['validation']

    # Create augmentors
    print("\n[2/3] Creating HybridAugmentor...")
    cocktail_aug = CocktailPartyAugmentor(dataset, seed=42)
    hybrid_aug = HybridAugmentor(
        dataset,
        cocktail_aug,
        cocktail_prob=0.7
    )

    # Test multiple samples
    print("\n[3/3] Testing on 10 samples...")
    cocktail_count = 0
    traditional_count = 0

    for i in range(10):
        test_idx = i * 50
        clean_audio = cocktail_aug._load_audio(test_idx)

        # Truncate for speed
        clean_audio = clean_audio[:80000]  # 5 seconds

        noisy_audio = hybrid_aug.augment(clean_audio, target_idx=test_idx)

        # Detect which type was used (rough heuristic)
        snr = cocktail_aug.compute_snr(clean_audio, noisy_audio)

        # Cocktail noise typically has higher SNR (5-15 dB)
        # Traditional noise can be very low (-5 to 20 dB)
        if snr > 5:
            cocktail_count += 1
        else:
            traditional_count += 1

    print(f"\n  Cocktail noise used: ~{cocktail_count}/10")
    print(f"  Traditional noise used: ~{traditional_count}/10")
    print(f"  (Expected ~70% cocktail, ~30% traditional)")

    print("\n" + "="*70)
    print("Hybrid Augmentor test completed!")
    print("="*70)


def test_snr_statistics():
    """Compute SNR statistics across multiple samples."""
    print("\n\n")
    print("="*70)
    print("SNR Statistics Analysis")
    print("="*70)

    # Load dataset
    print("\n[1/2] Loading dataset...")
    ds_val = load_dataset(
        "MLCommons/peoples_speech",
        "validation",
        cache_dir="./data/cache"
    )
    ds_val = ds_val.cast_column("audio", Audio(decode=False))
    dataset = ds_val['validation']

    print("\n[2/2] Computing SNR statistics over 50 samples...")
    augmentor = CocktailPartyAugmentor(dataset, seed=42)

    snr_values = []
    for i in range(50):
        test_idx = i * 10
        clean_audio = augmentor._load_audio(test_idx)
        clean_audio = clean_audio[:80000]  # 5 seconds

        noisy_audio, _, _ = augmentor.add_cocktail_noise(clean_audio, target_idx=test_idx)
        snr = augmentor.compute_snr(clean_audio, noisy_audio)
        snr_values.append(snr)

    snr_values = np.array(snr_values)

    print(f"\n  SNR Statistics:")
    print(f"    Mean: {snr_values.mean():.2f} dB")
    print(f"    Std:  {snr_values.std():.2f} dB")
    print(f"    Min:  {snr_values.min():.2f} dB")
    print(f"    Max:  {snr_values.max():.2f} dB")
    print(f"    Median: {np.median(snr_values):.2f} dB")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(snr_values, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('SNR Distribution - Cocktail Party Augmentation', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axvline(snr_values.mean(), color='red', linestyle='--',
                label=f'Mean: {snr_values.mean():.2f} dB')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../results/snr_distribution.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved SNR distribution to ../results/snr_distribution.png")
    plt.close()

    print("\n" + "="*70)
    print("SNR analysis completed!")
    print("="*70)


if __name__ == "__main__":
    # Create results directory
    os.makedirs('../results', exist_ok=True)

    # Run tests
    try:
        test_cocktail_augmentor()
        test_hybrid_augmentor()
        test_snr_statistics()

        print("\n\n")
        print("🎉 All tests passed successfully!")
        print("\nGenerated files:")
        print("  - ../results/cocktail_test_clean.wav")
        print("  - ../results/cocktail_test_noisy.wav")
        print("  - ../results/cocktail_comparison.png")
        print("  - ../results/snr_distribution.png")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
