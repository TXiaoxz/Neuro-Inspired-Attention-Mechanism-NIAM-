"""
Cocktail Party Noise Generation
Extracted from check_data.ipynb

Generates realistic multi-speaker noise by mixing multiple interfering speakers
with controlled SNR based on RMS scaling.
"""

import numpy as np
import random
import torch


class CocktailPartyNoise:
    """
    Cocktail Party Noise Augmentor

    Adds multiple interfering speakers to a clean audio signal
    with controlled Signal-to-Noise Ratio (SNR).

    This implementation follows the exact method from check_data.ipynb:
    - Select N interfering speakers randomly
    - Mix them with equal weight
    - Scale the mixture to achieve target SNR using RMS normalization
    """

    def __init__(self, dataset, num_interferers=5, snr_db=8, seed=None):
        """
        Initialize Cocktail Party Noise Augmentor

        Args:
            dataset: Hugging Face dataset with audio samples
            num_interferers: Number of interfering speakers (default: 5)
            snr_db: Target Signal-to-Noise Ratio in dB (default: 8)
            seed: Random seed for reproducibility (optional)
        """
        self.dataset = dataset
        self.num_interferers = num_interferers
        self.snr_db = snr_db
        self.dataset_size = len(dataset)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def add_noise(self, clean_audio, target_idx, target_length=None):
        """
        Add cocktail party noise to clean audio

        Args:
            clean_audio: Clean audio tensor/array [T]
            target_idx: Index of the clean audio (to avoid selecting itself)
            target_length: Target length in samples (optional, uses clean_audio length if None)

        Returns:
            noisy_audio: Audio with cocktail party noise [T]
            clean_audio: Clean audio (possibly padded/cropped) [T]
            noise_indices: List of selected interferer indices
        """
        # Convert to numpy if tensor
        if isinstance(clean_audio, torch.Tensor):
            clean_audio = clean_audio.numpy()

        clean_audio = clean_audio.astype(np.float32)

        # Determine target length
        if target_length is None:
            target_length = len(clean_audio)

        # Adjust clean audio length (pad or crop)
        if len(clean_audio) > target_length:
            clean_audio = clean_audio[:target_length]
        elif len(clean_audio) < target_length:
            clean_audio = np.pad(clean_audio, (0, target_length - len(clean_audio)), mode='constant')

        # Calculate clean audio RMS
        clean_rms = np.sqrt(np.mean(clean_audio ** 2))

        # Select random interferers (excluding the target itself)
        available_indices = list(range(self.dataset_size))
        if target_idx in available_indices:
            available_indices.remove(target_idx)

        noise_indices = random.sample(available_indices, min(self.num_interferers, len(available_indices)))

        # Mix interfering speakers
        noise_mix = np.zeros(target_length, dtype=np.float32)

        for idx in noise_indices:
            # Get interferer audio
            interferer_audio = self.dataset[idx]['audio']['array'].astype(np.float32)

            # Adjust length
            if len(interferer_audio) > target_length:
                interferer_audio = interferer_audio[:target_length]
            elif len(interferer_audio) < target_length:
                interferer_audio = np.pad(
                    interferer_audio,
                    (0, target_length - len(interferer_audio)),
                    mode='constant'
                )

            # Add to noise mixture (equal weight)
            noise_mix += interferer_audio

        # Calculate noise mixture RMS
        noise_rms = np.sqrt(np.mean(noise_mix ** 2))

        # Scale noise to achieve target SNR
        # Formula: target_noise_rms = clean_rms / (10 ** (SNR_dB / 20))
        target_noise_rms = clean_rms / (10 ** (self.snr_db / 20))

        if noise_rms > 0:
            gain = target_noise_rms / noise_rms
            noise_scaled = noise_mix * gain
        else:
            noise_scaled = noise_mix

        # Mix clean audio with scaled noise
        noisy_audio = clean_audio + noise_scaled

        # Prevent clipping (scale down if needed, preserving SNR ratio)
        max_val = np.abs(noisy_audio).max()
        if max_val > 1.0:
            scale = 0.99 / max_val
            noisy_audio = noisy_audio * scale
            clean_audio = clean_audio * scale  # Also scale clean to preserve SNR

        # Final clipping
        noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
        clean_audio = np.clip(clean_audio, -1.0, 1.0)

        return noisy_audio, clean_audio, noise_indices

    def get_stats(self, clean_audio, noisy_audio):
        """
        Calculate statistics of the augmentation

        Args:
            clean_audio: Clean audio array
            noisy_audio: Noisy audio array

        Returns:
            stats: Dictionary with RMS and SNR statistics
        """
        clean_rms = np.sqrt(np.mean(clean_audio ** 2))

        # Calculate noise component
        noise = noisy_audio - clean_audio
        noise_rms = np.sqrt(np.mean(noise ** 2))

        # Calculate actual SNR
        if noise_rms > 0:
            actual_snr_db = 20 * np.log10(clean_rms / noise_rms)
        else:
            actual_snr_db = float('inf')

        noisy_rms = np.sqrt(np.mean(noisy_audio ** 2))

        return {
            'clean_rms': clean_rms,
            'noise_rms': noise_rms,
            'noisy_rms': noisy_rms,
            'actual_snr_db': actual_snr_db,
            'target_snr_db': self.snr_db
        }


def test_noise_generation():
    """
    Test function to verify noise generation works correctly
    """
    from datasets import load_dataset

    print("Loading dataset...")
    dataset = load_dataset(
        "MLCommons/peoples_speech",
        "microset",
        split="train",
        cache_dir="/home/dlwlx05/project/NIAM/data/"
    )

    print(f"Dataset loaded: {len(dataset)} samples")

    # Create noise augmentor
    noise_aug = CocktailPartyNoise(
        dataset=dataset,
        num_interferers=5,
        snr_db=8,
        seed=42
    )

    # Test on first sample
    clean_audio = dataset[0]['audio']['array']
    noisy_audio, clean_audio_processed, noise_indices = noise_aug.add_noise(
        clean_audio,
        target_idx=0,
        target_length=160000  # 10 seconds at 16kHz
    )

    # Get statistics
    stats = noise_aug.get_stats(clean_audio_processed, noisy_audio)

    print("\n" + "="*70)
    print("Noise Generation Test Results")
    print("="*70)
    print(f"Clean audio RMS: {stats['clean_rms']:.6f}")
    print(f"Noise RMS: {stats['noise_rms']:.6f}")
    print(f"Noisy audio RMS: {stats['noisy_rms']:.6f}")
    print(f"Target SNR: {stats['target_snr_db']:.2f} dB")
    print(f"Actual SNR: {stats['actual_snr_db']:.2f} dB")
    print(f"Selected interferer indices: {noise_indices}")
    print("="*70)

    # Save test files
    import soundfile as sf
    sf.write('/home/dlwlx05/project/NIAM/data/test_clean.wav', clean_audio_processed, 16000)
    sf.write('/home/dlwlx05/project/NIAM/data/test_noisy.wav', noisy_audio, 16000)
    print("\n✓ Test files saved:")
    print("  - /home/dlwlx05/project/NIAM/data/test_clean.wav")
    print("  - /home/dlwlx05/project/NIAM/data/test_noisy.wav")


if __name__ == "__main__":
    test_noise_generation()
