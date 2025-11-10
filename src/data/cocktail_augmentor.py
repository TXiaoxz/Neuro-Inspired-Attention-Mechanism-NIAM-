"""
Cocktail Party Effect Data Augmentation

This module implements realistic multi-speaker noise augmentation for audio enhancement tasks.
Instead of using synthetic noise (white/pink/brown), it mixes real speech samples to simulate
the cocktail party effect where multiple speakers are present simultaneously.

Key Features:
- Uses 20% of dataset as noise pool
- Randomly selects 5 interfering speakers per sample
- Each interferer mixed at 20-50% volume relative to target
- More realistic training scenario for speech enhancement
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple


class CocktailPartyAugmentor:
    """
    Augments clean audio by mixing it with multiple random speech samples
    to simulate a multi-speaker environment (cocktail party effect).
    """

    def __init__(
        self,
        dataset,
        noise_pool_ratio: float = 0.2,
        num_interferers: int = 5,
        volume_range: Tuple[float, float] = (0.2, 0.5),
        seed: Optional[int] = None
    ):
        """
        Args:
            dataset: The source dataset to sample noise from
            noise_pool_ratio: Fraction of dataset to use as noise pool (default: 0.2 = 20%)
            num_interferers: Number of interfering speakers to mix (default: 5)
            volume_range: Min and max volume for interferers (default: 0.2-0.5 = 20%-50%)
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.noise_pool_ratio = noise_pool_ratio
        self.num_interferers = num_interferers
        self.volume_range = volume_range

        # Create noise pool indices (20% of dataset)
        total_samples = len(dataset)
        noise_pool_size = int(total_samples * noise_pool_ratio)

        if seed is not None:
            np.random.seed(seed)

        # Randomly select indices for noise pool
        self.noise_pool_indices = np.random.choice(
            total_samples,
            size=noise_pool_size,
            replace=False
        )

        print(f"[CocktailPartyAugmentor] Initialized:")
        print(f"  - Total dataset size: {total_samples}")
        print(f"  - Noise pool size: {noise_pool_size} ({noise_pool_ratio*100:.0f}%)")
        print(f"  - Interferers per sample: {num_interferers}")
        print(f"  - Volume range: {volume_range[0]*100:.0f}%-{volume_range[1]*100:.0f}%")

    def _load_audio(self, idx: int) -> torch.Tensor:
        """
        Load audio from dataset at given index.

        Args:
            idx: Dataset index

        Returns:
            Audio tensor (1D)
        """
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.utils.audio_utils import decode_audio

        audio_array = decode_audio(self.dataset[idx])
        audio_tensor = torch.FloatTensor(audio_array)

        # Convert stereo to mono if needed
        if len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor.mean(dim=0)

        return audio_tensor

    def _match_length(
        self,
        target_audio: torch.Tensor,
        noise_audio: torch.Tensor
    ) -> torch.Tensor:
        """
        Match noise audio length to target audio length by cropping or padding.

        Args:
            target_audio: Target audio tensor (1D)
            noise_audio: Noise audio tensor (1D)

        Returns:
            Length-matched noise tensor
        """
        target_len = len(target_audio)
        noise_len = len(noise_audio)

        if noise_len > target_len:
            # Random crop
            start = torch.randint(0, noise_len - target_len, (1,)).item()
            return noise_audio[start:start + target_len]
        elif noise_len < target_len:
            # Pad with zeros
            return F.pad(noise_audio, (0, target_len - noise_len))
        else:
            return noise_audio

    def add_cocktail_noise(
        self,
        clean_audio: torch.Tensor,
        target_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[int], List[float]]:
        """
        Add cocktail party noise to clean audio by mixing multiple interfering speakers.

        Args:
            clean_audio: Clean target audio (1D tensor)
            target_idx: Index of target audio (to avoid self-mixing), optional

        Returns:
            noisy_audio: Mixed audio with interferers
            interferer_indices: List of interferer sample indices used
            interferer_volumes: List of volume levels applied to each interferer
        """
        # Randomly select interferers from noise pool
        num_to_sample = min(self.num_interferers, len(self.noise_pool_indices))
        selected_indices = np.random.choice(
            self.noise_pool_indices,
            size=num_to_sample,
            replace=False
        )

        # Remove target index if present (avoid self-mixing)
        if target_idx is not None:
            selected_indices = selected_indices[selected_indices != target_idx]
            # If we removed one, sample one more
            if len(selected_indices) < num_to_sample:
                available = [idx for idx in self.noise_pool_indices
                           if idx != target_idx and idx not in selected_indices]
                if available:
                    extra = np.random.choice(available, size=1)
                    selected_indices = np.append(selected_indices, extra)

        # Start with clean audio
        mixed_audio = clean_audio.clone()
        interferer_volumes = []

        # Mix each interferer
        for interferer_idx in selected_indices:
            # Load interferer audio
            interferer_audio = self._load_audio(int(interferer_idx))

            # Match length
            interferer_audio = self._match_length(clean_audio, interferer_audio)

            # Random volume in range [0.2, 0.5]
            volume = np.random.uniform(*self.volume_range)
            interferer_volumes.append(volume)

            # Mix with target
            mixed_audio = mixed_audio + volume * interferer_audio

        return mixed_audio, selected_indices.tolist(), interferer_volumes

    def get_noise_sample(self, idx: int) -> torch.Tensor:
        """
        Get a noise sample from the noise pool.

        Args:
            idx: Index in noise pool

        Returns:
            Audio tensor
        """
        noise_idx = self.noise_pool_indices[idx % len(self.noise_pool_indices)]
        return self._load_audio(noise_idx)

    def compute_snr(
        self,
        clean_audio: torch.Tensor,
        noisy_audio: torch.Tensor
    ) -> float:
        """
        Compute Signal-to-Noise Ratio in dB.

        Args:
            clean_audio: Clean signal
            noisy_audio: Noisy signal

        Returns:
            SNR in dB
        """
        noise = noisy_audio - clean_audio
        signal_power = torch.mean(clean_audio ** 2)
        noise_power = torch.mean(noise ** 2)

        if noise_power < 1e-10:
            return float('inf')

        snr = 10 * torch.log10(signal_power / noise_power)
        return snr.item()


class HybridAugmentor:
    """
    Combines traditional noise augmentation with cocktail party augmentation.
    Randomly applies one of: white/pink/brown noise OR cocktail party noise.
    """

    def __init__(
        self,
        dataset,
        cocktail_augmentor: CocktailPartyAugmentor,
        traditional_noise_types: List[str] = ['white', 'pink', 'brown'],
        cocktail_prob: float = 0.7
    ):
        """
        Args:
            dataset: Source dataset
            cocktail_augmentor: CocktailPartyAugmentor instance
            traditional_noise_types: List of traditional noise types
            cocktail_prob: Probability of using cocktail noise vs traditional (default: 0.7)
        """
        self.cocktail_augmentor = cocktail_augmentor
        self.traditional_noise_types = traditional_noise_types
        self.cocktail_prob = cocktail_prob

        print(f"[HybridAugmentor] Cocktail probability: {cocktail_prob*100:.0f}%")

    def add_traditional_noise(
        self,
        clean_audio: torch.Tensor,
        snr_db: float,
        noise_type: str = 'white'
    ) -> torch.Tensor:
        """
        Add traditional synthetic noise (white/pink/brown).

        Args:
            clean_audio: Clean audio
            snr_db: Target SNR in dB
            noise_type: Type of noise ('white', 'pink', 'brown')

        Returns:
            Noisy audio
        """
        if noise_type == 'white':
            noise = torch.randn_like(clean_audio)
        elif noise_type == 'pink':
            noise = F.avg_pool1d(
                torch.randn_like(clean_audio).unsqueeze(0).unsqueeze(0),
                kernel_size=3, stride=1, padding=1
            ).squeeze()
        else:  # brown
            noise = torch.cumsum(torch.randn_like(clean_audio), dim=0)
            noise = noise / noise.std()

        # Scale noise to target SNR
        signal_power = torch.mean(clean_audio ** 2)
        noise_power = torch.mean(noise ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_scaled = noise * torch.sqrt(signal_power / (snr_linear * noise_power))

        return clean_audio + noise_scaled

    def augment(
        self,
        clean_audio: torch.Tensor,
        target_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply hybrid augmentation: randomly choose between cocktail and traditional noise.

        Args:
            clean_audio: Clean audio
            target_idx: Target audio index (for cocktail noise)

        Returns:
            Augmented audio
        """
        if np.random.random() < self.cocktail_prob:
            # Use cocktail party noise
            noisy_audio, _, _ = self.cocktail_augmentor.add_cocktail_noise(
                clean_audio, target_idx
            )
        else:
            # Use traditional noise
            snr_db = np.random.uniform(-5, 20)
            noise_type = np.random.choice(self.traditional_noise_types)
            noisy_audio = self.add_traditional_noise(clean_audio, snr_db, noise_type)

        return noisy_audio


if __name__ == "__main__":
    print("Cocktail Party Augmentor - Ready for testing")
    print("Import this module and use CocktailPartyAugmentor or HybridAugmentor")
