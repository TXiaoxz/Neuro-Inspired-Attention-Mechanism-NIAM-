#!/usr/bin/env python3
"""
NIAM v2 Inference Script

Generates test samples for evaluation using the trained NIAM v2 model.
"""

import torch
import torch.nn as nn
import soundfile as sf
import numpy as np
import os
import sys
import math

# Add project root to path and load config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PROJECT_ROOT, CLEAN_SA_CACHE, RESULTS_DIR

from datasets import load_dataset
from src.utils.noise_generation import CocktailPartyNoise
from niam_v2 import NIAM

# ============================================================================
# Model Definition (must match train_niam_v2.py)
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerEnhancer(nn.Module):
    def __init__(self, n_freq_bins=201, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1, use_niam=False, refinement_mode=False, refinement_alpha=0.2):
        super().__init__()
        self.n_freq_bins = n_freq_bins
        self.d_model = d_model
        self.use_niam = use_niam
        self.refinement_mode = refinement_mode

        self.input_proj = nn.Linear(n_freq_bins, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if use_niam:
            self.niam = NIAM(hidden_dim=d_model, refinement_mode=refinement_mode, refinement_alpha=refinement_alpha)
            print("\n" + "="*70)
            if refinement_mode:
                print("NIAM v2 INTEGRATED (Refinement Mode - Inference)")
            else:
                print("NIAM v2 INTEGRATED (Inference)")
            print("="*70)
        else:
            self.niam = None

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_freq_bins),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, n_freq_bins, time_steps = x.shape
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        # Save Transformer output before NIAM (for refinement mode)
        transformer_features = x.clone() if self.refinement_mode and self.niam is not None else None

        if self.niam is not None:
            if self.refinement_mode:
                # Refinement mode: NIAM returns (refined, baseline, residual)
                x, _, _ = self.niam(x)
            else:
                # Standard mode
                x = self.niam(x)

        mask = self.output_proj(x)
        mask = mask.transpose(1, 2)

        # Return mask and transformer features (for refinement loss)
        if self.refinement_mode and transformer_features is not None:
            transformer_mask = self.output_proj(transformer_features)
            transformer_mask = transformer_mask.transpose(1, 2)
            return mask, transformer_mask
        else:
            return mask


# ============================================================================
# Audio Enhancer Class
# ============================================================================

class AudioEnhancer:
    def __init__(self, model_path, device='cuda', n_fft=400, hop_length=160, sample_rate=16000, refinement_mode=False, refinement_alpha=0.2):
        """
        Initialize audio enhancer

        Args:
            model_path: Path to model checkpoint
            device: 'cuda' or 'cpu'
            n_fft: STFT n_fft parameter
            hop_length: STFT hop length
            sample_rate: Audio sample rate
            refinement_mode: Whether model was trained in refinement mode
            refinement_alpha: Residual correction strength (if refinement mode)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.refinement_mode = refinement_mode
        self.window = torch.hann_window(n_fft).to(self.device)

        # Load model
        print(f"Loading model from: {model_path}")
        self.model = TransformerEnhancer(
            n_freq_bins=n_fft // 2 + 1,
            d_model=256,
            nhead=8,
            num_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            use_niam=True,
            refinement_mode=refinement_mode,
            refinement_alpha=refinement_alpha
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")
        if refinement_mode:
            print(f"  Refinement mode enabled (α={refinement_alpha})")

    def enhance(self, noisy_audio):
        """
        Enhance a noisy audio signal

        Args:
            noisy_audio: Noisy audio waveform (numpy array or torch tensor)

        Returns:
            enhanced_audio: Enhanced audio waveform (numpy array)
        """
        # Convert to torch tensor
        if isinstance(noisy_audio, np.ndarray):
            noisy_audio = torch.FloatTensor(noisy_audio)

        if noisy_audio.dim() == 1:
            noisy_audio = noisy_audio.unsqueeze(0)

        noisy_audio = noisy_audio.to(self.device)
        original_length = noisy_audio.shape[-1]

        # STFT
        noisy_stft = torch.stft(
            noisy_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
            center=True
        )

        # Separate magnitude and phase
        noisy_mag = torch.abs(noisy_stft)  # [1, freq, time]
        noisy_phase = torch.angle(noisy_stft)  # [1, freq, time]

        # Model inference (predicts mask)
        with torch.no_grad():
            if self.refinement_mode:
                # Refinement mode returns (mask, transformer_mask)
                mask, _ = self.model(noisy_mag)
            else:
                mask = self.model(noisy_mag)

        # Apply mask to get enhanced magnitude
        enhanced_mag = mask * noisy_mag

        # Reconstruct using noisy phase (KEY STEP to avoid static noise!)
        enhanced_stft = enhanced_mag * torch.exp(1j * noisy_phase)

        # iSTFT
        enhanced_audio = torch.istft(
            enhanced_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            length=original_length
        )

        # Convert to numpy
        enhanced_audio = enhanced_audio.squeeze().cpu().numpy()

        return enhanced_audio


# ============================================================================
# Test Sample Generation
# ============================================================================

def generate_test_samples(model_path, num_samples=10, output_dir=None, refinement_mode=False, refinement_alpha=0.2):
    """
    Generate test samples for evaluation

    Creates clean/noisy/enhanced triplets for Whisper evaluation

    Args:
        model_path: Path to trained model
        num_samples: Number of test samples to generate
        output_dir: Output directory (default: results/niam_v2/test_samples/)
        refinement_mode: Whether model was trained in refinement mode
        refinement_alpha: Residual correction strength
    """
    if output_dir is None:
        if refinement_mode:
            output_dir = f"{PROJECT_ROOT}/results/niam_v2_refined/test_samples"
        else:
            output_dir = f"{PROJECT_ROOT}/results/niam_v2/test_samples"

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    if refinement_mode:
        print("GENERATING NIAM v2 REFINED TEST SAMPLES")
    else:
        print("GENERATING NIAM v2 TEST SAMPLES")
    print("="*80)

    # Load dataset
    print("\nLoading dataset...")
    cache_dir = CLEAN_SA_CACHE
    dataset = load_dataset(
        "MLCommons/peoples_speech",
        "clean_sa",
        split="train",
        cache_dir=cache_dir
    )

    # Create noise augmentor
    print("\nInitializing Cocktail Party noise generation...")
    noise_aug = CocktailPartyNoise(
        dataset=dataset,
        num_interferers=5,
        snr_db=8,
        seed=42
    )

    # Create enhancer
    enhancer = AudioEnhancer(model_path, refinement_mode=refinement_mode, refinement_alpha=refinement_alpha)

    # Use validation set indices (270-279 for 10 samples)
    val_indices = list(range(270, min(336, 270 + num_samples)))

    print(f"\nGenerating {len(val_indices)} test samples...")
    print(f"Output directory: {output_dir}\n")

    for i, idx in enumerate(val_indices):
        print(f"Processing sample {i}/{len(val_indices)} (dataset index {idx})...")

        # Get clean audio
        sample = dataset[idx]
        if isinstance(sample['audio'], dict) and 'array' in sample['audio']:
            clean_audio = sample['audio']['array'].astype(np.float32)
        else:
            clean_audio = sample['audio'].astype(np.float32)

        text = sample['text']

        # Crop to 10 seconds
        max_length = 160000
        if len(clean_audio) > max_length:
            clean_audio = clean_audio[:max_length]

        # Generate noisy audio
        noisy_audio, clean_audio, noise_indices = noise_aug.add_noise(
            clean_audio,
            target_idx=idx,
            target_length=len(clean_audio)
        )

        # Enhance
        enhanced_audio = enhancer.enhance(noisy_audio)

        # Save all three versions
        clean_path = f"{output_dir}/sample_{i}_clean.wav"
        noisy_path = f"{output_dir}/sample_{i}_noisy.wav"
        enhanced_path = f"{output_dir}/sample_{i}_enhanced.wav"

        sf.write(clean_path, clean_audio, 16000)
        sf.write(noisy_path, noisy_audio, 16000)
        sf.write(enhanced_path, enhanced_audio, 16000)

        # Save metadata
        with open(f"{output_dir}/sample_{i}_metadata.txt", 'w') as f:
            f.write(f"Sample {i}\n")
            f.write(f"Dataset Index: {idx}\n")
            f.write(f"Text: {text}\n")
            f.write(f"Interferer Indices: {noise_indices}\n")
            f.write(f"SNR: 8 dB\n")
            f.write(f"Duration: {len(clean_audio)/16000:.2f}s\n")

        print(f"  ✓ Saved: sample_{i}_*.wav")

    print(f"\n✓ Generated {len(val_indices)} test samples in {output_dir}/")
    print(f"\nNext: Run Whisper evaluation")
    print(f"  conda run -n niam python evaluate_whisper.py -d {output_dir}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='NIAM v2 Inference')
    parser.add_argument('-m', '--model', type=str,
                        default=os.path.join(RESULTS_DIR, 'niam_v2', 'checkpoints', 'transformer_niam_v2_best.pt'),
                        help='Model checkpoint path')
    parser.add_argument('-n', '--num_samples', type=int, default=10,
                        help='Number of test samples')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--refinement', action='store_true',
                        help='Use refinement mode (for models trained with refinement)')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Residual correction strength (default: 0.2)')

    args = parser.parse_args()

    # Auto-detect refinement mode from model path
    if 'refined' in args.model:
        args.refinement = True
        print("Auto-detected refinement mode from model path")

    generate_test_samples(args.model, args.num_samples, args.output_dir,
                         refinement_mode=args.refinement, refinement_alpha=args.alpha)
