#!/usr/bin/env python3
"""
Transformer Audio Enhancement Training Script (V2)
Fixed version using STFT instead of Mel-Spectrogram to prevent static noise

Key improvements:
1. Uses STFT magnitude instead of Mel-Spectrogram
2. Preserves noisy phase for reconstruction
3. Uses cocktail party noise from check_data.ipynb
4. Works with microset (336 samples)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from datasets import load_dataset
import soundfile as sf
import numpy as np
from tqdm import tqdm
import time
import sys
import math
import os

# Add src to path
PROJECT_ROOT = '/home/dlwlx05/project/NIAM/Neuro-Inspired-Attention-Mechanism-NIAM--main'
sys.path.insert(0, PROJECT_ROOT)
from src.utils.noise_generation import CocktailPartyNoise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# STFT Audio Processor (Replaces Mel-Spectrogram)
# ============================================================================

class STFTProcessor:
    """
    STFT-based audio processor

    Uses STFT magnitude as features instead of Mel-Spectrogram
    to avoid information loss and static noise issues.
    """
    def __init__(self, n_fft=400, hop_length=160, win_length=400, sample_rate=16000):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.window = torch.hann_window(win_length)

    def audio_to_stft_mag(self, waveform):
        """
        Convert audio to STFT magnitude

        Args:
            waveform: Audio tensor [T] or [1, T]

        Returns:
            magnitude: STFT magnitude [freq_bins, time_frames]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Move window to same device as waveform
        window = self.window.to(waveform.device)

        # STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True
        )

        # Return magnitude only
        magnitude = torch.abs(stft).squeeze(0)  # [freq_bins, time_frames]
        return magnitude

    def stft_mag_to_audio(self, magnitude, phase):
        """
        Reconstruct audio from STFT magnitude and phase

        Args:
            magnitude: STFT magnitude [freq_bins, time_frames]
            phase: STFT phase [freq_bins, time_frames]

        Returns:
            waveform: Reconstructed audio [T]
        """
        if magnitude.dim() == 2:
            magnitude = magnitude.unsqueeze(0)
        if phase.dim() == 2:
            phase = phase.unsqueeze(0)

        # Reconstruct complex STFT
        stft_complex = magnitude * torch.exp(1j * phase)

        # Move window to same device
        window = self.window.to(magnitude.device)

        # iSTFT
        waveform = torch.istft(
            stft_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True
        )

        return waveform.squeeze(0)

# ============================================================================
# Dataset with Cocktail Party Noise
# ============================================================================

class AudioDatasetSTFT(Dataset):
    """
    Dataset using STFT features and cocktail party noise
    """
    def __init__(self, hf_dataset, stft_processor, noise_aug, indices, max_length=160000):
        self.dataset = hf_dataset
        self.stft_processor = stft_processor
        self.noise_aug = noise_aug
        self.indices = indices
        self.max_length = max_length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get actual dataset index
        dataset_idx = self.indices[idx]

        # Load clean audio
        clean_audio = self.dataset[dataset_idx]['audio']['array']
        clean_audio = torch.FloatTensor(clean_audio)

        if len(clean_audio.shape) > 1:
            clean_audio = clean_audio.mean(dim=0)

        # Random crop or pad
        if len(clean_audio) > self.max_length:
            start = torch.randint(0, len(clean_audio) - self.max_length, (1,)).item()
            clean_audio = clean_audio[start:start + self.max_length]
        else:
            clean_audio = F.pad(clean_audio, (0, self.max_length - len(clean_audio)))

        # Add cocktail party noise
        noisy_audio_np, clean_audio_np, _ = self.noise_aug.add_noise(
            clean_audio.numpy(),
            target_idx=dataset_idx,
            target_length=self.max_length
        )

        noisy_audio = torch.FloatTensor(noisy_audio_np)
        clean_audio = torch.FloatTensor(clean_audio_np)

        # Convert to STFT magnitude
        clean_mag = self.stft_processor.audio_to_stft_mag(clean_audio)
        noisy_mag = self.stft_processor.audio_to_stft_mag(noisy_audio)

        # Also get noisy phase for reconstruction
        noisy_waveform = noisy_audio.unsqueeze(0)
        window = self.stft_processor.window
        noisy_stft = torch.stft(
            noisy_waveform,
            n_fft=self.stft_processor.n_fft,
            hop_length=self.stft_processor.hop_length,
            win_length=self.stft_processor.win_length,
            window=window,
            return_complex=True,
            center=True
        )
        noisy_phase = torch.angle(noisy_stft).squeeze(0)

        return noisy_mag, clean_mag, noisy_phase

# ============================================================================
# Transformer Model (Unchanged Architecture)
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
    def __init__(self, n_freq_bins=201, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        """
        Transformer-based Audio Enhancer (STFT version)

        Args:
            n_freq_bins: Number of frequency bins (n_fft // 2 + 1 = 201 for n_fft=400)
            d_model: Transformer embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.n_freq_bins = n_freq_bins
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(n_freq_bins, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection (outputs mask in range [0, 1])
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_freq_bins),
            nn.Sigmoid()  # Constrain mask to [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: Input STFT magnitude (batch, n_freq_bins, time)
        Returns:
            mask: Magnitude mask (batch, n_freq_bins, time) in range [0, 1]
        """
        batch_size, n_freq_bins, time_steps = x.shape

        # Transpose to (batch, time, n_freq_bins)
        x = x.transpose(1, 2)

        # Input projection
        x = self.input_proj(x)  # (batch, time, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, time, d_model)

        # Output projection (with sigmoid for mask)
        mask = self.output_proj(x)  # (batch, time, n_freq_bins)

        # Transpose back to (batch, n_freq_bins, time)
        mask = mask.transpose(1, 2)

        return mask

# ============================================================================
# Loss & Training
# ============================================================================

def combined_loss(estimated_mag, target_mag):
    """
    Combined loss for magnitude estimation

    Args:
        estimated_mag: Estimated clean magnitude
        target_mag: Target clean magnitude
    """
    # MSE loss on magnitude
    mse_loss = F.mse_loss(estimated_mag, target_mag)

    # L1 loss on magnitude (for sparsity)
    l1_loss = F.l1_loss(estimated_mag, target_mag)

    # Combined
    return mse_loss + 0.1 * l1_loss

def train_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0

    for noisy_mag, clean_mag, noisy_phase in tqdm(loader, desc="Training"):
        noisy_mag = noisy_mag.to(device)
        clean_mag = clean_mag.to(device)
        # noisy_phase not used in training, only in inference

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            # Model predicts mask
            mask = model(noisy_mag)

            # Apply mask to get estimated clean magnitude
            estimated_clean_mag = mask * noisy_mag

            # Loss
            loss = combined_loss(estimated_clean_mag, clean_mag)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    total_loss = 0

    with torch.no_grad(), autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
        for noisy_mag, clean_mag, noisy_phase in tqdm(loader, desc="Validation"):
            noisy_mag = noisy_mag.to(device)
            clean_mag = clean_mag.to(device)

            # Model predicts mask
            mask = model(noisy_mag)

            # Apply mask
            estimated_clean_mag = mask * noisy_mag

            # Loss
            loss = combined_loss(estimated_clean_mag, clean_mag)
            total_loss += loss.item()

    return total_loss / len(loader)

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("TRANSFORMER AUDIO ENHANCEMENT TRAINING (V2 - STFT VERSION)")
    print("="*80)

    # Create checkpoints directory
    os.makedirs(f"{PROJECT_ROOT}/checkpoints", exist_ok=True)

    # Load dataset
    print("\n Loading microset dataset (336 samples)...")
    dataset = load_dataset(
        "MLCommons/peoples_speech",
        "microset",
        split="train",
        cache_dir="/home/dlwlx05/project/NIAM/data/"
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")

    # Create STFT processor
    print("\nCreating STFT processor...")
    stft_processor = STFTProcessor(
        n_fft=400,
        hop_length=160,
        win_length=400,
        sample_rate=16000
    )
    print("✓ STFT processor created")

    # Create noise augmentor
    print("\nCreating cocktail party noise augmentor...")
    noise_aug = CocktailPartyNoise(
        dataset=dataset,
        num_interferers=5,
        snr_db=8,
        seed=42
    )
    print("✓ Noise augmentor created (5 interferers, SNR=8dB)")

    # Split dataset: 270 train / 66 val (80/20 split)
    train_indices = list(range(270))
    val_indices = list(range(270, 336))

    print(f"\nDataset split:")
    print(f"  Training: {len(train_indices)} samples")
    print(f"  Validation: {len(val_indices)} samples")

    # Create datasets
    train_dataset = AudioDatasetSTFT(
        hf_dataset=dataset,
        stft_processor=stft_processor,
        noise_aug=noise_aug,
        indices=train_indices,
        max_length=160000  # 10 seconds
    )

    val_dataset = AudioDatasetSTFT(
        hf_dataset=dataset,
        stft_processor=stft_processor,
        noise_aug=noise_aug,
        indices=val_indices,
        max_length=160000
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  # Smaller batch size for limited data
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Create model
    print("\nCreating Transformer model...")
    model = TransformerEnhancer(
        n_freq_bins=201,  # n_fft//2 + 1 = 400//2 + 1 = 201
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)

    best_val_loss = float('inf')
    num_epochs = 15

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)

        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scaler)
        val_loss = validate(model, val_loader)
        scheduler.step()

        epoch_time = time.time() - start_time

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = f"{PROJECT_ROOT}/checkpoints/transformer_stft_best.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {PROJECT_ROOT}/checkpoints/transformer_stft_best.pt")
    print()
