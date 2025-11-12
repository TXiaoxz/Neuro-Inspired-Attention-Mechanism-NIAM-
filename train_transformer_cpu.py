#!/usr/bin/env python3
"""
Transformer Audio Enhancement Training Script
Modified for zyx's CPU environment
Based on zxp's train_transformer.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio
import numpy as np
from tqdm import tqdm
import time
import sys
import math
import os

# MODIFIED: Add project root to path (adjust this to your actual path)
# For Windows: Use forward slashes or raw string
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Try to import cocktail augmentor, fallback to simple noise if not found
try:
    from src.data.cocktail_augmentor import CocktailPartyAugmentor, HybridAugmentor
    from src.utils.audio_utils import decode_audio
    COCKTAIL_AVAILABLE = True
except ImportError:
    print("Warning: Cocktail augmentor not found, using simple noise augmentation")
    COCKTAIL_AVAILABLE = False
    
    # Simple fallback decode function
    def decode_audio(sample):
        if 'audio' in sample:
            if isinstance(sample['audio'], dict):
                return sample['audio']['array']
            return sample['audio']
        return sample

# MODIFIED: Use CPU or CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# Data Processing
# ============================================================================

class AudioProcessor:
    def __init__(self):
        self.mel_spec = T.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=80, power=2.0
        )

    def audio_to_mel(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.mel_spec(waveform)
        return torch.log(mel + 1e-9).squeeze(0)

class NoiseAugmentor:
    def __init__(self):
        self.noise_types = ['white', 'pink', 'brown']

    def add_noise(self, clean_audio, snr_db, noise_type='white'):
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

        signal_power = torch.mean(clean_audio ** 2)
        noise_power = torch.mean(noise ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_scaled = noise * torch.sqrt(signal_power / (snr_linear * noise_power))
        return clean_audio + noise_scaled

class FastAudioDataset(Dataset):
    def __init__(self, hf_dataset, audio_processor, noise_aug, num_samples=5000, max_length=160000, use_cocktail=False):
        self.dataset = hf_dataset
        self.audio_processor = audio_processor
        self.noise_aug = noise_aug
        self.num_samples = min(num_samples, len(hf_dataset))
        self.max_length = max_length
        self.use_cocktail = use_cocktail

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        audio_array = decode_audio(self.dataset[idx])
        clean_audio = torch.FloatTensor(audio_array)

        if len(clean_audio.shape) > 1:
            clean_audio = clean_audio.mean(dim=0)

        # Random crop or pad
        if len(clean_audio) > self.max_length:
            start = torch.randint(0, len(clean_audio) - self.max_length, (1,))
            clean_audio = clean_audio[start:start + self.max_length]
        else:
            clean_audio = F.pad(clean_audio, (0, self.max_length - len(clean_audio)))

        # Add noise
        if self.use_cocktail and COCKTAIL_AVAILABLE:
            noisy_audio = self.noise_aug.augment(clean_audio, target_idx=idx)
        else:
            snr_db = torch.randint(-5, 21, (1,)).item()
            noise_type = np.random.choice(self.noise_aug.noise_types)
            noisy_audio = self.noise_aug.add_noise(clean_audio, snr_db, noise_type)

        # Convert to mel
        clean_mel = self.audio_processor.audio_to_mel(clean_audio)
        noisy_mel = self.audio_processor.audio_to_mel(noisy_audio)

        return noisy_mel, clean_mel

# ============================================================================
# Transformer Model Architecture
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
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerEnhancer(nn.Module):
    def __init__(self, n_mels=80, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        """
        Transformer-based Audio Enhancer

        Args:
            n_mels: Number of mel frequency bins
            d_model: Transformer embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(n_mels, d_model)

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
            norm_first=True  # Pre-layer normalization (more stable)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_mels)
        )

    def forward(self, x):
        """
        Args:
            x: Input mel-spectrogram (batch, n_mels, time)
        Returns:
            Enhanced mel-spectrogram (batch, n_mels, time)
        """
        batch_size, n_mels, time_steps = x.shape

        # Transpose to (batch, time, n_mels)
        x = x.transpose(1, 2)

        # Input projection
        x = self.input_proj(x)  # (batch, time, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, time, d_model)

        # Output projection
        x = self.output_proj(x)  # (batch, time, n_mels)

        # Transpose back to (batch, n_mels, time)
        x = x.transpose(1, 2)

        return x


# ============================================================================
# Loss & Training
# ============================================================================

def combined_loss(estimated, target):
    # Flatten
    est_flat = estimated.reshape(estimated.shape[0], -1)
    tgt_flat = target.reshape(target.shape[0], -1)

    # Zero-mean
    est_flat = est_flat - est_flat.mean(dim=1, keepdim=True)
    tgt_flat = tgt_flat - tgt_flat.mean(dim=1, keepdim=True)

    # SI-SNR
    dot_product = torch.sum(est_flat * tgt_flat, dim=1, keepdim=True)
    target_norm = torch.sum(tgt_flat ** 2, dim=1, keepdim=True)
    s_target = (dot_product / (target_norm + 1e-8)) * tgt_flat
    e_noise = est_flat - s_target
    si_snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + 1e-8))

    # Combined
    return -si_snr.mean() + 0.1 * F.mse_loss(estimated, target)

def train_epoch(model, loader, optimizer, scaler=None):
    model.train()
    total_loss = 0

    for noisy_mel, clean_mel in tqdm(loader, desc="Training"):
        noisy_mel, clean_mel = noisy_mel.to(device), clean_mel.to(device)

        optimizer.zero_grad(set_to_none=True)

        # MODIFIED: Only use autocast if CUDA is available
        if torch.cuda.is_available() and scaler is not None:
            from torch.amp import autocast
            with autocast(device_type='cuda'):
                enhanced_mel = model(noisy_mel)
                loss = combined_loss(enhanced_mel, clean_mel)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            enhanced_mel = model(noisy_mel)
            loss = combined_loss(enhanced_mel, clean_mel)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        # MODIFIED: Only use autocast if CUDA is available
        if torch.cuda.is_available():
            from torch.amp import autocast
            with autocast(device_type='cuda'):
                for noisy_mel, clean_mel in tqdm(loader, desc="Validation"):
                    noisy_mel, clean_mel = noisy_mel.to(device), clean_mel.to(device)
                    enhanced_mel = model(noisy_mel)
                    loss = combined_loss(enhanced_mel, clean_mel)
                    total_loss += loss.item()
        else:
            for noisy_mel, clean_mel in tqdm(loader, desc="Validation"):
                noisy_mel, clean_mel = noisy_mel.to(device), clean_mel.to(device)
                enhanced_mel = model(noisy_mel)
                loss = combined_loss(enhanced_mel, clean_mel)
                total_loss += loss.item()

    return total_loss / len(loader)

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # MODIFIED: Larger scale for meaningful comparison
    USE_COCKTAIL_PARTY = COCKTAIL_AVAILABLE  # Auto-detect
    NUM_TRAIN_SAMPLES = 1000  # MODIFIED: Comparable scale
    NUM_VAL_SAMPLES = 100     # MODIFIED: Comparable scale
    BATCH_SIZE = 4            # MODIFIED: CPU limit
    NUM_WORKERS = 0           # MODIFIED: CPU stability
    NUM_EPOCHS = 10           # MODIFIED: Comparable scale

    print("="*70)
    print("Configuration:")
    print(f"  Device: {device}")
    print(f"  Cocktail Party: {USE_COCKTAIL_PARTY}")
    print(f"  Train samples: {NUM_TRAIN_SAMPLES}")
    print(f"  Val samples: {NUM_VAL_SAMPLES}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print("="*70)

    print("\nLoading dataset...")
    # MODIFIED: Use relative path for cache
    cache_dir = os.path.join(project_root, "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    ds_val = load_dataset("MLCommons/peoples_speech", "validation", cache_dir=cache_dir)
    ds_val = ds_val.cast_column("audio", Audio(decode=False))
    dataset = ds_val['validation']

    print("Creating audio processor and augmentors...")
    audio_processor = AudioProcessor()

    if USE_COCKTAIL_PARTY:
        print("\n" + "="*70)
        print("COCKTAIL PARTY AUGMENTATION ENABLED")
        print("="*70)
        cocktail_aug = CocktailPartyAugmentor(
            dataset=dataset,
            noise_pool_ratio=0.2,
            num_interferers=5,
            volume_range=(0.2, 0.5),
            seed=42
        )
        noise_aug = HybridAugmentor(
            dataset=dataset,
            cocktail_augmentor=cocktail_aug,
            traditional_noise_types=['white', 'pink', 'brown'],
            cocktail_prob=0.7
        )
    else:
        print("\n" + "="*70)
        print("TRADITIONAL NOISE AUGMENTATION")
        print("="*70)
        noise_aug = NoiseAugmentor()

    print("\nCreating datasets...")
    train_dataset = FastAudioDataset(
        dataset, audio_processor, noise_aug,
        num_samples=NUM_TRAIN_SAMPLES,
        use_cocktail=USE_COCKTAIL_PARTY
    )
    val_dataset = FastAudioDataset(
        dataset, audio_processor, noise_aug,
        num_samples=NUM_VAL_SAMPLES,
        use_cocktail=USE_COCKTAIL_PARTY
    )

    # MODIFIED: Adjust DataLoader for CPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False  # Disabled for CPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False  # Disabled for CPU
    )

    print("Creating Transformer model...")
    model = TransformerEnhancer(
        n_mels=80,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # MODIFIED: Only use GradScaler if CUDA is available
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # MODIFIED: Create checkpoints directory
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\n" + "="*70)
    print("Training Transformer Enhancer")
    print("="*70)

    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scaler)
        val_loss = validate(model, val_loader)
        scheduler.step()

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train={train_loss:.4f}, Val={val_loss:.4f}, Time={epoch_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, "transformer_enhancer_best.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved to {checkpoint_path} (best_val={val_loss:.4f})")

    print(f"\n✓ Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {os.path.join(checkpoint_dir, 'transformer_enhancer_best.pt')}")