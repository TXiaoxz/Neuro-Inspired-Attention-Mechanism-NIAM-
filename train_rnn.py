#!/usr/bin/env python3
"""
RNN (GRU) Audio Enhancement Training Script
Based on train_fast.py but with RNN architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from datasets import load_dataset, Audio
import soundfile as sf
import io
import numpy as np
from tqdm import tqdm
import time
import sys

# Add src to path for cocktail augmentor
sys.path.insert(0, '/home/dlwlx05/JHU_Course/MLSP/MLSP_Project')
from src.data.cocktail_augmentor import CocktailPartyAugmentor, HybridAugmentor
from src.utils.audio_utils import decode_audio

device = torch.device('cuda')
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# Data Processing (Same as train_fast.py)
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
        if self.use_cocktail:
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
# RNN Model Architecture
# ============================================================================

class RNNEnhancer(nn.Module):
    def __init__(self, n_mels=80, hidden_dim=256, num_layers=3, dropout=0.2):
        """
        RNN-based Audio Enhancer using GRU

        Args:
            n_mels: Number of mel frequency bins
            hidden_dim: Hidden dimension size
            num_layers: Number of GRU layers
            dropout: Dropout rate
        """
        super().__init__()
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(n_mels, hidden_dim)

        # Bidirectional GRU layers
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_mels)
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
        x = self.input_proj(x)  # (batch, time, hidden_dim)

        # GRU processing
        x, _ = self.gru(x)  # (batch, time, hidden_dim * 2)

        # Layer norm
        x = self.layer_norm(x)

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

def train_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0

    for noisy_mel, clean_mel in tqdm(loader, desc="Training"):
        noisy_mel, clean_mel = noisy_mel.to(device), clean_mel.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda'):
            enhanced_mel = model(noisy_mel)
            loss = combined_loss(enhanced_mel, clean_mel)

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

    with torch.no_grad(), autocast(device_type='cuda'):
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
    # Configuration
    USE_COCKTAIL_PARTY = True  # Set to False for traditional noise augmentation

    print("Loading dataset...")
    ds_val = load_dataset("MLCommons/peoples_speech", "validation", cache_dir="./data/cache")
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
        num_samples=5000,
        use_cocktail=USE_COCKTAIL_PARTY
    )
    val_dataset = FastAudioDataset(
        dataset, audio_processor, noise_aug,
        num_samples=500,
        use_cocktail=USE_COCKTAIL_PARTY
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    print("Creating RNN model...")
    model = RNNEnhancer(
        n_mels=80,
        hidden_dim=256,
        num_layers=3,
        dropout=0.2
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    scaler = torch.amp.GradScaler('cuda')

    print("\n" + "="*70)
    print("Training RNN (GRU) Enhancer")
    print("="*70)

    best_val_loss = float('inf')
    for epoch in range(15):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scaler)
        val_loss = validate(model, val_loader)
        scheduler.step()

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/15: Train={train_loss:.4f}, Val={val_loss:.4f}, Time={epoch_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/rnn_enhancer_best.pt")
            print(f"  → Saved! (best_val={val_loss:.4f})")

    print(f"\n✓ Training complete! Best val loss: {best_val_loss:.4f}")
