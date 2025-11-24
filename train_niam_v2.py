#!/usr/bin/env python3
"""
NIAM v2 Training Script
Improvements:
1. Learnable residual weights (init=0.2)
2. LayerNorm for stability
3. Soft thresholding for better gradients
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio
import numpy as np
from tqdm import tqdm
import time
import sys
import math
import os

PROJECT_ROOT = '/home/dlwlx05/project/NIAM/Neuro-Inspired-Attention-Mechanism-NIAM--main'
sys.path.insert(0, PROJECT_ROOT)

# Import NIAM v2
try:
    from niam_v2 import NIAM
    print("✓ NIAM v2 loaded (with improvements)")
    NIAM_AVAILABLE = True
except ImportError:
    try:
        from niam import NIAM
        print("⚠ Using original NIAM (niam_v2.py not found)")
        NIAM_AVAILABLE = True
    except ImportError:
        print("✗ NIAM not found!")
        NIAM_AVAILABLE = False

try:
    from src.utils.noise_generation import CocktailPartyNoise
    COCKTAIL_AVAILABLE = True
except ImportError:
    COCKTAIL_AVAILABLE = False

def decode_audio(sample):
    if 'audio' in sample:
        if isinstance(sample['audio'], dict):
            return sample['audio']['array']
        return sample['audio']
    return sample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
USE_NIAM = NIAM_AVAILABLE
USE_COCKTAIL_PARTY = COCKTAIL_AVAILABLE
NUM_TRAIN_SAMPLES = 270
NUM_VAL_SAMPLES = 66
BATCH_SIZE = 4
NUM_WORKERS = 4
NUM_EPOCHS = 10

print("\n" + "="*70)
print("NIAM v2 Training")
print("="*70)
print(f"Device: {device}")
print(f"NIAM v2: {USE_NIAM}")
print(f"Improvements: LayerNorm, Learnable residual (init=0.2)")
print(f"Cocktail Party: {USE_COCKTAIL_PARTY}")
print(f"Train samples: {NUM_TRAIN_SAMPLES}")
print(f"Val samples: {NUM_VAL_SAMPLES}")
print(f"Epochs: {NUM_EPOCHS}")
print("="*70)

# Data Processing
class STFTProcessor:
    def __init__(self, n_fft=400, hop_length=160, win_length=400, sample_rate=16000):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.window = torch.hann_window(win_length)

    def audio_to_stft_mag(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True
        )

        magnitude = torch.abs(stft)
        return magnitude.squeeze(0)

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
        else:
            noise = torch.cumsum(torch.randn_like(clean_audio), dim=0)
            noise = noise / noise.std()

        signal_power = torch.mean(clean_audio ** 2)
        noise_power = torch.mean(noise ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_scaled = noise * torch.sqrt(signal_power / (snr_linear * noise_power))
        return clean_audio + noise_scaled

class FastAudioDataset(Dataset):
    def __init__(self, hf_dataset, audio_processor, noise_aug, indices, max_length=160000, use_cocktail=False):
        self.dataset = hf_dataset
        self.audio_processor = audio_processor
        self.noise_aug = noise_aug
        self.indices = indices
        self.max_length = max_length
        self.use_cocktail = use_cocktail

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        sample = self.dataset[actual_idx]

        # Extract audio array from sample
        if isinstance(sample['audio'], dict) and 'array' in sample['audio']:
            audio_array = sample['audio']['array']
        elif isinstance(sample['audio'], np.ndarray):
            audio_array = sample['audio']
        else:
            audio_array = sample['audio']

        clean_audio = torch.FloatTensor(audio_array)

        if len(clean_audio.shape) > 1:
            clean_audio = clean_audio.mean(dim=0)

        if len(clean_audio) > self.max_length:
            start = torch.randint(0, len(clean_audio) - self.max_length, (1,))
            clean_audio = clean_audio[start:start + self.max_length]
        else:
            clean_audio = F.pad(clean_audio, (0, self.max_length - len(clean_audio)))

        if self.use_cocktail and COCKTAIL_AVAILABLE:
            # CocktailPartyNoise returns (noisy, clean, indices)
            noisy_audio_np, clean_audio_np, _ = self.noise_aug.add_noise(
                clean_audio.numpy() if isinstance(clean_audio, torch.Tensor) else clean_audio,
                target_idx=actual_idx,
                target_length=len(clean_audio)
            )
            noisy_audio = torch.FloatTensor(noisy_audio_np)
            clean_audio = torch.FloatTensor(clean_audio_np)
        else:
            snr_db = torch.randint(-5, 21, (1,)).item()
            noise_type = np.random.choice(self.noise_aug.noise_types)
            noisy_audio = self.noise_aug.add_noise(clean_audio, snr_db, noise_type)

        clean_stft = self.audio_processor.audio_to_stft_mag(clean_audio)
        noisy_stft = self.audio_processor.audio_to_stft_mag(noisy_audio)

        return noisy_stft, clean_stft

# Model
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
    def __init__(self, n_freq_bins=201, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1, use_niam=False):
        super().__init__()
        self.n_freq_bins = n_freq_bins
        self.d_model = d_model
        self.use_niam = use_niam

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
        
        if use_niam and NIAM_AVAILABLE:
            self.niam = NIAM(hidden_dim=d_model)
            print("\n" + "="*70)
            print("NIAM v2 INTEGRATED")
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

        if self.niam is not None:
            x = self.niam(x)

        mask = self.output_proj(x)
        mask = mask.transpose(1, 2)
        return mask

# Loss & Training
def combined_loss(estimated, target):
    est_flat = estimated.reshape(estimated.shape[0], -1)
    tgt_flat = target.reshape(target.shape[0], -1)
    est_flat = est_flat - est_flat.mean(dim=1, keepdim=True)
    tgt_flat = tgt_flat - tgt_flat.mean(dim=1, keepdim=True)
    dot_product = torch.sum(est_flat * tgt_flat, dim=1, keepdim=True)
    target_norm = torch.sum(tgt_flat ** 2, dim=1, keepdim=True)
    s_target = (dot_product / (target_norm + 1e-8)) * tgt_flat
    e_noise = est_flat - s_target
    si_snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + 1e-8))
    return -si_snr.mean() + 0.1 * F.mse_loss(estimated, target)

def train_epoch(model, loader, optimizer, scaler=None):
    model.train()
    total_loss = 0

    for noisy_stft, clean_stft in tqdm(loader, desc="Training"):
        noisy_stft, clean_stft = noisy_stft.to(device), clean_stft.to(device)
        optimizer.zero_grad(set_to_none=True)

        if torch.cuda.is_available() and scaler is not None:
            from torch.amp import autocast
            with autocast(device_type='cuda'):
                mask = model(noisy_stft)
                enhanced_stft = mask * noisy_stft
                loss = combined_loss(enhanced_stft, clean_stft)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            mask = model(noisy_stft)
            enhanced_stft = mask * noisy_stft
            loss = combined_loss(enhanced_stft, clean_stft)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for noisy_stft, clean_stft in tqdm(loader, desc="Validation"):
            noisy_stft, clean_stft = noisy_stft.to(device), clean_stft.to(device)
            mask = model(noisy_stft)
            enhanced_stft = mask * noisy_stft
            loss = combined_loss(enhanced_stft, clean_stft)
            total_loss += loss.item()

    return total_loss / len(loader)

# Main
if __name__ == "__main__":
    print("\nLoading dataset...")
    cache_dir = '/home/dlwlx05/project/NIAM/data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    ds_microset = load_dataset("MLCommons/peoples_speech", "microset", split="train", cache_dir=cache_dir)
    ds_microset = ds_microset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = ds_microset

    # Create train/val split
    train_indices = list(range(0, 270))
    val_indices = list(range(270, 336))

    audio_processor = STFTProcessor()

    if USE_COCKTAIL_PARTY:
        print("\nInitializing Cocktail Party noise generation...")
        noise_aug = CocktailPartyNoise(
            dataset=dataset,
            num_interferers=5,
            snr_db=8,
            seed=42
        )
    else:
        noise_aug = NoiseAugmentor()

    train_dataset = FastAudioDataset(
        dataset, audio_processor, noise_aug,
        indices=train_indices,
        use_cocktail=USE_COCKTAIL_PARTY
    )
    val_dataset = FastAudioDataset(
        dataset, audio_processor, noise_aug,
        indices=val_indices,
        use_cocktail=USE_COCKTAIL_PARTY
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    print("\nCreating model...")
    model = TransformerEnhancer(
        n_freq_bins=201, d_model=256, nhead=8, num_layers=4,
        dim_feedforward=1024, dropout=0.1, use_niam=USE_NIAM
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    checkpoint_dir = '/home/dlwlx05/project/NIAM/results/niam_v2/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\n" + "="*70)
    print("Training NIAM v2 Transformer")
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
            checkpoint_path = os.path.join(checkpoint_dir, "transformer_niam_v2_best.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved (best_val={val_loss:.4f})")
        
        # Print learned residual weights at end
        if epoch == NUM_EPOCHS - 1 and hasattr(model, 'niam') and model.niam is not None:
            print("\nLearned residual weights:")
            print(f"  Module 1: {model.niam.residual_weight_1.item():.3f}")
            print(f"  Module 2: {model.niam.residual_weight_2.item():.3f}")
            print(f"  Module 3: {model.niam.residual_weight_3.item():.3f}")
            print(f"  Module 4: {model.niam.residual_weight_4.item():.3f}")

    print(f"\n✓ Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"\nCheckpoint saved: {checkpoint_dir}/transformer_niam_v2_best.pt")
    print("\nNext: Send results to zxp for GPU testing")