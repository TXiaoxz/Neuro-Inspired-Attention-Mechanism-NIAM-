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
# Data Processing (Optimized)
# ============================================================================

class AudioProcessor:
    def __init__(self):
        # Keep on CPU to avoid issues with multiprocessing DataLoader
        self.mel_spec = T.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=80, power=2.0
        )

    def audio_to_mel(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        # Compute on CPU to avoid multiprocessing issues
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

        # Add noise (cocktail party or traditional)
        if self.use_cocktail:
            # Use HybridAugmentor (70% cocktail, 30% traditional)
            noisy_audio = self.noise_aug.augment(clean_audio, target_idx=idx)
        else:
            # Use traditional noise augmentation
            snr_db = torch.randint(-5, 21, (1,)).item()
            noise_type = np.random.choice(self.noise_aug.noise_types)
            noisy_audio = self.noise_aug.add_noise(clean_audio, snr_db, noise_type)

        # Convert to mel (computed on CPU for multiprocessing compatibility)
        clean_mel = self.audio_processor.audio_to_mel(clean_audio)
        noisy_mel = self.audio_processor.audio_to_mel(noisy_audio)

        return noisy_mel, clean_mel

# ============================================================================
# Models (Same as before)
# ============================================================================

class CNNEnhancer(nn.Module):
    def __init__(self, n_mels=80, hidden_dim=64, num_layers=4):
        super().__init__()
        self.encoder = nn.ModuleList()
        in_ch = 1
        for i in range(num_layers):
            out_ch = hidden_dim * (2 ** i)
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch
        
        self.decoder = nn.ModuleList()
        for i in range(num_layers - 1, -1, -1):
            in_ch = hidden_dim * (2 ** i)
            out_ch = hidden_dim * (2 ** (i - 1)) if i > 0 else hidden_dim
            self.decoder.append(nn.Sequential(
                nn.Conv2d(in_ch * 2, in_ch, 3, padding=1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
        
        self.output = nn.Conv2d(hidden_dim, 1, 1)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        skip_connections = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            skip_connections.append(x)
        
        for i, decoder_layer in enumerate(self.decoder):
            skip = skip_connections[-(i + 1)]
            x = torch.cat([x, skip], dim=1)
            x = decoder_layer(x)
        
        return self.output(x).squeeze(1)

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
    # Configuration: Set USE_COCKTAIL_PARTY to True to enable cocktail party augmentation
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
        # Create cocktail party augmentor
        cocktail_aug = CocktailPartyAugmentor(
            dataset=dataset,
            noise_pool_ratio=0.2,      # 20% of dataset as noise pool
            num_interferers=5,          # Mix 5 interfering speakers
            volume_range=(0.2, 0.5),   # 20-50% volume for each interferer
            seed=42
        )
        # Create hybrid augmentor (70% cocktail, 30% traditional)
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
    
    # Optimized DataLoader settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,  # Reduced for memory constraints with cocktail augmentation
        shuffle=True,
        num_workers=8,  # Increased from 4
        pin_memory=True,  # NEW
        persistent_workers=True,  # NEW
        prefetch_factor=2  # NEW
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,  # Reduced for memory constraints
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    print("Creating model...")
    model = CNNEnhancer().to(device)
    # Note: torch.compile disabled to reduce memory usage
    # Uncomment below if you have enough GPU memory
    # model = torch.compile(model)  # PyTorch 2.0 speedup
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    scaler = torch.amp.GradScaler('cuda')
    
    print("\n" + "="*70)
    print("Training CNN Enhancer (Optimized)")
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
            torch.save(model.state_dict(), "checkpoints/cnn_enhancer_best.pt")
            print(f"  → Saved! (best_val={val_loss:.4f})")
    
    print(f"\n✓ Training complete! Best val loss: {best_val_loss:.4f}")
