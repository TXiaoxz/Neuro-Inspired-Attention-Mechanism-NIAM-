import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio
import soundfile as sf
import io
import numpy as np
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.audio_utils import decode_audio

print("="*70)
print("DEBUGGING TRAINING PIPELINE")
print("="*70)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n1. GPU Check:")
print(f"   Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test data loading
print(f"\n2. Loading dataset...")
try:
    ds_val = load_dataset("MLCommons/peoples_speech", "validation", cache_dir="./data/cache")
    ds_val = ds_val.cast_column("audio", Audio(decode=False))
    print(f"   ✓ Dataset loaded: {len(ds_val['validation'])} samples")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test audio processing
print(f"\n3. Testing audio processing...")
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

audio_processor = AudioProcessor()

try:
    start = time.time()
    sample = ds_val['validation'][0]
    audio_array = decode_audio(sample)
    clean_audio = torch.FloatTensor(audio_array)
    if len(clean_audio.shape) > 1:
        clean_audio = clean_audio.mean(dim=0)
    
    # Truncate
    max_length = 160000
    if len(clean_audio) > max_length:
        clean_audio = clean_audio[:max_length]
    else:
        clean_audio = F.pad(clean_audio, (0, max_length - len(clean_audio)))
    
    clean_mel = audio_processor.audio_to_mel(clean_audio)
    elapsed = time.time() - start
    print(f"   ✓ Audio processed in {elapsed:.3f}s")
    print(f"   - Audio shape: {clean_audio.shape}")
    print(f"   - Mel shape: {clean_mel.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test dataset
print(f"\n4. Testing dataset class...")
class SimpleDataset(Dataset):
    def __init__(self, hf_dataset, audio_processor, num_samples=100):
        self.dataset = hf_dataset
        self.audio_processor = audio_processor
        self.num_samples = min(num_samples, len(hf_dataset))
        self.max_length = 160000
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        audio_array = decode_audio(self.dataset[idx])
        clean_audio = torch.FloatTensor(audio_array)
        if len(clean_audio.shape) > 1:
            clean_audio = clean_audio.mean(dim=0)
        
        if len(clean_audio) > self.max_length:
            clean_audio = clean_audio[:self.max_length]
        else:
            clean_audio = F.pad(clean_audio, (0, self.max_length - len(clean_audio)))
        
        # Add simple noise
        noise = torch.randn_like(clean_audio) * 0.1
        noisy_audio = clean_audio + noise
        
        clean_mel = self.audio_processor.audio_to_mel(clean_audio)
        noisy_mel = self.audio_processor.audio_to_mel(noisy_audio)
        
        return noisy_mel, clean_mel

try:
    test_dataset = SimpleDataset(ds_val['validation'], audio_processor, num_samples=10)
    print(f"   ✓ Dataset created: {len(test_dataset)} samples")
    
    # Test single item
    start = time.time()
    item = test_dataset[0]
    elapsed = time.time() - start
    print(f"   ✓ Single item accessed in {elapsed:.3f}s")
    print(f"   - Item shapes: {item[0].shape}, {item[1].shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test dataloader
print(f"\n5. Testing DataLoader...")
try:
    # Test with num_workers=0
    print(f"   Testing with num_workers=0...")
    loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    start = time.time()
    batch = next(iter(loader))
    elapsed = time.time() - start
    print(f"   ✓ First batch loaded in {elapsed:.3f}s")
    print(f"   - Batch shapes: {batch[0].shape}, {batch[1].shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test model
print(f"\n6. Testing model...")
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 1, 3, padding=1)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x.squeeze(1)

try:
    model = SimpleCNN().to(device)
    test_input = batch[0].to(device)
    
    start = time.time()
    with torch.no_grad():
        output = model(test_input)
    elapsed = time.time() - start
    print(f"   ✓ Model forward pass in {elapsed:.3f}s")
    print(f"   - Input shape: {test_input.shape}")
    print(f"   - Output shape: {output.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test training loop
print(f"\n7. Testing training loop (1 batch)...")
try:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    noisy_mel = batch[0].to(device)
    clean_mel = batch[1].to(device)
    
    start = time.time()
    optimizer.zero_grad()
    output = model(noisy_mel)
    loss = F.mse_loss(output, clean_mel)
    loss.backward()
    optimizer.step()
    elapsed = time.time() - start
    
    print(f"   ✓ Training step completed in {elapsed:.3f}s")
    print(f"   - Loss: {loss.item():.4f}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Full epoch test
print(f"\n8. Testing full epoch (10 samples, batch_size=2)...")
try:
    loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    start = time.time()
    total_loss = 0
    for i, (noisy_mel, clean_mel) in enumerate(loader):
        noisy_mel = noisy_mel.to(device)
        clean_mel = clean_mel.to(device)
        
        optimizer.zero_grad()
        output = model(noisy_mel)
        loss = F.mse_loss(output, clean_mel)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"   Batch {i+1}/{len(loader)}: loss={loss.item():.4f}")
    
    elapsed = time.time() - start
    avg_loss = total_loss / len(loader)
    print(f"   ✓ Epoch completed in {elapsed:.3f}s")
    print(f"   - Average loss: {avg_loss:.4f}")
    print(f"   - Time per batch: {elapsed/len(loader):.3f}s")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n{'='*70}")
print(f"✓ ALL TESTS PASSED!")
print(f"{'='*70}")
print(f"\nYour setup is working correctly. The issue might be:")
print(f"1. First batch takes longer (data loading/model compilation)")
print(f"2. Need to wait longer (30-60 seconds for first batch)")
print(f"3. Batch size too large - try batch_size=16 or 8")
