#!/usr/bin/env python3
"""
Generate a fixed 30-second cocktail party noise file for demo
"""

import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio
import sys
import os

PROJECT_ROOT = '/home/dlwlx05/project/NIAM/Neuro-Inspired-Attention-Mechanism-NIAM--main'
sys.path.insert(0, PROJECT_ROOT)

print("="*80)
print("Generating Fixed Demo Noise File (30 seconds, 5 speakers)")
print("="*80)

# Load dataset
print("\nLoading dataset...")
cache_dir = '/home/dlwlx05/project/NIAM/data/clean_sa'
dataset = load_dataset("MLCommons/peoples_speech", "clean_sa", split="train", cache_dir=cache_dir)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Parameters
num_speakers = 5
target_length = 30  # seconds
sample_rate = 16000
total_samples = target_length * sample_rate

# Randomly select 5 speakers
np.random.seed(42)
speaker_indices = np.random.choice(len(dataset), num_speakers, replace=False)

print(f"\nSelected speakers (dataset indices): {speaker_indices.tolist()}")

# Collect audio segments
noise_components = []
for idx in speaker_indices:
    sample = dataset[int(idx)]
    if isinstance(sample['audio'], dict) and 'array' in sample['audio']:
        audio = sample['audio']['array'].astype(np.float32)
    else:
        audio = sample['audio'].astype(np.float32)

    # Handle stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=0)

    # Loop/crop to 30 seconds
    if len(audio) < total_samples:
        # Loop if too short
        repeats = int(np.ceil(total_samples / len(audio)))
        audio = np.tile(audio, repeats)[:total_samples]
    else:
        # Crop if too long
        audio = audio[:total_samples]

    noise_components.append(audio)
    print(f"  Speaker {idx}: {len(audio)/sample_rate:.1f}s")

# Mix all speakers with equal weights
print("\nMixing speakers...")
mixed_noise = np.zeros(total_samples, dtype=np.float32)
for audio in noise_components:
    mixed_noise += audio / num_speakers

# Normalize
mixed_noise = mixed_noise / np.max(np.abs(mixed_noise)) * 0.9

# Save
output_dir = f"{PROJECT_ROOT}/results/niam_v2_refined"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/demo_noise_30s.wav"

sf.write(output_path, mixed_noise, sample_rate)

print(f"\nNoise file saved: {output_path}")
print(f"Duration: {len(mixed_noise)/sample_rate:.1f}s")
print(f"Sample rate: {sample_rate}Hz")
print(f"RMS level: {np.sqrt(np.mean(mixed_noise**2)):.4f}")
print("\nDone!")
print("="*80)
