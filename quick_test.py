#!/usr/bin/env python3
"""
Quick Test Script
Generate before/after comparison audio and visualizations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from datasets import load_dataset, Audio
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
from pathlib import Path

sys.path.insert(0, '/home/dlwlx05/JHU_Course/MLSP/MLSP_Project')
from src.data.cocktail_augmentor import CocktailPartyAugmentor
from src.utils.audio_utils import decode_audio


# ============================================================================
# Model Definition
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
# Audio Processing
# ============================================================================

class AudioProcessor:
    def __init__(self, device='cpu'):
        self.mel_spec = T.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=80, power=2.0
        ).to(device)

        self.inverse_mel = T.InverseMelScale(
            n_stft=201, n_mels=80, sample_rate=16000
        ).to(device)

        self.griffin_lim = T.GriffinLim(
            n_fft=400, hop_length=160, power=2.0, n_iter=32
        ).to(device)

        self.device = device

    def audio_to_mel(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.mel_spec(waveform.to(self.device))
        return torch.log(mel + 1e-9).squeeze(0)

    def mel_to_audio(self, mel_spec):
        mel_linear = torch.exp(mel_spec) - 1e-9
        stft = self.inverse_mel(mel_linear)
        audio = self.griffin_lim(stft)
        return audio


# ============================================================================
# Visualization
# ============================================================================

def plot_comparison(clean_mel, noisy_mel, enhanced_mel, output_path):
    """
    Create comparison visualization of mel-spectrograms

    Args:
        clean_mel: Clean mel-spectrogram (F, T)
        noisy_mel: Noisy mel-spectrogram (F, T)
        enhanced_mel: Enhanced mel-spectrogram (F, T)
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Convert to numpy
    clean_np = clean_mel.cpu().numpy()
    noisy_np = noisy_mel.cpu().numpy()
    enhanced_np = enhanced_mel.cpu().numpy()

    # Plot clean
    im1 = axes[0].imshow(clean_np, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Clean Audio (Target)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Mel Frequency Bins')
    plt.colorbar(im1, ax=axes[0], label='Log Magnitude')

    # Plot noisy
    im2 = axes[1].imshow(noisy_np, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Noisy Audio (Input)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Mel Frequency Bins')
    plt.colorbar(im2, ax=axes[1], label='Log Magnitude')

    # Plot enhanced
    im3 = axes[2].imshow(enhanced_np, aspect='auto', origin='lower', cmap='viridis')
    axes[2].set_title('Enhanced Audio (Model Output)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Mel Frequency Bins')
    axes[2].set_xlabel('Time Frames')
    plt.colorbar(im3, ax=axes[2], label='Log Magnitude')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_path}")
    plt.close()


def plot_waveforms(clean_audio, noisy_audio, enhanced_audio, output_path, sr=16000):
    """
    Plot waveform comparison

    Args:
        clean_audio: Clean waveform
        noisy_audio: Noisy waveform
        enhanced_audio: Enhanced waveform
        output_path: Path to save figure
        sr: Sample rate
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))

    time = np.arange(len(clean_audio)) / sr

    # Plot clean
    axes[0].plot(time, clean_audio, linewidth=0.5, color='blue')
    axes[0].set_title('Clean Audio (Target)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(-1, 1)

    # Plot noisy
    axes[1].plot(time, noisy_audio, linewidth=0.5, color='red')
    axes[1].set_title('Noisy Audio (Input)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(-1, 1)

    # Plot enhanced
    axes[2].plot(time, enhanced_audio, linewidth=0.5, color='green')
    axes[2].set_title('Enhanced Audio (Model Output)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].grid(alpha=0.3)
    axes[2].set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved waveform plot: {output_path}")
    plt.close()


# ============================================================================
# Main Test Function
# ============================================================================

def quick_test(model_path, output_dir='results', sample_idx=0, use_cocktail=True, device='cuda'):
    """
    Run quick test and generate comparison outputs

    Args:
        model_path: Path to model checkpoint
        output_dir: Directory to save results
        sample_idx: Index of sample to test
        use_cocktail: Use cocktail party augmentation
        device: Device to use
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    ds_val = load_dataset("MLCommons/peoples_speech", "validation", cache_dir="./data/cache")
    ds_val = ds_val.cast_column("audio", Audio(decode=False))
    dataset = ds_val['validation']

    # Load model
    print(f"Loading model: {model_path}")
    model = CNNEnhancer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✓ Model loaded")

    # Create audio processor
    audio_processor = AudioProcessor(device=device)

    # Load sample
    print(f"Loading sample {sample_idx}...")
    audio_array = decode_audio(dataset[sample_idx])
    clean_audio = torch.FloatTensor(audio_array)

    if len(clean_audio.shape) > 1:
        clean_audio = clean_audio.mean(dim=0)

    # Crop to 10 seconds for faster processing
    max_length = 160000  # 10 seconds at 16kHz
    if len(clean_audio) > max_length:
        clean_audio = clean_audio[:max_length]
    else:
        clean_audio = F.pad(clean_audio, (0, max_length - len(clean_audio)))

    # Add noise
    print("Adding noise...")
    if use_cocktail:
        print("  Using cocktail party augmentation")
        augmentor = CocktailPartyAugmentor(
            dataset=dataset,
            noise_pool_ratio=0.2,
            num_interferers=5,
            volume_range=(0.2, 0.5),
            seed=42
        )
        noisy_audio = augmentor.augment(clean_audio, target_idx=sample_idx)
    else:
        print("  Using white noise (SNR=5dB)")
        signal_power = torch.mean(clean_audio ** 2)
        noise = torch.randn_like(clean_audio)
        noise_power = torch.mean(noise ** 2)
        snr_db = 5
        snr_linear = 10 ** (snr_db / 10)
        noise_scaled = noise * torch.sqrt(signal_power / (snr_linear * noise_power))
        noisy_audio = clean_audio + noise_scaled

    # Convert to mel-spectrogram
    print("Converting to mel-spectrograms...")
    clean_mel = audio_processor.audio_to_mel(clean_audio)
    noisy_mel = audio_processor.audio_to_mel(noisy_audio)

    # Model inference
    print("Running model inference...")
    with torch.no_grad():
        enhanced_mel = model(noisy_mel.unsqueeze(0).to(device))
        enhanced_mel = enhanced_mel.squeeze(0)

    # Convert enhanced mel back to audio
    print("Reconstructing audio from enhanced mel-spectrogram...")
    enhanced_audio = audio_processor.mel_to_audio(enhanced_mel)
    enhanced_audio = enhanced_audio[:len(clean_audio)]  # Crop to original length

    # Convert to numpy
    clean_np = clean_audio.cpu().numpy()
    noisy_np = noisy_audio.cpu().numpy()
    enhanced_np = enhanced_audio.cpu().numpy()

    # Save audio files
    print("\nSaving audio files...")
    sf.write(output_dir / 'clean.wav', clean_np, 16000)
    print(f"  ✓ {output_dir}/clean.wav")

    sf.write(output_dir / 'noisy.wav', noisy_np, 16000)
    print(f"  ✓ {output_dir}/noisy.wav")

    sf.write(output_dir / 'enhanced.wav', enhanced_np, 16000)
    print(f"  ✓ {output_dir}/enhanced.wav")

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_comparison(
        clean_mel, noisy_mel, enhanced_mel,
        output_dir / 'mel_comparison.png'
    )

    plot_waveforms(
        clean_np, noisy_np, enhanced_np,
        output_dir / 'waveform_comparison.png'
    )

    # Compute metrics
    print("\n" + "="*70)
    print("METRICS")
    print("="*70)

    # SI-SNR
    def compute_si_snr(est, tgt):
        est = est - est.mean()
        tgt = tgt - tgt.mean()
        dot = (est * tgt).sum()
        tgt_norm = (tgt ** 2).sum()
        s_target = (dot / (tgt_norm + 1e-8)) * tgt
        e_noise = est - s_target
        si_snr = 10 * np.log10((s_target ** 2).sum() / ((e_noise ** 2).sum() + 1e-8))
        return si_snr

    si_snr_noisy = compute_si_snr(noisy_np, clean_np)
    si_snr_enhanced = compute_si_snr(enhanced_np, clean_np)

    print(f"SI-SNR (Noisy):    {si_snr_noisy:.4f} dB")
    print(f"SI-SNR (Enhanced): {si_snr_enhanced:.4f} dB")
    print(f"SI-SNR Improvement: {si_snr_enhanced - si_snr_noisy:+.4f} dB")
    print("="*70)

    print(f"\n✓ All outputs saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - clean.wav             (target clean audio)")
    print("  - noisy.wav             (noisy input)")
    print("  - enhanced.wav          (model output)")
    print("  - mel_comparison.png    (mel-spectrogram comparison)")
    print("  - waveform_comparison.png (waveform comparison)")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Quick Test: Generate before/after comparison')
    parser.add_argument('-m', '--model', type=str,
                        default='checkpoints/cnn_enhancer_best.pt',
                        help='Model checkpoint path')
    parser.add_argument('-o', '--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('-i', '--index', type=int, default=0,
                        help='Sample index to test')
    parser.add_argument('--no-cocktail', action='store_true',
                        help='Use white noise instead of cocktail party')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')

    args = parser.parse_args()

    quick_test(
        model_path=args.model,
        output_dir=args.output,
        sample_idx=args.index,
        use_cocktail=not args.no_cocktail,
        device=args.device
    )


if __name__ == "__main__":
    main()
