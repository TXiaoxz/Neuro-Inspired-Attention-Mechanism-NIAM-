#!/usr/bin/env python3
"""
Audio Enhancement Inference Script
Process single audio files using trained CNN model
"""
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
import argparse
import os
from pathlib import Path

# ============================================================================
# Model Definition (same as train_fast.py)
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
# Audio Enhancement Class
# ============================================================================

class AudioEnhancer:
    def __init__(self, model_path, device='cuda', sample_rate=16000):
        """
        Initialize audio enhancer

        Args:
            model_path: Path to model checkpoint
            device: 'cuda' or 'cpu'
            sample_rate: Audio sample rate
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sample_rate = sample_rate

        # Load model
        print(f"Loading model: {model_path}")
        self.model = CNNEnhancer().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")

        # Create mel-spectrogram transform
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=80,
            power=2.0
        ).to(self.device)

        # Create inverse mel-spectrogram transform
        self.inverse_mel = T.InverseMelScale(
            n_stft=201,  # n_fft // 2 + 1
            n_mels=80,
            sample_rate=sample_rate
        ).to(self.device)

        # Griffin-Lim for phase reconstruction
        self.griffin_lim = T.GriffinLim(
            n_fft=400,
            hop_length=160,
            power=2.0,
            n_iter=32
        ).to(self.device)

    def load_audio(self, audio_path):
        """Load audio file"""
        audio, sr = torchaudio.load(audio_path)

        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)

        return audio.squeeze(0)

    def audio_to_mel(self, waveform):
        """Convert audio to mel-spectrogram"""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.mel_spec(waveform)
        return torch.log(mel + 1e-9).squeeze(0)

    def mel_to_audio(self, mel_spec):
        """Convert mel-spectrogram to audio using Griffin-Lim"""
        # Convert to linear scale
        mel_linear = torch.exp(mel_spec) - 1e-9

        # Mel -> STFT
        stft = self.inverse_mel(mel_linear)

        # STFT -> Audio (Griffin-Lim phase reconstruction)
        audio = self.griffin_lim(stft)

        return audio

    def enhance(self, audio_path, output_path=None):
        """
        Enhance a single audio file

        Args:
            audio_path: Path to input audio
            output_path: Path to output audio (optional)

        Returns:
            enhanced_audio: Enhanced audio as numpy array
        """
        print(f"\nProcessing audio: {audio_path}")

        # Load audio
        noisy_audio = self.load_audio(audio_path)
        original_length = len(noisy_audio)

        # Convert to mel-spectrogram
        noisy_mel = self.audio_to_mel(noisy_audio).unsqueeze(0).to(self.device)

        # Model inference
        print("Running model inference...")
        with torch.no_grad():
            enhanced_mel = self.model(noisy_mel)

        # Convert back to audio
        print("Reconstructing audio waveform...")
        enhanced_audio = self.mel_to_audio(enhanced_mel.squeeze(0))

        # Truncate to original length
        enhanced_audio = enhanced_audio[:original_length]

        # Convert to numpy
        enhanced_audio = enhanced_audio.cpu().numpy()

        # Save enhanced audio
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            sf.write(output_path, enhanced_audio, self.sample_rate)
            print(f"✓ Saved enhanced audio: {output_path}")

        return enhanced_audio


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Audio Enhancement Inference')
    parser.add_argument('input', type=str, help='Input audio file path')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output audio file path (default: <input>_enhanced.wav)')
    parser.add_argument('-m', '--model', type=str,
                        default='checkpoints/cnn_enhancer_best.pt',
                        help='Model checkpoint path')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')

    args = parser.parse_args()

    # Check input file
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return

    # Set output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_enhanced.wav")

    # Create enhancer
    enhancer = AudioEnhancer(args.model, device=args.device)

    # Process audio
    enhanced_audio = enhancer.enhance(args.input, args.output)

    print(f"\n✓ Processing complete!")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Duration: {len(enhanced_audio) / 16000:.2f}s")


if __name__ == "__main__":
    main()
