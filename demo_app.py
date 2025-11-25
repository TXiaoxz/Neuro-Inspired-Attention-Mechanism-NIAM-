#!/usr/bin/env python3
"""
NIAM v2 Refined - Interactive Web Demo
Audio Enhancement with Neuro-Inspired Attention Mechanism
"""

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import sys
import os
import math
import tempfile

# Add project root to path and load config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PROJECT_ROOT, NIAM_V2_REFINED_MODEL, RESULTS_DIR
from niam_v2 import NIAM

# ============================================================================
# Model Definition (must match training)
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
    def __init__(self, n_freq_bins=201, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.1, use_niam=False,
                 refinement_mode=False, refinement_alpha=0.2):
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
            self.niam = NIAM(hidden_dim=d_model, refinement_mode=refinement_mode,
                           refinement_alpha=refinement_alpha)
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

        transformer_features = x.clone() if self.refinement_mode and self.niam is not None else None

        if self.niam is not None:
            if self.refinement_mode:
                x, _, _ = self.niam(x)
            else:
                x = self.niam(x)

        mask = self.output_proj(x)
        mask = mask.transpose(1, 2)

        if self.refinement_mode and transformer_features is not None:
            transformer_mask = self.output_proj(transformer_features)
            transformer_mask = transformer_mask.transpose(1, 2)
            return mask, transformer_mask
        else:
            return mask


# ============================================================================
# Audio Enhancer
# ============================================================================

class AudioEnhancer:
    def __init__(self, model_path, device='cuda', n_fft=400, hop_length=160,
                 sample_rate=16000, refinement_mode=True, refinement_alpha=0.2):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.refinement_mode = refinement_mode
        self.window = torch.hann_window(n_fft).to(self.device)

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
        print(f"Model loaded on {self.device}")

    def enhance(self, audio_input, input_sr):
        """
        Enhance audio from file or microphone

        Args:
            audio_input: (sample_rate, audio_data) tuple from Gradio
            input_sr: Input sample rate

        Returns:
            (sample_rate, enhanced_audio) tuple for Gradio
        """
        # Handle Gradio audio input format
        if isinstance(audio_input, tuple):
            input_sr, audio_data = audio_input
        else:
            audio_data = audio_input

        # Convert to float32 and normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        else:
            audio_data = audio_data.astype(np.float32)

        # Handle stereo by taking mean
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 16kHz if needed
        if input_sr != self.sample_rate:
            duration = len(audio_data) / input_sr
            new_length = int(duration * self.sample_rate)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), new_length),
                np.arange(len(audio_data)),
                audio_data
            )

        # Convert to torch tensor
        noisy_audio = torch.FloatTensor(audio_data).unsqueeze(0).to(self.device)
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

        noisy_mag = torch.abs(noisy_stft)
        noisy_phase = torch.angle(noisy_stft)

        # Model inference
        with torch.no_grad():
            if self.refinement_mode:
                mask, _ = self.model(noisy_mag)
            else:
                mask = self.model(noisy_mag)

        # Apply mask
        enhanced_mag = mask * noisy_mag
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

        return (self.sample_rate, enhanced_audio)


# ============================================================================
# Gradio Interface
# ============================================================================

# Initialize model
MODEL_PATH = NIAM_V2_REFINED_MODEL
print("\n" + "="*80)
print("NIAM v2 Refined - Audio Enhancement Demo")
print("="*80)

enhancer = AudioEnhancer(MODEL_PATH, refinement_mode=True, refinement_alpha=0.2)

# Load fixed noise file
print("\nLoading fixed cocktail party noise file...")
NOISE_FILE = os.path.join(RESULTS_DIR, 'niam_v2_refined', 'demo_noise_30s.wav')
try:
    fixed_noise, noise_sr = sf.read(NOISE_FILE)
    if noise_sr != 16000:
        print(f"Warning: Noise file sample rate is {noise_sr}Hz, expected 16000Hz")
    print(f"Fixed noise loaded: {len(fixed_noise)/16000:.1f}s, RMS={np.sqrt(np.mean(fixed_noise**2)):.4f}")
except Exception as e:
    print(f"Warning: Could not load noise file: {e}")
    fixed_noise = None

print("\nDemo ready!")
print("="*80 + "\n")


def add_fixed_noise(clean_audio, snr_db=8):
    """
    Add fixed cocktail party noise to clean audio

    Args:
        clean_audio: Clean audio array (16kHz)
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Noisy audio array
    """
    if fixed_noise is None:
        print("Warning: No noise file loaded, returning original audio")
        return clean_audio

    target_length = len(clean_audio)
    noise_length = len(fixed_noise)

    # Crop or loop noise to match audio length
    if target_length <= noise_length:
        # Crop noise
        noise_segment = fixed_noise[:target_length]
    else:
        # Loop noise
        repeats = int(np.ceil(target_length / noise_length))
        noise_segment = np.tile(fixed_noise, repeats)[:target_length]

    # Calculate scaling factor for desired SNR
    signal_power = np.mean(clean_audio ** 2)
    noise_power = np.mean(noise_segment ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))

    # Mix
    noisy_audio = clean_audio + noise_scale * noise_segment

    return noisy_audio


def enhance_audio(audio_input, add_noise):
    """Gradio callback function"""
    if audio_input is None:
        return None, None

    try:
        input_sr, audio_data = audio_input

        # Convert to float32 and normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        else:
            audio_data = audio_data.astype(np.float32)

        # Handle stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 16kHz if needed
        if input_sr != 16000:
            duration = len(audio_data) / input_sr
            new_length = int(duration * 16000)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), new_length),
                np.arange(len(audio_data)),
                audio_data
            )
            input_sr = 16000

        # Add fixed cocktail party noise if requested
        if add_noise:
            print("Adding fixed cocktail party noise (5 speakers, SNR=8dB)...")
            noisy_audio = add_fixed_noise(audio_data, snr_db=8)
        else:
            noisy_audio = audio_data

        # Enhance
        enhanced = enhancer.enhance((input_sr, noisy_audio), input_sr=input_sr)

        # First output shows current audio (with or without noise)
        current_output = (input_sr, noisy_audio)

        return current_output, enhanced
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# Create Gradio interface
with gr.Blocks(title="NIAM v2 Refined") as demo:
    gr.Markdown("""
# NIAM v2 Refined: Neuro-Inspired Speech Enhancement

A deep learning model that enhances noisy speech using biologically-inspired attention mechanisms.

## Model Architecture
- Transformer encoder with 4 layers
- Neuro-inspired attention with 4 specialized modules
- Residual refinement mode for temporal smoothness
- Trained on 50,000 speech samples with cocktail party noise

## Performance Metrics
- Word Error Rate improvement: 5.6% (53.3% to 50.3%)
- Best validation loss: -17.5026 dB SI-SNR
- Learned module weights: [0.307, 0.044, 0.003, 0.500]
    """)

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="numpy",
                label="Input Audio"
            )
            add_noise_checkbox = gr.Checkbox(
                label="Add Noise",
                value=False,
                info="Add fixed cocktail party noise (5 speakers, SNR=8dB)"
            )
            submit_btn = gr.Button("Enhance Audio", variant="primary")

        with gr.Column():
            current_audio = gr.Audio(
                type="numpy",
                label="Current Audio"
            )
            enhanced_audio = gr.Audio(
                type="numpy",
                label="Enhanced Audio"
            )

    submit_btn.click(
        fn=enhance_audio,
        inputs=[audio_input, add_noise_checkbox],
        outputs=[current_audio, enhanced_audio]
    )

    test_samples_dir = os.path.join(RESULTS_DIR, 'niam_v2_refined', 'test_samples')
    if os.path.exists(os.path.join(test_samples_dir, 'sample_0_noisy.wav')):
        gr.Markdown("## Example Audio Samples")
        gr.Examples(
            examples=[
                [os.path.join(test_samples_dir, 'sample_6_noisy.wav')],
                [os.path.join(test_samples_dir, 'sample_5_noisy.wav')],
                [os.path.join(test_samples_dir, 'sample_4_noisy.wav')],
            ],
            inputs=audio_input
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
