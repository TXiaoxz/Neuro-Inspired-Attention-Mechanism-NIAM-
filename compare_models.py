#!/usr/bin/env python3
"""
Compare CNN, RNN, and Transformer models
Generate comparison visualizations and metrics
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio
import numpy as np
from tqdm import tqdm
import argparse
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, '/home/dlwlx05/JHU_Course/MLSP/MLSP_Project')
from src.data.cocktail_augmentor import CocktailPartyAugmentor, HybridAugmentor
from src.utils.audio_utils import decode_audio

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# Model Definitions (same as evaluate.py)
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


class RNNEnhancer(nn.Module):
    def __init__(self, n_mels=80, hidden_dim=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(n_mels, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_mels)
        )

    def forward(self, x):
        batch_size, n_mels, time_steps = x.shape
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x, _ = self.gru(x)
        x = self.layer_norm(x)
        x = self.output_proj(x)
        x = x.transpose(1, 2)
        return x


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
    def __init__(self, n_mels=80, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model

        self.input_proj = nn.Linear(n_mels, d_model)
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
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_mels)
        )

    def forward(self, x):
        batch_size, n_mels, time_steps = x.shape
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.output_proj(x)
        x = x.transpose(1, 2)
        return x


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


class EvalDataset(Dataset):
    def __init__(self, hf_dataset, audio_processor, noise_aug, num_samples=500,
                 max_length=160000, use_cocktail=False):
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
# Metrics
# ============================================================================

def compute_si_snr(estimated, target):
    est_flat = estimated.reshape(estimated.shape[0], -1)
    tgt_flat = target.reshape(target.shape[0], -1)

    est_flat = est_flat - est_flat.mean(dim=1, keepdim=True)
    tgt_flat = tgt_flat - tgt_flat.mean(dim=1, keepdim=True)

    dot_product = torch.sum(est_flat * tgt_flat, dim=1, keepdim=True)
    target_norm = torch.sum(tgt_flat ** 2, dim=1, keepdim=True)
    s_target = (dot_product / (target_norm + 1e-8)) * tgt_flat
    e_noise = est_flat - s_target
    si_snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=1) /
                              (torch.sum(e_noise ** 2, dim=1) + 1e-8))

    return si_snr.mean().item()


def compute_snr(estimated, target):
    signal_power = torch.mean(target ** 2)
    noise_power = torch.mean((estimated - target) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    return snr.item()


def compute_mse(estimated, target):
    return F.mse_loss(estimated, target).item()


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, dataloader, device, model_name):
    model.eval()

    si_snr_scores = []
    snr_scores = []
    mse_scores = []
    si_snr_noisy_scores = []
    snr_noisy_scores = []

    with torch.no_grad():
        for noisy_mel, clean_mel in tqdm(dataloader, desc=f"Evaluating {model_name}"):
            noisy_mel = noisy_mel.to(device)
            clean_mel = clean_mel.to(device)

            enhanced_mel = model(noisy_mel)

            si_snr = compute_si_snr(enhanced_mel, clean_mel)
            snr = compute_snr(enhanced_mel, clean_mel)
            mse = compute_mse(enhanced_mel, clean_mel)

            si_snr_scores.append(si_snr)
            snr_scores.append(snr)
            mse_scores.append(mse)

            si_snr_noisy = compute_si_snr(noisy_mel, clean_mel)
            snr_noisy = compute_snr(noisy_mel, clean_mel)

            si_snr_noisy_scores.append(si_snr_noisy)
            snr_noisy_scores.append(snr_noisy)

    metrics = {
        'si_snr_enhanced': np.mean(si_snr_scores),
        'si_snr_noisy': np.mean(si_snr_noisy_scores),
        'si_snr_improvement': np.mean(si_snr_scores) - np.mean(si_snr_noisy_scores),
        'snr_enhanced': np.mean(snr_scores),
        'snr_noisy': np.mean(snr_noisy_scores),
        'snr_improvement': np.mean(snr_scores) - np.mean(snr_noisy_scores),
        'mse': np.mean(mse_scores),
    }

    return metrics


# ============================================================================
# Visualization
# ============================================================================

def plot_comparison(all_metrics, output_dir, use_cocktail):
    """Create comprehensive comparison plots"""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    models = list(all_metrics.keys())

    # Figure 1: SI-SNR Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # SI-SNR absolute values
    ax = axes[0]
    x = np.arange(len(models))
    width = 0.35

    noisy = [all_metrics[m]['si_snr_noisy'] for m in models]
    enhanced = [all_metrics[m]['si_snr_enhanced'] for m in models]

    bars1 = ax.bar(x - width/2, noisy, width, label='Noisy Input', color='#ff7f0e', alpha=0.8)
    bars2 = ax.bar(x + width/2, enhanced, width, label='Enhanced Output', color='#2ca02c', alpha=0.8)

    ax.set_ylabel('SI-SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_title('SI-SNR: Input vs Output', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)

    # SI-SNR Improvement
    ax = axes[1]
    improvements = [all_metrics[m]['si_snr_improvement'] for m in models]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    bars = ax.bar(models, improvements, color=colors, alpha=0.8)
    ax.set_ylabel('SI-SNR Improvement (dB)', fontsize=12, fontweight='bold')
    ax.set_title('SI-SNR Improvement by Model', fontsize=14, fontweight='bold')
    ax.set_xticklabels([m.upper() for m in models], fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'+{height:.2f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    aug_type = 'cocktail' if use_cocktail else 'traditional'
    plt.savefig(output_dir / f'si_snr_comparison_{aug_type}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / f'si_snr_comparison_{aug_type}.png'}")

    # Figure 2: All metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # SNR Improvement
    ax = axes[0, 0]
    snr_imp = [all_metrics[m]['snr_improvement'] for m in models]
    bars = ax.bar(models, snr_imp, color=colors, alpha=0.8)
    ax.set_ylabel('SNR Improvement (dB)', fontsize=11, fontweight='bold')
    ax.set_title('SNR Improvement', fontsize=12, fontweight='bold')
    ax.set_xticklabels([m.upper() for m in models])
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'+{height:.2f}', ha='center', va='bottom', fontsize=9)

    # MSE
    ax = axes[0, 1]
    mse = [all_metrics[m]['mse'] for m in models]
    bars = ax.bar(models, mse, color=colors, alpha=0.8)
    ax.set_ylabel('MSE (lower is better)', fontsize=11, fontweight='bold')
    ax.set_title('Mean Squared Error', fontsize=12, fontweight='bold')
    ax.set_xticklabels([m.upper() for m in models])
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # Overall Performance Radar (SI-SNR and SNR improvements)
    ax = axes[1, 0]
    x = np.arange(len(models))
    width = 0.35

    si_snr_imp = [all_metrics[m]['si_snr_improvement'] for m in models]
    snr_imp = [all_metrics[m]['snr_improvement'] for m in models]

    ax.bar(x - width/2, si_snr_imp, width, label='SI-SNR Improvement', color='#1f77b4', alpha=0.8)
    ax.bar(x + width/2, snr_imp, width, label='SNR Improvement', color='#ff7f0e', alpha=0.8)

    ax.set_ylabel('Improvement (dB)', fontsize=11, fontweight='bold')
    ax.set_title('SI-SNR vs SNR Improvement', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Summary table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    headers = ['Model', 'SI-SNR↑', 'SNR↑', 'MSE↓']
    for m in models:
        table_data.append([
            m.upper(),
            f"+{all_metrics[m]['si_snr_improvement']:.2f}",
            f"+{all_metrics[m]['snr_improvement']:.2f}",
            f"{all_metrics[m]['mse']:.2f}"
        ])

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colColours=['#f0f0f0']*4)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Highlight best values
    for i, m in enumerate(models):
        for j in range(1, 4):
            table[(i+1, j)].set_facecolor('#e8f4f8')

    ax.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / f'all_metrics_comparison_{aug_type}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / f'all_metrics_comparison_{aug_type}.png'}")

    plt.close('all')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compare CNN, RNN, and Transformer Models')
    parser.add_argument('-n', '--num_samples', type=int, default=500,
                        help='Number of samples to evaluate')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--cocktail', action='store_true',
                        help='Use cocktail party augmentation')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('-o', '--output', type=str, default='results',
                        help='Output directory for plots')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Model configurations
    model_configs = {
        'cnn': {
            'class': CNNEnhancer,
            'path': 'checkpoints/cnn_enhancer_best.pt',
            'kwargs': {}
        },
        'rnn': {
            'class': RNNEnhancer,
            'path': 'checkpoints/rnn_enhancer_best.pt',
            'kwargs': {'n_mels': 80, 'hidden_dim': 256, 'num_layers': 3, 'dropout': 0.2}
        },
        'transformer': {
            'class': TransformerEnhancer,
            'path': 'checkpoints/transformer_enhancer_best.pt',
            'kwargs': {'n_mels': 80, 'd_model': 256, 'nhead': 8, 'num_layers': 4,
                      'dim_feedforward': 1024, 'dropout': 0.1}
        }
    }

    # Load dataset
    print("Loading dataset...")
    ds_val = load_dataset("MLCommons/peoples_speech", "validation", cache_dir="./data/cache")
    ds_val = ds_val.cast_column("audio", Audio(decode=False))
    dataset = ds_val['validation']

    # Create audio processor and augmentor
    print("Creating audio processor and augmentor...")
    audio_processor = AudioProcessor()

    if args.cocktail:
        print("Using cocktail party augmentation\n")
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
        print("Using traditional noise augmentation\n")
        noise_aug = NoiseAugmentor()

    # Create dataset
    print("Creating evaluation dataset...")
    eval_dataset = EvalDataset(
        dataset, audio_processor, noise_aug,
        num_samples=args.num_samples,
        use_cocktail=args.cocktail
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Evaluate all models
    all_metrics = {}

    print("\n" + "="*70)
    print("EVALUATING ALL MODELS")
    print("="*70 + "\n")

    for model_name, config in model_configs.items():
        print(f"\n--- {model_name.upper()} ---")

        # Load model
        model = config['class'](**config['kwargs']).to(device)
        try:
            model.load_state_dict(torch.load(config['path'], map_location=device))
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✓ Loaded from {config['path']}")
            print(f"  Parameters: {total_params:,}")
        except FileNotFoundError:
            print(f"✗ Model file not found: {config['path']}")
            print(f"  Skipping {model_name.upper()}")
            continue

        # Evaluate
        metrics = evaluate_model(model, eval_loader, device, model_name.upper())
        all_metrics[model_name] = metrics

        # Print results
        print(f"\n{model_name.upper()} Results:")
        print(f"  SI-SNR Improvement: +{metrics['si_snr_improvement']:.4f} dB")
        print(f"  SNR Improvement:    +{metrics['snr_improvement']:.4f} dB")
        print(f"  MSE:                 {metrics['mse']:.6f}")

    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"Augmentation: {'Cocktail Party' if args.cocktail else 'Traditional Noise'}")
    print(f"Samples: {args.num_samples}\n")

    print(f"{'Model':<15} {'SI-SNR Imp.':<15} {'SNR Imp.':<15} {'MSE':<15}")
    print("-" * 60)
    for model_name in all_metrics:
        metrics = all_metrics[model_name]
        print(f"{model_name.upper():<15} "
              f"+{metrics['si_snr_improvement']:<14.4f} "
              f"+{metrics['snr_improvement']:<14.4f} "
              f"{metrics['mse']:<15.6f}")

    # Find best model
    best_si_snr = max(all_metrics.items(), key=lambda x: x[1]['si_snr_improvement'])
    print(f"\n🏆 Best SI-SNR Improvement: {best_si_snr[0].upper()} (+{best_si_snr[1]['si_snr_improvement']:.4f} dB)")

    best_snr = max(all_metrics.items(), key=lambda x: x[1]['snr_improvement'])
    print(f"🏆 Best SNR Improvement: {best_snr[0].upper()} (+{best_snr[1]['snr_improvement']:.4f} dB)")

    best_mse = min(all_metrics.items(), key=lambda x: x[1]['mse'])
    print(f"🏆 Best MSE: {best_mse[0].upper()} ({best_mse[1]['mse']:.6f})")

    print("="*70)

    # Generate plots
    print(f"\nGenerating comparison plots...")
    plot_comparison(all_metrics, args.output, args.cocktail)

    print(f"\n✓ All comparisons complete!")
    print(f"  Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
