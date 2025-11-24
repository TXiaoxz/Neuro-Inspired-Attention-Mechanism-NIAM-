#!/usr/bin/env python3
"""
Create Audio Comparison Visualizations

Generates:
1. Waveform comparisons (clean vs noisy vs baseline vs NIAM v2)
2. Spectrogram comparisons
3. HTML page with embedded audio players for easy listening
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Paths
BASELINE_DIR = '/home/dlwlx05/project/NIAM/Neuro-Inspired-Attention-Mechanism-NIAM--main/results/test_samples'
NIAM_DIR = '/home/dlwlx05/project/NIAM/Neuro-Inspired-Attention-Mechanism-NIAM--main/results/niam_v2/test_samples'
OUTPUT_DIR = '/home/dlwlx05/project/NIAM/results/audio_comparison'

def create_waveform_comparison(sample_idx, output_dir):
    """Create waveform comparison plot"""

    # Load audio files
    clean_path = f"{BASELINE_DIR}/sample_{sample_idx}_clean.wav"
    noisy_path = f"{BASELINE_DIR}/sample_{sample_idx}_noisy.wav"
    baseline_path = f"{BASELINE_DIR}/sample_{sample_idx}_enhanced.wav"
    niam_path = f"{NIAM_DIR}/sample_{sample_idx}_enhanced.wav"

    clean, sr = sf.read(clean_path)
    noisy, _ = sf.read(noisy_path)
    baseline, _ = sf.read(baseline_path)
    niam, _ = sf.read(niam_path)

    # Create time axis
    time = np.arange(len(clean)) / sr

    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 10))
    fig.suptitle(f'Sample {sample_idx} - Waveform Comparison', fontsize=16, fontweight='bold')

    # Plot clean
    axes[0].plot(time, clean, color='green', linewidth=0.5)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Clean Audio (Ground Truth)', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-1, 1)

    # Plot noisy
    axes[1].plot(time, noisy, color='red', linewidth=0.5)
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Noisy Audio (Input)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-1, 1)

    # Plot baseline
    axes[2].plot(time, baseline, color='blue', linewidth=0.5)
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Baseline Enhanced (Transformer only)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-1, 1)

    # Plot NIAM v2
    axes[3].plot(time, niam, color='purple', linewidth=0.5)
    axes[3].set_ylabel('Amplitude')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title('NIAM v2 Enhanced (Transformer + NIAM)', fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim(-1, 1)

    plt.tight_layout()

    # Save
    output_path = f"{output_dir}/sample_{sample_idx}_waveform.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Waveform saved: {output_path}")

    return output_path


def create_spectrogram_comparison(sample_idx, output_dir):
    """Create spectrogram comparison plot"""

    # Load audio files
    clean_path = f"{BASELINE_DIR}/sample_{sample_idx}_clean.wav"
    noisy_path = f"{BASELINE_DIR}/sample_{sample_idx}_noisy.wav"
    baseline_path = f"{BASELINE_DIR}/sample_{sample_idx}_enhanced.wav"
    niam_path = f"{NIAM_DIR}/sample_{sample_idx}_enhanced.wav"

    clean, sr = sf.read(clean_path)
    noisy, _ = sf.read(noisy_path)
    baseline, _ = sf.read(baseline_path)
    niam, _ = sf.read(niam_path)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Sample {sample_idx} - Spectrogram Comparison', fontsize=16, fontweight='bold')

    # Spectrogram parameters
    nfft = 1024
    hop_length = 256

    # Plot clean
    axes[0, 0].specgram(clean, NFFT=nfft, Fs=sr, noverlap=nfft-hop_length, cmap='viridis')
    axes[0, 0].set_ylabel('Frequency (Hz)')
    axes[0, 0].set_title('Clean Audio', fontweight='bold', color='green')
    axes[0, 0].set_ylim(0, 8000)

    # Plot noisy
    axes[0, 1].specgram(noisy, NFFT=nfft, Fs=sr, noverlap=nfft-hop_length, cmap='viridis')
    axes[0, 1].set_title('Noisy Audio', fontweight='bold', color='red')
    axes[0, 1].set_ylim(0, 8000)

    # Plot baseline
    axes[1, 0].specgram(baseline, NFFT=nfft, Fs=sr, noverlap=nfft-hop_length, cmap='viridis')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_title('Baseline Enhanced', fontweight='bold', color='blue')
    axes[1, 0].set_ylim(0, 8000)

    # Plot NIAM v2
    axes[1, 1].specgram(niam, NFFT=nfft, Fs=sr, noverlap=nfft-hop_length, cmap='viridis')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_title('NIAM v2 Enhanced', fontweight='bold', color='purple')
    axes[1, 1].set_ylim(0, 8000)

    plt.tight_layout()

    # Save
    output_path = f"{output_dir}/sample_{sample_idx}_spectrogram.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Spectrogram saved: {output_path}")

    return output_path


def create_html_player(num_samples, output_dir):
    """Create HTML page with audio players"""

    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Comparison: Baseline vs NIAM v2</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .summary {
            background-color: #fff;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sample-container {
            background-color: #fff;
            margin: 30px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sample-header {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ddd;
            color: #333;
        }
        .audio-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 20px 0;
        }
        .audio-item {
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            background-color: #fafafa;
        }
        .audio-item.clean {
            border-color: #4CAF50;
        }
        .audio-item.noisy {
            border-color: #f44336;
        }
        .audio-item.baseline {
            border-color: #2196F3;
        }
        .audio-item.niam {
            border-color: #9C27B0;
        }
        .audio-label {
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        .audio-item.clean .audio-label {
            color: #4CAF50;
        }
        .audio-item.noisy .audio-label {
            color: #f44336;
        }
        .audio-item.baseline .audio-label {
            color: #2196F3;
        }
        .audio-item.niam .audio-label {
            color: #9C27B0;
        }
        audio {
            width: 100%;
            margin: 10px 0;
        }
        .metadata {
            background-color: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            font-size: 0.95em;
        }
        .visualization {
            margin: 20px 0;
        }
        .visualization img {
            width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .stats-table th, .stats-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .stats-table th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .stats-table tr:hover {
            background-color: #f5f5f5;
        }
        .better {
            color: #4CAF50;
            font-weight: bold;
        }
        .worse {
            color: #f44336;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>🎧 Audio Enhancement Comparison: Baseline vs NIAM v2</h1>

    <div class="summary">
        <h2>📊 Overall Results Summary</h2>
        <table class="stats-table">
            <tr>
                <th>Metric</th>
                <th>Baseline</th>
                <th>NIAM v2</th>
                <th>Difference</th>
            </tr>
            <tr>
                <td>Enhanced WER</td>
                <td>56.0%</td>
                <td>55.5%</td>
                <td class="better">-0.5% ↓</td>
            </tr>
            <tr>
                <td>Enhanced CER</td>
                <td>42.1%</td>
                <td>41.2%</td>
                <td class="better">-0.9% ↓</td>
            </tr>
            <tr>
                <td>Samples Improved</td>
                <td>3/10 (30%)</td>
                <td>3/10 (30%)</td>
                <td>Same</td>
            </tr>
        </table>
        <p><strong>Conclusion:</strong> NIAM v2 shows slight improvement over baseline with lower WER and CER.</p>
    </div>
"""

    # Add samples
    for i in range(num_samples):
        # Read metadata
        metadata_path = f"{BASELINE_DIR}/sample_{i}_metadata.txt"
        with open(metadata_path, 'r') as f:
            metadata = f.read()

        # Extract text
        text = ""
        for line in metadata.split('\n'):
            if line.startswith('Text:'):
                text = line.replace('Text:', '').strip()
                break

        html_content += f"""
    <div class="sample-container">
        <div class="sample-header">Sample {i}</div>

        <div class="metadata">
            <strong>Ground Truth Text:</strong><br>
            {text}
        </div>

        <div class="audio-grid">
            <div class="audio-item clean">
                <div class="audio-label">🟢 Clean Audio (Ground Truth)</div>
                <audio controls>
                    <source src="{BASELINE_DIR}/sample_{i}_clean.wav" type="audio/wav">
                </audio>
            </div>

            <div class="audio-item noisy">
                <div class="audio-label">🔴 Noisy Audio (Input)</div>
                <audio controls>
                    <source src="{BASELINE_DIR}/sample_{i}_noisy.wav" type="audio/wav">
                </audio>
            </div>

            <div class="audio-item baseline">
                <div class="audio-label">🔵 Baseline Enhanced (Transformer only)</div>
                <audio controls>
                    <source src="{BASELINE_DIR}/sample_{i}_enhanced.wav" type="audio/wav">
                </audio>
            </div>

            <div class="audio-item niam">
                <div class="audio-label">🟣 NIAM v2 Enhanced (Transformer + NIAM)</div>
                <audio controls>
                    <source src="{NIAM_DIR}/sample_{i}_enhanced.wav" type="audio/wav">
                </audio>
            </div>
        </div>

        <div class="visualization">
            <h3>Waveform Comparison</h3>
            <img src="sample_{i}_waveform.png" alt="Waveform comparison">
        </div>

        <div class="visualization">
            <h3>Spectrogram Comparison</h3>
            <img src="sample_{i}_spectrogram.png" alt="Spectrogram comparison">
        </div>
    </div>
"""

    html_content += """
    <div class="summary">
        <h2>🔬 Technical Details</h2>
        <ul>
            <li><strong>Dataset:</strong> MLCommons People's Speech microset</li>
            <li><strong>Noise Type:</strong> Cocktail Party (5 interferers, SNR=8dB)</li>
            <li><strong>Baseline Model:</strong> Transformer (4 layers, 256 hidden dim, 8 attention heads)</li>
            <li><strong>NIAM v2:</strong> Baseline + 4 bio-inspired attention modules (Selective Attention, Frequency Tuning, Temporal Focus, Noise Adaptation)</li>
            <li><strong>Training:</strong> 270 samples, 10 epochs, SI-SNR loss</li>
            <li><strong>Evaluation:</strong> Whisper ASR (base model) - WER and CER metrics</li>
        </ul>
    </div>
</body>
</html>
"""

    # Save HTML
    html_path = f"{output_dir}/audio_comparison.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n✓ HTML player saved: {html_path}")
    return html_path


def main():
    """Generate all comparisons"""

    print("="*80)
    print("CREATING AUDIO COMPARISONS")
    print("="*80)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Check how many samples we have
    num_samples = len(list(Path(BASELINE_DIR).glob("sample_*_clean.wav")))
    print(f"Found {num_samples} samples\n")

    # Generate visualizations for each sample
    for i in range(num_samples):
        print(f"Processing sample {i}...")
        create_waveform_comparison(i, OUTPUT_DIR)
        create_spectrogram_comparison(i, OUTPUT_DIR)

    # Create HTML player
    print("\nCreating HTML audio player...")
    html_path = create_html_player(num_samples, OUTPUT_DIR)

    print("\n" + "="*80)
    print("✅ DONE!")
    print("="*80)
    print(f"\nOpen this file in your browser:")
    print(f"  {html_path}")
    print("\nYou can:")
    print("  1. Listen to all audio samples side-by-side")
    print("  2. Compare waveforms visually")
    print("  3. Compare spectrograms")
    print("  4. See ground truth transcriptions")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
