#!/usr/bin/env python3
"""
Whisper-based Evaluation Script

Evaluates audio enhancement quality using OpenAI Whisper ASR.

Metrics:
- WER (Word Error Rate): Lower is better
- Character Error Rate (CER): Lower is better
- Transcription accuracy comparison

Evaluation flow:
1. Clean audio → Whisper → Ground truth transcription
2. Noisy audio → Whisper → Noisy transcription (baseline)
3. Enhanced audio → Whisper → Enhanced transcription (our model)

Compare WER/CER to measure improvement.
"""

import whisper
import torch
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from jiwer import wer, cer
import argparse
import os
import sys
import time
from tqdm import tqdm

PROJECT_ROOT = '/home/dlwlx05/project/NIAM/Neuro-Inspired-Attention-Mechanism-NIAM--main'


def evaluate_with_whisper(test_samples_dir, model_size='base', output_csv=None):
    """
    Evaluate test samples using Whisper

    Args:
        test_samples_dir: Directory containing test samples (clean/noisy/enhanced triplets)
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        output_csv: Path to save evaluation results CSV
    """
    print("="*80)
    print("WHISPER-BASED AUDIO ENHANCEMENT EVALUATION")
    print("="*80)

    # Check if test samples exist
    if not os.path.exists(test_samples_dir):
        print(f"Error: Test samples directory not found: {test_samples_dir}")
        print("Please run inference_transformer.py generate_test first!")
        return

    # Load Whisper model
    print(f"\nLoading Whisper '{model_size}' model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    whisper_model = whisper.load_model(model_size, device=device)
    print(f"✓ Whisper model loaded on {device}")

    # Find all test samples
    test_samples_dir = Path(test_samples_dir)
    metadata_files = sorted(test_samples_dir.glob("sample_*_metadata.txt"))

    if len(metadata_files) == 0:
        print(f"Error: No test samples found in {test_samples_dir}")
        return

    print(f"\nFound {len(metadata_files)} test samples")

    # Evaluate each sample
    results = []

    print("\nEvaluating samples...")
    for metadata_file in tqdm(metadata_files):
        sample_id = metadata_file.stem.replace("_metadata", "")

        # Read metadata
        with open(metadata_file, 'r') as f:
            lines = f.readlines()
            ground_truth_text = None
            for line in lines:
                if line.startswith("Text:"):
                    ground_truth_text = line.replace("Text:", "").strip()
                    break

        if ground_truth_text is None:
            print(f"Warning: No ground truth text found for {sample_id}")
            continue

        # Paths to audio files
        clean_path = test_samples_dir / f"{sample_id}_clean.wav"
        noisy_path = test_samples_dir / f"{sample_id}_noisy.wav"
        enhanced_path = test_samples_dir / f"{sample_id}_enhanced.wav"

        # Check if all files exist
        if not all([clean_path.exists(), noisy_path.exists(), enhanced_path.exists()]):
            print(f"Warning: Missing audio files for {sample_id}")
            continue

        # Transcribe with Whisper
        try:
            clean_result = whisper_model.transcribe(str(clean_path), language="en")
            noisy_result = whisper_model.transcribe(str(noisy_path), language="en")
            enhanced_result = whisper_model.transcribe(str(enhanced_path), language="en")

            clean_transcription = clean_result["text"].strip()
            noisy_transcription = noisy_result["text"].strip()
            enhanced_transcription = enhanced_result["text"].strip()

        except Exception as e:
            print(f"Error transcribing {sample_id}: {e}")
            continue

        # Calculate WER and CER
        # Use ground truth as reference
        wer_clean = wer(ground_truth_text, clean_transcription)
        wer_noisy = wer(ground_truth_text, noisy_transcription)
        wer_enhanced = wer(ground_truth_text, enhanced_transcription)

        cer_clean = cer(ground_truth_text, clean_transcription)
        cer_noisy = cer(ground_truth_text, noisy_transcription)
        cer_enhanced = cer(ground_truth_text, enhanced_transcription)

        # Calculate improvement
        wer_improvement = wer_noisy - wer_enhanced  # Positive = better
        cer_improvement = cer_noisy - cer_enhanced

        # Store results
        results.append({
            'sample_id': sample_id,
            'ground_truth': ground_truth_text,
            'clean_transcription': clean_transcription,
            'noisy_transcription': noisy_transcription,
            'enhanced_transcription': enhanced_transcription,
            'wer_clean': wer_clean,
            'wer_noisy': wer_noisy,
            'wer_enhanced': wer_enhanced,
            'wer_improvement': wer_improvement,
            'cer_clean': cer_clean,
            'cer_noisy': cer_noisy,
            'cer_enhanced': cer_enhanced,
            'cer_improvement': cer_improvement
        })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Calculate summary statistics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    print(f"\nSamples evaluated: {len(df)}")

    print("\n--- Word Error Rate (WER) ---")
    print(f"Clean audio:     {df['wer_clean'].mean():.1%} ± {df['wer_clean'].std():.1%}")
    print(f"Noisy audio:     {df['wer_noisy'].mean():.1%} ± {df['wer_noisy'].std():.1%}")
    print(f"Enhanced audio:  {df['wer_enhanced'].mean():.1%} ± {df['wer_enhanced'].std():.1%}")
    print(f"Improvement:     {df['wer_improvement'].mean():.1%} ± {df['wer_improvement'].std():.1%}")

    print("\n--- Character Error Rate (CER) ---")
    print(f"Clean audio:     {df['cer_clean'].mean():.1%} ± {df['cer_clean'].std():.1%}")
    print(f"Noisy audio:     {df['cer_noisy'].mean():.1%} ± {df['cer_noisy'].std():.1%}")
    print(f"Enhanced audio:  {df['cer_enhanced'].mean():.1%} ± {df['cer_enhanced'].std():.1%}")
    print(f"Improvement:     {df['cer_improvement'].mean():.1%} ± {df['cer_improvement'].std():.1%}")

    # Relative improvement
    if df['wer_noisy'].mean() > 0:
        relative_wer_improvement = (df['wer_noisy'].mean() - df['wer_enhanced'].mean()) / df['wer_noisy'].mean()
        print(f"\n--- Relative WER Improvement ---")
        print(f"  {relative_wer_improvement:.1%} (from {df['wer_noisy'].mean():.1%} to {df['wer_enhanced'].mean():.1%})")

    # Count how many samples improved
    num_improved_wer = (df['wer_improvement'] > 0).sum()
    num_degraded_wer = (df['wer_improvement'] < 0).sum()
    num_same_wer = (df['wer_improvement'] == 0).sum()

    print(f"\n--- Sample-wise WER Analysis ---")
    print(f"  Improved:  {num_improved_wer}/{len(df)} ({num_improved_wer/len(df)*100:.1f}%)")
    print(f"  Degraded:  {num_degraded_wer}/{len(df)} ({num_degraded_wer/len(df)*100:.1f}%)")
    print(f"  Same:      {num_same_wer}/{len(df)} ({num_same_wer/len(df)*100:.1f}%)")

    # Save to CSV
    if output_csv is None:
        output_csv = test_samples_dir / "whisper_evaluation_results.csv"

    df.to_csv(output_csv, index=False)
    print(f"\n✓ Detailed results saved to: {output_csv}")

    # Print a few examples
    print("\n" + "="*80)
    print("SAMPLE TRANSCRIPTIONS (First 3 samples)")
    print("="*80)

    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"\n--- Sample {i} ({row['sample_id']}) ---")
        print(f"Ground Truth: {row['ground_truth'][:100]}...")
        print(f"Clean:        {row['clean_transcription'][:100]}...")
        print(f"Noisy:        {row['noisy_transcription'][:100]}...")
        print(f"Enhanced:     {row['enhanced_transcription'][:100]}...")
        print(f"WER: Clean={row['wer_clean']:.1%}, Noisy={row['wer_noisy']:.1%}, Enhanced={row['wer_enhanced']:.1%}")

    print("\n" + "="*80)

    return df


def main():
    parser = argparse.ArgumentParser(description='Evaluate audio enhancement using Whisper ASR')
    parser.add_argument('-d', '--test_dir', type=str,
                        default=f'{PROJECT_ROOT}/results/test_samples',
                        help='Directory containing test samples')
    parser.add_argument('-m', '--model', type=str, default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output CSV file path')

    args = parser.parse_args()

    evaluate_with_whisper(args.test_dir, args.model, args.output)


if __name__ == "__main__":
    main()
