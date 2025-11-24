#!/usr/bin/env python3
"""
Compare Baseline vs NIAM v2 Results

Analyzes and compares Whisper evaluation results between:
- Baseline Transformer (STFT only)
- NIAM v2 (Transformer + NIAM modules)
"""

import pandas as pd
import numpy as np

# Paths
BASELINE_CSV = '/home/dlwlx05/project/NIAM/Neuro-Inspired-Attention-Mechanism-NIAM--main/results/test_samples/whisper_evaluation_results.csv'
NIAM_V2_CSV = '/home/dlwlx05/project/NIAM/results/niam_v2/whisper_evaluation.csv'

def load_and_analyze(csv_path, model_name):
    """Load CSV and compute statistics"""
    df = pd.read_csv(csv_path)

    stats = {
        'model': model_name,
        'num_samples': len(df),
        'wer_clean_mean': df['wer_clean'].mean(),
        'wer_clean_std': df['wer_clean'].std(),
        'wer_noisy_mean': df['wer_noisy'].mean(),
        'wer_noisy_std': df['wer_noisy'].std(),
        'wer_enhanced_mean': df['wer_enhanced'].mean(),
        'wer_enhanced_std': df['wer_enhanced'].std(),
        'wer_improvement_mean': df['wer_improvement'].mean(),
        'wer_improvement_std': df['wer_improvement'].std(),
        'cer_clean_mean': df['cer_clean'].mean(),
        'cer_clean_std': df['cer_clean'].std(),
        'cer_noisy_mean': df['cer_noisy'].mean(),
        'cer_noisy_std': df['cer_noisy'].std(),
        'cer_enhanced_mean': df['cer_enhanced'].mean(),
        'cer_enhanced_std': df['cer_enhanced'].std(),
        'cer_improvement_mean': df['cer_improvement'].mean(),
        'cer_improvement_std': df['cer_improvement'].std(),
        'samples_improved': (df['wer_improvement'] > 0).sum(),
        'samples_degraded': (df['wer_improvement'] < 0).sum(),
        'samples_same': (df['wer_improvement'] == 0).sum(),
    }

    return df, stats

def compare_models():
    """Compare baseline and NIAM v2"""

    print("="*80)
    print("BASELINE vs NIAM v2 COMPARISON")
    print("="*80)

    # Load both results
    print("\nLoading results...")
    baseline_df, baseline_stats = load_and_analyze(BASELINE_CSV, "Baseline Transformer")
    niam_df, niam_stats = load_and_analyze(NIAM_V2_CSV, "NIAM v2")

    print(f"✓ Baseline: {baseline_stats['num_samples']} samples")
    print(f"✓ NIAM v2: {niam_stats['num_samples']} samples")

    # WER Comparison
    print("\n" + "="*80)
    print("WORD ERROR RATE (WER) COMPARISON")
    print("="*80)

    print(f"\n{'Metric':<30} {'Baseline':<20} {'NIAM v2':<20} {'Difference':<15}")
    print("-"*80)

    # Clean audio
    print(f"{'Clean Audio WER':<30} {baseline_stats['wer_clean_mean']:.1%} ± {baseline_stats['wer_clean_std']:.1%}   {niam_stats['wer_clean_mean']:.1%} ± {niam_stats['wer_clean_std']:.1%}   {(niam_stats['wer_clean_mean'] - baseline_stats['wer_clean_mean']):.1%}")

    # Noisy audio
    print(f"{'Noisy Audio WER':<30} {baseline_stats['wer_noisy_mean']:.1%} ± {baseline_stats['wer_noisy_std']:.1%}   {niam_stats['wer_noisy_mean']:.1%} ± {niam_stats['wer_noisy_std']:.1%}   {(niam_stats['wer_noisy_mean'] - baseline_stats['wer_noisy_mean']):.1%}")

    # Enhanced audio
    print(f"{'Enhanced Audio WER':<30} {baseline_stats['wer_enhanced_mean']:.1%} ± {baseline_stats['wer_enhanced_std']:.1%}   {niam_stats['wer_enhanced_mean']:.1%} ± {niam_stats['wer_enhanced_std']:.1%}   {(niam_stats['wer_enhanced_mean'] - baseline_stats['wer_enhanced_mean']):.1%}")

    # WER Improvement
    print(f"{'WER Improvement':<30} {baseline_stats['wer_improvement_mean']:.1%} ± {baseline_stats['wer_improvement_std']:.1%}   {niam_stats['wer_improvement_mean']:.1%} ± {niam_stats['wer_improvement_std']:.1%}   {(niam_stats['wer_improvement_mean'] - baseline_stats['wer_improvement_mean']):.1%}")

    # Relative improvement
    baseline_rel_imp = (baseline_stats['wer_noisy_mean'] - baseline_stats['wer_enhanced_mean']) / baseline_stats['wer_noisy_mean']
    niam_rel_imp = (niam_stats['wer_noisy_mean'] - niam_stats['wer_enhanced_mean']) / niam_stats['wer_noisy_mean']

    print(f"\n{'Relative WER Improvement':<30} {baseline_rel_imp:.1%}             {niam_rel_imp:.1%}             {(niam_rel_imp - baseline_rel_imp):.1%}")

    # CER Comparison
    print("\n" + "="*80)
    print("CHARACTER ERROR RATE (CER) COMPARISON")
    print("="*80)

    print(f"\n{'Metric':<30} {'Baseline':<20} {'NIAM v2':<20} {'Difference':<15}")
    print("-"*80)

    # Clean audio
    print(f"{'Clean Audio CER':<30} {baseline_stats['cer_clean_mean']:.1%} ± {baseline_stats['cer_clean_std']:.1%}   {niam_stats['cer_clean_mean']:.1%} ± {niam_stats['cer_clean_std']:.1%}   {(niam_stats['cer_clean_mean'] - baseline_stats['cer_clean_mean']):.1%}")

    # Noisy audio
    print(f"{'Noisy Audio CER':<30} {baseline_stats['cer_noisy_mean']:.1%} ± {baseline_stats['cer_noisy_std']:.1%}   {niam_stats['cer_noisy_mean']:.1%} ± {niam_stats['cer_noisy_std']:.1%}   {(niam_stats['cer_noisy_mean'] - baseline_stats['cer_noisy_mean']):.1%}")

    # Enhanced audio
    print(f"{'Enhanced Audio CER':<30} {baseline_stats['cer_enhanced_mean']:.1%} ± {baseline_stats['cer_enhanced_std']:.1%}   {niam_stats['cer_enhanced_mean']:.1%} ± {niam_stats['cer_enhanced_std']:.1%}   {(niam_stats['cer_enhanced_mean'] - baseline_stats['cer_enhanced_mean']):.1%}")

    # CER Improvement
    print(f"{'CER Improvement':<30} {baseline_stats['cer_improvement_mean']:.1%} ± {baseline_stats['cer_improvement_std']:.1%}   {niam_stats['cer_improvement_mean']:.1%} ± {niam_stats['cer_improvement_std']:.1%}   {(niam_stats['cer_improvement_mean'] - baseline_stats['cer_improvement_mean']):.1%}")

    # Sample-wise analysis
    print("\n" + "="*80)
    print("SAMPLE-WISE WER ANALYSIS")
    print("="*80)

    print(f"\n{'Metric':<30} {'Baseline':<20} {'NIAM v2':<20}")
    print("-"*80)
    print(f"{'Improved samples':<30} {baseline_stats['samples_improved']}/{baseline_stats['num_samples']} ({baseline_stats['samples_improved']/baseline_stats['num_samples']*100:.1f}%)       {niam_stats['samples_improved']}/{niam_stats['num_samples']} ({niam_stats['samples_improved']/niam_stats['num_samples']*100:.1f}%)")
    print(f"{'Degraded samples':<30} {baseline_stats['samples_degraded']}/{baseline_stats['num_samples']} ({baseline_stats['samples_degraded']/baseline_stats['num_samples']*100:.1f}%)       {niam_stats['samples_degraded']}/{niam_stats['num_samples']} ({niam_stats['samples_degraded']/niam_stats['num_samples']*100:.1f}%)")
    print(f"{'Same samples':<30} {baseline_stats['samples_same']}/{baseline_stats['num_samples']} ({baseline_stats['samples_same']/baseline_stats['num_samples']*100:.1f}%)       {niam_stats['samples_same']}/{niam_stats['num_samples']} ({niam_stats['samples_same']/niam_stats['num_samples']*100:.1f}%)")

    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    wer_diff = niam_stats['wer_enhanced_mean'] - baseline_stats['wer_enhanced_mean']
    cer_diff = niam_stats['cer_enhanced_mean'] - baseline_stats['cer_enhanced_mean']

    print(f"\n1. Enhanced Audio Quality:")
    if wer_diff < 0:
        print(f"   ✓ NIAM v2 achieved {abs(wer_diff):.1%} LOWER WER than baseline")
    elif wer_diff > 0:
        print(f"   ✗ NIAM v2 has {wer_diff:.1%} HIGHER WER than baseline")
    else:
        print(f"   - NIAM v2 and baseline have the same WER")

    if cer_diff < 0:
        print(f"   ✓ NIAM v2 achieved {abs(cer_diff):.1%} LOWER CER than baseline")
    elif cer_diff > 0:
        print(f"   ✗ NIAM v2 has {cer_diff:.1%} HIGHER CER than baseline")
    else:
        print(f"   - NIAM v2 and baseline have the same CER")

    imp_diff = niam_stats['wer_improvement_mean'] - baseline_stats['wer_improvement_mean']
    print(f"\n2. WER Improvement over Noisy:")
    if imp_diff > 0:
        print(f"   ✓ NIAM v2 improved {abs(imp_diff):.1%} MORE than baseline")
    elif imp_diff < 0:
        print(f"   ✗ NIAM v2 improved {abs(imp_diff):.1%} LESS than baseline")
    else:
        print(f"   - NIAM v2 and baseline have the same improvement")

    print(f"\n3. Sample-wise Performance:")
    improved_diff = niam_stats['samples_improved'] - baseline_stats['samples_improved']
    if improved_diff > 0:
        print(f"   ✓ NIAM v2 improved {improved_diff} MORE samples than baseline")
    elif improved_diff < 0:
        print(f"   ✗ NIAM v2 improved {abs(improved_diff)} FEWER samples than baseline")
    else:
        print(f"   - NIAM v2 and baseline improved the same number of samples")

    print("\n" + "="*80)

    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    print("\nPossible reasons for current results:")
    print("1. Model capacity: NIAM adds 4 attention modules with learnable weights")
    print("2. Training data: Only 270 samples may be insufficient for NIAM to learn")
    print("3. Hyperparameters: Learning rate, dropout, or residual weights may need tuning")
    print("4. Loss function: Current SI-SNR loss may not align with WER/CER metrics")
    print("5. Module weights: Learned weights show Module 3 (temporal) contributes least")

    print("\nSuggested improvements:")
    print("- Train on more data (full People's Speech dataset)")
    print("- Tune NIAM module hyperparameters (dropout, attention heads)")
    print("- Try perceptual loss or direct WER-based loss")
    print("- Adjust residual weight initialization")
    print("- Analyze per-module contributions via ablation study")

    print("\n" + "="*80)


if __name__ == "__main__":
    compare_models()
