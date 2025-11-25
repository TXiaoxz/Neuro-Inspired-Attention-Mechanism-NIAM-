#!/usr/bin/env python3
"""
Comprehensive Audio Enhancement Evaluation Script

This script evaluates audio enhancement systems using multiple objective metrics
spanning speech quality, intelligibility, spectral fidelity, and ASR performance.

Author: NIAM Research Team
Date: 2025-01-23
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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.linalg import norm

# Metric libraries
from pesq import pesq
from pystoi import stoi
import mir_eval

PROJECT_ROOT = '/home/dlwlx05/project/NIAM/Neuro-Inspired-Attention-Mechanism-NIAM--main'


# ============================================================================
# COMPREHENSIVE METRICS DOCUMENTATION
# ============================================================================

"""
================================================================================
AUDIO ENHANCEMENT EVALUATION METRICS
================================================================================

This section documents all metrics used for evaluating audio enhancement systems.
Each metric measures different aspects of speech quality, intelligibility, and
distortion. Metrics are organized by category.

--------------------------------------------------------------------------------
CATEGORY 1: SPEECH QUALITY METRICS
--------------------------------------------------------------------------------

1. PESQ (Perceptual Evaluation of Speech Quality)
   ITU-T P.862 Standard

   Range: -0.5 to 4.5 (higher is better)

   Description:
   PESQ predicts the subjective quality of speech by modeling the human auditory
   system. It computes a perceptual distance between reference and degraded signals.

   Formula:
   PESQ = f(reference, degraded)

   where f includes:
   - Level alignment
   - Time alignment
   - Auditory transform (Bark scale)
   - Cognitive modeling

   ASCII Visualization:

   Reference ──┐
               ├──> [Auditory Model] ──> [Cognitive Model] ──> PESQ Score
   Degraded ───┘

   Usage:
   Widely used in telecommunications. Higher scores indicate better perceptual
   quality. Scores above 3.0 are considered good quality.

   Limitations:
   - Sample rate must be 8kHz or 16kHz
   - Designed for narrowband/wideband speech only
   - May not work well for music or very low SNR

   Reference:
   ITU-T Recommendation P.862, "Perceptual evaluation of speech quality (PESQ)"

--------------------------------------------------------------------------------

2. STOI (Short-Time Objective Intelligibility)

   Range: 0.0 to 1.0 (higher is better)

   Description:
   STOI predicts speech intelligibility by comparing short-time spectral envelopes
   of clean and processed speech. Highly correlated with human intelligibility
   scores in noisy environments.

   Formula:
   STOI = (1/J) * sum_{j=1}^{J} d_j(x, y)

   where:
   - J = number of 1/3 octave frequency bands
   - d_j = correlation coefficient in band j
   - x, y = temporal envelopes of reference and degraded signals

   ASCII Visualization:

   Time-Frequency Analysis:

   Frequency
      ^
      |  [Band J] ████░░░░████░░░░  <- Envelope correlation d_J
      |  [Band 2] ░░██████░░██████  <- Envelope correlation d_2
      |  [Band 1] ████░░░░████░░░░  <- Envelope correlation d_1
      +--------------------------------> Time

   STOI = Average(d_1, d_2, ..., d_J)

   Usage:
   Excellent predictor of intelligibility in presence of noise, reverberation,
   and nonlinear processing. Values above 0.7 generally indicate good intelligibility.

   Limitations:
   - Requires time-aligned signals
   - May not correlate well with quality (only intelligibility)

   Reference:
   Taal et al., "A short-time objective intelligibility measure for
   time-frequency weighted noisy speech," ICASSP 2010

--------------------------------------------------------------------------------

3. SI-SNR (Scale-Invariant Signal-to-Noise Ratio)

   Range: -infinity to +infinity dB (higher is better)

   Description:
   SI-SNR measures signal fidelity while being invariant to scale (amplitude)
   differences. Widely used in speech separation and enhancement.

   Formula:
   s_target = (<s', s> / ||s||^2) * s
   e_noise = s' - s_target

   SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)

   where:
   - s = reference signal
   - s' = estimated signal
   - <,> = inner product
   - ||.|| = L2 norm

   ASCII Visualization:

   Vector Projection:

        s' (estimated)
        /|
       / |
      /  | e_noise (error)
     /   |
    /    |
   /_____|
      s_target (projection onto s)
         \
          \
           s (reference)

   SI-SNR = 10*log10(||s_target||^2 / ||e_noise||^2)

   Usage:
   Common training objective in deep learning models. Scale-invariance makes it
   robust to gain mismatches. Values above 15 dB are considered good.

   Advantages:
   - Scale invariant (no need for gain calibration)
   - Differentiable (can be used as loss function)
   - Correlates well with speech quality

   Reference:
   Le Roux et al., "SDR - Half-baked or Well Done?," ICASSP 2019

--------------------------------------------------------------------------------

4. SDR (Signal-to-Distortion Ratio)

   Range: -infinity to +infinity dB (higher is better)

   Description:
   SDR measures overall signal quality by decomposing error into target, interference,
   and artifacts components using optimal projection.

   Formula:
   s_target = P_s(s')
   e_interf = P_interf(s' - s_target)
   e_artif = s' - s_target - e_interf

   SDR = 10 * log10(||s_target||^2 / ||e_interf + e_artif||^2)

   where P_s is projection onto subspace spanned by reference s

   ASCII Visualization:

   Signal Decomposition:

   s' (estimated) = s_target + e_interf + e_artif
                    ^          ^          ^
                    |          |          |
                    Target   Interference  Artifacts

   SDR measures: Target vs (Interference + Artifacts)

   Usage:
   Standard metric in source separation tasks (BSS Eval toolkit). Used alongside
   SIR and SAR for comprehensive evaluation.

   Reference:
   Vincent et al., "Performance measurement in blind audio source separation,"
   IEEE TASLP 2006

--------------------------------------------------------------------------------

5. SIR (Signal-to-Interference Ratio)

   Range: -infinity to +infinity dB (higher is better)

   Description:
   SIR specifically measures interference from other sources (e.g., competing speakers
   in cocktail party problem).

   Formula:
   SIR = 10 * log10(||s_target||^2 / ||e_interf||^2)

   Usage:
   Measures how well the system suppresses interfering sources while preserving
   target speech. High SIR indicates good source separation.

--------------------------------------------------------------------------------

6. SAR (Signal-to-Artifacts Ratio)

   Range: -infinity to +infinity dB (higher is better)

   Description:
   SAR measures artifacts introduced by the processing system (e.g., musical noise,
   spectral distortion from aggressive noise suppression).

   Formula:
   SAR = 10 * log10(||(s_target + e_interf)||^2 / ||e_artif||^2)

   Usage:
   Quantifies processing artifacts. Low SAR indicates the algorithm introduces
   audible distortions like musical noise or reverb artifacts.

--------------------------------------------------------------------------------
CATEGORY 2: SPECTRAL METRICS
--------------------------------------------------------------------------------

7. LSD (Log-Spectral Distance)

   Range: 0 to infinity (lower is better)

   Description:
   LSD measures spectral distortion by computing the L2 distance between log-magnitude
   spectra. Captures frequency-domain differences.

   Formula:
   LSD = sqrt( (1/K) * sum_{k=1}^{K} [log(|X(k)|) - log(|Y(k)|)]^2 )

   where:
   - X(k), Y(k) = DFT coefficients of reference and degraded
   - K = number of frequency bins

   ASCII Visualization:

   Log Magnitude Spectrum:

   Mag (dB)
      ^
      |     Reference: ___/\___/\___
      |    /\        /            \
      |   /  \      /              \
      |  /    \___ /                \___
      |
      | Degraded:  ___/\/\___/\/\___
      +--------------------------------> Frequency

   LSD = sqrt(mean((log(Ref) - log(Deg))^2))

   Usage:
   Common in speech coding and TTS. Values below 1.0 indicate high fidelity.

   Reference:
   Gray & Markel, "Distance measures for speech processing," IEEE TASSP 1976

--------------------------------------------------------------------------------

8. Spectral Convergence

   Range: 0 to infinity (lower is better)

   Description:
   Measures normalized spectral distance. Similar to LSD but uses linear magnitude.

   Formula:
   SC = ||X - Y||_F / ||X||_F

   where ||.||_F is Frobenius norm

   Usage:
   Used in neural vocoders and audio synthesis. Lower values indicate better
   spectral reconstruction.

--------------------------------------------------------------------------------
CATEGORY 3: ASR-BASED METRICS
--------------------------------------------------------------------------------

9. WER (Word Error Rate)

   Range: 0.0 to infinity (lower is better, 0 = perfect)

   Description:
   WER measures word-level transcription errors using edit distance (substitutions,
   insertions, deletions).

   Formula:
   WER = (S + D + I) / N

   where:
   - S = number of substitutions
   - D = number of deletions
   - I = number of insertions
   - N = number of words in reference

   ASCII Visualization:

   Reference:  the  cat  sat  on  the  mat
   Hypothesis: the  dog  sat  on  a    ___

   Alignment:  ===  SUB  ===  === SUB  DEL

   WER = (2 substitutions + 1 deletion) / 6 = 0.50 = 50%

   Usage:
   Primary metric for ASR systems. Also used to measure downstream task performance
   for speech enhancement. Lower WER indicates better speech quality/intelligibility.

--------------------------------------------------------------------------------

10. CER (Character Error Rate)

    Range: 0.0 to infinity (lower is better, 0 = perfect)

    Description:
    Similar to WER but computed at character level. More fine-grained than WER.

    Formula:
    CER = (S_c + D_c + I_c) / N_c

    where subscript c denotes character-level operations

    Usage:
    Useful for languages without clear word boundaries or when partial word
    preservation matters.

--------------------------------------------------------------------------------
CATEGORY 4: PERFORMANCE METRICS
--------------------------------------------------------------------------------

11. RTF (Real-Time Factor)

    Range: 0 to infinity (lower is better, <1.0 means real-time capable)

    Description:
    RTF measures computational efficiency as the ratio of processing time to
    audio duration.

    Formula:
    RTF = Processing_Time / Audio_Duration

    ASCII Visualization:

    Timeline:

    Audio:     |====== 10 seconds ======|

    Processing:|== 5 sec ==|

    RTF = 5 / 10 = 0.5 (can process 2x faster than real-time)

    Usage:
    Critical for real-time applications (telephony, hearing aids, live streaming).
    RTF < 1.0 required for real-time processing.

    Categories:
    - RTF < 0.1: Very efficient (e.g., simple filters)
    - RTF < 1.0: Real-time capable
    - RTF > 1.0: Cannot process in real-time (offline only)

--------------------------------------------------------------------------------
METRIC CORRELATION AND INTERPRETATION
--------------------------------------------------------------------------------

Typical Metric Ranges for Speech Enhancement:

Quality Level  | PESQ   | STOI  | SI-SNR | WER    | LSD   | RTF
---------------|--------|-------|--------|--------|-------|-------
Excellent      | >4.0   | >0.95 | >20 dB | <10%   | <0.5  | <0.1
Good           | 3.5-4.0| 0.85  | 15-20  | 10-20% | 0.5-1 | <0.5
Fair           | 3.0-3.5| 0.75  | 10-15  | 20-40% | 1-2   | <1.0
Poor           | 2.5-3.0| 0.65  | 5-10   | 40-60% | 2-3   | >1.0
Bad            | <2.5   | <0.65 | <5     | >60%   | >3    | N/A

Metric Trade-offs:
- High PESQ/STOI but high WER: Good perceptual quality but poor content preservation
- High SI-SNR but low PESQ: Low distortion but may sound unnatural
- High SAR but low SIR: Few artifacts but poor interference suppression
- Low RTF but low quality: Fast but ineffective processing

Recommended Metric Combinations:
1. PESQ + STOI: Overall quality and intelligibility
2. SI-SNR + WER: Signal fidelity and content preservation
3. SDR/SIR/SAR: Decomposed error analysis
4. LSD + PESQ: Spectral accuracy and perceptual quality
5. All metrics + RTF: Comprehensive quality and efficiency

--------------------------------------------------------------------------------
VISUAL METRIC SUMMARY
--------------------------------------------------------------------------------

Metric Categories Radar Plot:

        Quality (PESQ, STOI)
                  ^
                  |
                  |
    Spectral <----+----> ASR-based
    (LSD, SC)     |      (WER, CER)
                  |
                  |
                  v
        Fidelity (SI-SNR, SDR)

Each axis represents normalized metric scores (0-1 scale after normalization).

--------------------------------------------------------------------------------
EXTERNAL REFERENCES
--------------------------------------------------------------------------------

For detailed visualizations, see:
- docs/metrics/metric_comparison_radar.png: Radar plot of all metrics
- docs/metrics/metric_correlation_heatmap.png: Correlation between metrics
- docs/metrics/score_distributions.png: Distribution of scores across samples

For implementation details, see:
- https://github.com/ludlows/python-pesq (PESQ)
- https://github.com/mpariente/pystoi (STOI)
- https://github.com/sigsep/bsseval (SDR/SIR/SAR)

================================================================================
"""


# ============================================================================
# METRIC COMPUTATION FUNCTIONS
# ============================================================================

def compute_pesq(reference, degraded, sr=16000):
    """
    Compute PESQ (Perceptual Evaluation of Speech Quality).

    Args:
        reference: Clean reference signal (numpy array)
        degraded: Degraded/enhanced signal (numpy array)
        sr: Sample rate (8000 or 16000)

    Returns:
        PESQ score (float): Range [-0.5, 4.5], higher is better
    """
    try:
        # PESQ requires specific sample rates
        if sr not in [8000, 16000]:
            raise ValueError(f"PESQ only supports 8kHz or 16kHz, got {sr}Hz")

        mode = 'wb' if sr == 16000 else 'nb'  # wideband or narrowband
        score = pesq(sr, reference, degraded, mode)
        return score
    except Exception as e:
        print(f"PESQ computation failed: {e}")
        return None


def compute_stoi(reference, degraded, sr=16000):
    """
    Compute STOI (Short-Time Objective Intelligibility).

    Args:
        reference: Clean reference signal
        degraded: Degraded/enhanced signal
        sr: Sample rate

    Returns:
        STOI score (float): Range [0.0, 1.0], higher is better
    """
    try:
        score = stoi(reference, degraded, sr, extended=False)
        return score
    except Exception as e:
        print(f"STOI computation failed: {e}")
        return None


def compute_si_snr(reference, estimated):
    """
    Compute SI-SNR (Scale-Invariant Signal-to-Noise Ratio).

    Args:
        reference: Reference signal
        estimated: Estimated signal

    Returns:
        SI-SNR in dB (float): Higher is better
    """
    try:
        # Zero-mean normalization
        reference = reference - np.mean(reference)
        estimated = estimated - np.mean(estimated)

        # Compute projection
        alpha = np.dot(estimated, reference) / (np.dot(reference, reference) + 1e-8)
        s_target = alpha * reference
        e_noise = estimated - s_target

        # Compute SI-SNR
        si_snr = 10 * np.log10(np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-8))
        return si_snr
    except Exception as e:
        print(f"SI-SNR computation failed: {e}")
        return None


def compute_sdr_sir_sar(reference, estimated):
    """
    Compute SDR, SIR, SAR using mir_eval.

    Args:
        reference: Reference signal (1D or 2D array)
        estimated: Estimated signal (same shape as reference)

    Returns:
        Dictionary with 'sdr', 'sir', 'sar' keys
    """
    try:
        # Ensure 2D shape (n_sources, n_samples)
        if reference.ndim == 1:
            reference = reference.reshape(1, -1)
        if estimated.ndim == 1:
            estimated = estimated.reshape(1, -1)

        # Compute BSS metrics
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
            reference, estimated, compute_permutation=False
        )

        return {
            'sdr': sdr[0],  # Take first source
            'sir': sir[0],
            'sar': sar[0]
        }
    except Exception as e:
        print(f"SDR/SIR/SAR computation failed: {e}")
        return {'sdr': None, 'sir': None, 'sar': None}


def compute_lsd(reference, degraded, n_fft=512, hop_length=256):
    """
    Compute LSD (Log-Spectral Distance).

    Args:
        reference: Reference signal
        degraded: Degraded signal
        n_fft: FFT size
        hop_length: Hop length

    Returns:
        LSD score (float): Lower is better
    """
    try:
        # Compute STFT
        _, _, ref_stft = signal.stft(reference, nperseg=n_fft, noverlap=n_fft-hop_length)
        _, _, deg_stft = signal.stft(degraded, nperseg=n_fft, noverlap=n_fft-hop_length)

        # Compute log magnitude
        ref_mag = np.abs(ref_stft)
        deg_mag = np.abs(deg_stft)

        # Avoid log(0)
        ref_log = np.log(ref_mag + 1e-8)
        deg_log = np.log(deg_mag + 1e-8)

        # Compute LSD
        lsd = np.sqrt(np.mean((ref_log - deg_log) ** 2))
        return lsd
    except Exception as e:
        print(f"LSD computation failed: {e}")
        return None


def compute_spectral_convergence(reference, degraded, n_fft=512, hop_length=256):
    """
    Compute Spectral Convergence.

    Args:
        reference: Reference signal
        degraded: Degraded signal
        n_fft: FFT size
        hop_length: Hop length

    Returns:
        Spectral Convergence (float): Lower is better
    """
    try:
        # Compute STFT
        _, _, ref_stft = signal.stft(reference, nperseg=n_fft, noverlap=n_fft-hop_length)
        _, _, deg_stft = signal.stft(degraded, nperseg=n_fft, noverlap=n_fft-hop_length)

        # Compute Frobenius norm
        diff_norm = norm(ref_stft - deg_stft, 'fro')
        ref_norm = norm(ref_stft, 'fro')

        sc = diff_norm / (ref_norm + 1e-8)
        return sc
    except Exception as e:
        print(f"Spectral Convergence computation failed: {e}")
        return None


def compute_snr(reference, degraded):
    """
    Compute traditional SNR.

    Args:
        reference: Reference signal
        degraded: Degraded signal

    Returns:
        SNR in dB (float)
    """
    try:
        noise = degraded - reference
        signal_power = np.mean(reference ** 2)
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
        return snr
    except Exception as e:
        print(f"SNR computation failed: {e}")
        return None


# ============================================================================
# COMPREHENSIVE EVALUATION FUNCTION
# ============================================================================

def evaluate_comprehensive(test_samples_dir, whisper_model_size='base', output_dir=None):
    """
    Comprehensive audio enhancement evaluation with all metrics.

    Args:
        test_samples_dir: Directory containing test samples
        whisper_model_size: Whisper model size for ASR evaluation
        output_dir: Output directory for results

    Returns:
        DataFrame with all evaluation results
    """
    print("="*80)
    print("COMPREHENSIVE AUDIO ENHANCEMENT EVALUATION")
    print("="*80)

    # Check directory
    test_samples_dir = Path(test_samples_dir)
    if not test_samples_dir.exists():
        print(f"Error: Directory not found: {test_samples_dir}")
        return None

    # Create output directory
    if output_dir is None:
        output_dir = test_samples_dir.parent / "comprehensive_results"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load Whisper model for ASR metrics
    print(f"\nLoading Whisper '{whisper_model_size}' model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    whisper_model = whisper.load_model(whisper_model_size, device=device)
    print(f"Whisper model loaded on {device}")

    # Find test samples
    metadata_files = sorted(test_samples_dir.glob("sample_*_metadata.txt"))
    print(f"\nFound {len(metadata_files)} test samples")

    # Evaluate each sample
    results = []

    print("\nEvaluating samples with all metrics...")
    for metadata_file in tqdm(metadata_files, desc="Processing"):
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
            continue

        # Load audio files
        clean_path = test_samples_dir / f"{sample_id}_clean.wav"
        noisy_path = test_samples_dir / f"{sample_id}_noisy.wav"
        enhanced_path = test_samples_dir / f"{sample_id}_enhanced.wav"

        if not all([clean_path.exists(), noisy_path.exists(), enhanced_path.exists()]):
            continue

        clean, sr = sf.read(str(clean_path))
        noisy, _ = sf.read(str(noisy_path))
        enhanced, _ = sf.read(str(enhanced_path))

        # Ensure same length
        min_len = min(len(clean), len(noisy), len(enhanced))
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        enhanced = enhanced[:min_len]

        # ===== Compute all metrics =====
        start_time = time.time()

        # Speech Quality Metrics
        pesq_score = compute_pesq(clean, enhanced, sr)
        stoi_score = compute_stoi(clean, enhanced, sr)
        si_snr_score = compute_si_snr(clean, enhanced)
        sdr_sir_sar = compute_sdr_sir_sar(clean, enhanced)
        snr_score = compute_snr(clean, enhanced)

        # Spectral Metrics
        lsd_score = compute_lsd(clean, enhanced)
        sc_score = compute_spectral_convergence(clean, enhanced)

        # Baseline comparisons (noisy vs enhanced)
        si_snr_noisy = compute_si_snr(clean, noisy)
        si_snr_improvement = si_snr_score - si_snr_noisy if si_snr_score and si_snr_noisy else None

        # ASR Metrics (using Whisper)
        try:
            clean_result = whisper_model.transcribe(str(clean_path), language="en")
            noisy_result = whisper_model.transcribe(str(noisy_path), language="en")
            enhanced_result = whisper_model.transcribe(str(enhanced_path), language="en")

            clean_text = clean_result["text"].strip()
            noisy_text = noisy_result["text"].strip()
            enhanced_text = enhanced_result["text"].strip()

            wer_clean = wer(ground_truth_text, clean_text)
            wer_noisy = wer(ground_truth_text, noisy_text)
            wer_enhanced = wer(ground_truth_text, enhanced_text)
            wer_improvement = wer_noisy - wer_enhanced

            cer_clean = cer(ground_truth_text, clean_text)
            cer_noisy = cer(ground_truth_text, noisy_text)
            cer_enhanced = cer(ground_truth_text, enhanced_text)
            cer_improvement = cer_noisy - cer_enhanced
        except Exception as e:
            print(f"ASR evaluation failed for {sample_id}: {e}")
            wer_clean = wer_noisy = wer_enhanced = wer_improvement = None
            cer_clean = cer_noisy = cer_enhanced = cer_improvement = None
            clean_text = noisy_text = enhanced_text = ""

        # Performance Metrics
        processing_time = time.time() - start_time
        audio_duration = len(enhanced) / sr
        rtf = processing_time / audio_duration

        # Store results
        results.append({
            'sample_id': sample_id,
            'ground_truth': ground_truth_text,

            # Speech Quality
            'pesq': pesq_score,
            'stoi': stoi_score,
            'si_snr': si_snr_score,
            'si_snr_improvement': si_snr_improvement,
            'sdr': sdr_sir_sar['sdr'],
            'sir': sdr_sir_sar['sir'],
            'sar': sdr_sir_sar['sar'],
            'snr': snr_score,

            # Spectral
            'lsd': lsd_score,
            'spectral_convergence': sc_score,

            # ASR
            'wer_clean': wer_clean,
            'wer_noisy': wer_noisy,
            'wer_enhanced': wer_enhanced,
            'wer_improvement': wer_improvement,
            'cer_clean': cer_clean,
            'cer_noisy': cer_noisy,
            'cer_enhanced': cer_enhanced,
            'cer_improvement': cer_improvement,

            # Performance
            'rtf': rtf,
            'processing_time': processing_time,
            'audio_duration': audio_duration,

            # Transcriptions
            'clean_transcription': clean_text,
            'noisy_transcription': noisy_text,
            'enhanced_transcription': enhanced_text
        })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save detailed results
    csv_path = output_dir / "comprehensive_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")

    # Print summary
    print_summary(df)

    # Create visualizations
    create_visualizations(df, output_dir)

    return df


def print_summary(df):
    """Print summary statistics for all metrics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\nSamples evaluated: {len(df)}")

    # Speech Quality Metrics
    print("\n--- Speech Quality Metrics ---")
    if df['pesq'].notna().any():
        print(f"PESQ:          {df['pesq'].mean():.3f} ± {df['pesq'].std():.3f}  (range: -0.5 to 4.5, higher better)")
    if df['stoi'].notna().any():
        print(f"STOI:          {df['stoi'].mean():.3f} ± {df['stoi'].std():.3f}  (range: 0 to 1, higher better)")
    if df['si_snr'].notna().any():
        print(f"SI-SNR:        {df['si_snr'].mean():.2f} ± {df['si_snr'].std():.2f} dB  (higher better)")
        print(f"SI-SNR Improv: {df['si_snr_improvement'].mean():.2f} ± {df['si_snr_improvement'].std():.2f} dB")
    if df['sdr'].notna().any():
        print(f"SDR:           {df['sdr'].mean():.2f} ± {df['sdr'].std():.2f} dB  (higher better)")
    if df['sir'].notna().any():
        print(f"SIR:           {df['sir'].mean():.2f} ± {df['sir'].std():.2f} dB  (higher better)")
    if df['sar'].notna().any():
        print(f"SAR:           {df['sar'].mean():.2f} ± {df['sar'].std():.2f} dB  (higher better)")

    # Spectral Metrics
    print("\n--- Spectral Metrics ---")
    if df['lsd'].notna().any():
        print(f"LSD:           {df['lsd'].mean():.3f} ± {df['lsd'].std():.3f}  (lower better)")
    if df['spectral_convergence'].notna().any():
        print(f"Spectral Conv: {df['spectral_convergence'].mean():.3f} ± {df['spectral_convergence'].std():.3f}  (lower better)")

    # ASR Metrics
    print("\n--- ASR Metrics ---")
    if df['wer_enhanced'].notna().any():
        print(f"WER Enhanced:  {df['wer_enhanced'].mean():.1%} ± {df['wer_enhanced'].std():.1%}  (lower better)")
        print(f"WER Improv:    {df['wer_improvement'].mean():.1%} ± {df['wer_improvement'].std():.1%}")
    if df['cer_enhanced'].notna().any():
        print(f"CER Enhanced:  {df['cer_enhanced'].mean():.1%} ± {df['cer_enhanced'].std():.1%}  (lower better)")
        print(f"CER Improv:    {df['cer_improvement'].mean():.1%} ± {df['cer_improvement'].std():.1%}")

    # Performance
    print("\n--- Performance Metrics ---")
    if df['rtf'].notna().any():
        print(f"RTF:           {df['rtf'].mean():.3f} ± {df['rtf'].std():.3f}  (<1.0 = real-time capable)")
    if df['processing_time'].notna().any():
        print(f"Proc Time:     {df['processing_time'].mean():.3f} ± {df['processing_time'].std():.3f} seconds")

    print("\n" + "="*80)


def create_visualizations(df, output_dir):
    """Create visualization plots for metrics."""
    print("\nCreating visualizations...")

    # Create docs/metrics directory
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True, parents=True)

    # 1. Radar plot of normalized metrics
    create_radar_plot(df, vis_dir)

    # 2. Metric correlation heatmap
    create_correlation_heatmap(df, vis_dir)

    # 3. Distribution plots
    create_distribution_plots(df, vis_dir)

    print(f"Visualizations saved to: {vis_dir}")


def create_radar_plot(df, output_dir):
    """Create radar plot comparing average metrics."""
    # Select key metrics
    metrics = {
        'PESQ': 'pesq',
        'STOI': 'stoi',
        'SI-SNR': 'si_snr',
        'SDR': 'sdr',
        '1-WER': 'wer_enhanced',  # Invert so higher is better
        '1-CER': 'cer_enhanced',
        '1-LSD': 'lsd',
        'SAR': 'sar'
    }

    # Compute normalized scores (0-1 scale)
    scores = []
    labels = []

    for label, col in metrics.items():
        if col in df.columns and df[col].notna().any():
            values = df[col].dropna()

            # Normalize to 0-1 (higher is better)
            if 'WER' in label or 'CER' in label or 'LSD' in label:
                # Invert: lower is better -> higher normalized score is better
                normalized = 1 - (values.mean() / (values.max() + 1e-8))
            elif col == 'pesq':
                normalized = values.mean() / 4.5  # PESQ max is 4.5
            elif col in ['stoi', 'sar', 'sdr', 'sir']:
                # Already in reasonable range, normalize by max
                normalized = values.mean() / max(values.max(), 1.0)
            elif col == 'si_snr':
                # SI-SNR: normalize to 0-1 assuming 20dB is excellent
                normalized = min(values.mean() / 20.0, 1.0)
            else:
                normalized = values.mean()

            scores.append(max(0, min(1, normalized)))
            labels.append(label)

    if len(scores) == 0:
        print("Not enough metrics for radar plot")
        return

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    scores_plot = scores + [scores[0]]  # Close the plot
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles_plot, scores_plot, 'o-', linewidth=2, label='Enhanced Audio')
    ax.fill(angles_plot, scores_plot, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)
    ax.set_title('Comprehensive Metric Evaluation (Normalized)', size=16, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(output_dir / 'metric_radar_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Created radar plot: metric_radar_plot.png")


def create_correlation_heatmap(df, output_dir):
    """Create correlation heatmap between metrics."""
    # Select numeric columns
    numeric_cols = [
        'pesq', 'stoi', 'si_snr', 'sdr', 'sir', 'sar',
        'lsd', 'spectral_convergence',
        'wer_enhanced', 'cer_enhanced'
    ]

    available_cols = [c for c in numeric_cols if c in df.columns and df[c].notna().any()]

    if len(available_cols) < 2:
        print("Not enough metrics for correlation heatmap")
        return

    # Compute correlation matrix
    corr = df[available_cols].corr()

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Metric Correlation Matrix', size=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Created correlation heatmap: metric_correlation_heatmap.png")


def create_distribution_plots(df, output_dir):
    """Create distribution plots for key metrics."""
    key_metrics = {
        'PESQ': 'pesq',
        'STOI': 'stoi',
        'SI-SNR (dB)': 'si_snr',
        'WER': 'wer_enhanced',
        'LSD': 'lsd'
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (label, col) in enumerate(key_metrics.items()):
        if col in df.columns and df[col].notna().any():
            ax = axes[idx]
            values = df[col].dropna()
            ax.hist(values, bins=15, edgecolor='black', alpha=0.7)
            ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {values.mean():.3f}')
            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'{label} Distribution', fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)

    # Remove extra subplots
    for idx in range(len(key_metrics), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_dir / 'metric_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Created distribution plots: metric_distributions.png")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Audio Enhancement Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_comprehensive.py -d results/test_samples
  python evaluate_comprehensive.py -d results/niam_v2/test_samples -m base -o results/niam_v2/eval
        """
    )

    parser.add_argument('-d', '--test_dir', type=str, required=True,
                        help='Directory containing test samples (clean/noisy/enhanced triplets)')
    parser.add_argument('-m', '--whisper_model', type=str, default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size for ASR evaluation')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Output directory for results (default: <test_dir>/../comprehensive_results)')

    args = parser.parse_args()

    evaluate_comprehensive(args.test_dir, args.whisper_model, args.output_dir)


if __name__ == "__main__":
    main()
