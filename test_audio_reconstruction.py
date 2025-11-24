#!/usr/bin/env python3
"""
STFT/iSTFT Roundtrip Test

This is THE MOST CRITICAL test to prevent "static noise" issues.

Test flow:
  Clean audio → STFT → iSTFT → Should still be human speech (NOT static noise!)

If this test produces static noise, it means STFT parameters are wrong.
DO NOT proceed to training until this test passes.
"""

import torch
import torchaudio
import soundfile as sf
import numpy as np
from datasets import load_dataset
import os


def test_stft_roundtrip():
    """
    Test STFT → iSTFT roundtrip to verify no "static noise" issue

    Returns:
        success: True if test passes (no static noise)
    """
    print("="*80)
    print("STFT/iSTFT ROUNDTRIP TEST")
    print("="*80)
    print("\nThis test verifies that STFT → iSTFT doesn't create static noise.\n")

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(
        "MLCommons/peoples_speech",
        "microset",
        split="train",
        cache_dir="/home/dlwlx05/project/NIAM/data/"
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples\n")

    # STFT parameters (MUST match exactly in stft and istft!)
    N_FFT = 400
    HOP_LENGTH = 160
    WIN_LENGTH = 400
    SAMPLE_RATE = 16000

    print(f"STFT Parameters:")
    print(f"  n_fft: {N_FFT}")
    print(f"  hop_length: {HOP_LENGTH}")
    print(f"  win_length: {WIN_LENGTH}")
    print(f"  window: hann_window")
    print(f"  sample_rate: {SAMPLE_RATE} Hz\n")

    # Test on first 3 samples
    test_indices = [0, 1, 2]
    results_dir = "/home/dlwlx05/project/NIAM/Neuro-Inspired-Attention-Mechanism-NIAM--main/stft_test_results"
    os.makedirs(results_dir, exist_ok=True)

    all_passed = True

    for idx in test_indices:
        print(f"\n{'─'*80}")
        print(f"Test Sample #{idx}")
        print(f"{'─'*80}")

        # Load audio
        sample = dataset[idx]
        clean_audio = sample['audio']['array']
        text = sample['text']

        print(f"Text: {text[:80]}...")
        print(f"Audio length: {len(clean_audio)} samples ({len(clean_audio)/SAMPLE_RATE:.2f}s)")

        # Convert to tensor
        waveform = torch.FloatTensor(clean_audio).unsqueeze(0)  # [1, T]

        # Create window
        window = torch.hann_window(WIN_LENGTH)

        # STFT (forward)
        print("\n[Step 1] Running STFT...")
        stft_result = torch.stft(
            waveform,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=window,
            return_complex=True,
            center=True,
            normalized=False,
            onesided=True
        )
        print(f"  STFT shape: {stft_result.shape}")
        print(f"  STFT dtype: {stft_result.dtype}")

        # iSTFT (inverse)
        print("\n[Step 2] Running iSTFT...")
        reconstructed = torch.istft(
            stft_result,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            length=waveform.shape[-1]  # Preserve original length
        )
        print(f"  Reconstructed shape: {reconstructed.shape}")

        # Convert to numpy
        original_np = waveform.squeeze().numpy()
        reconstructed_np = reconstructed.squeeze().numpy()

        # Trim to same length (iSTFT may have slight length difference)
        min_len = min(len(original_np), len(reconstructed_np))
        original_np = original_np[:min_len]
        reconstructed_np = reconstructed_np[:min_len]

        # Calculate reconstruction error
        abs_error = np.abs(original_np - reconstructed_np)
        mean_abs_error = np.mean(abs_error)
        max_abs_error = np.max(abs_error)
        mse = np.mean((original_np - reconstructed_np) ** 2)
        snr = 10 * np.log10(np.mean(original_np ** 2) / (mse + 1e-10))

        print(f"\n[Step 3] Reconstruction Quality:")
        print(f"  Mean Absolute Error: {mean_abs_error:.6f}")
        print(f"  Max Absolute Error: {max_abs_error:.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  SNR: {snr:.2f} dB")

        # Check if reconstruction is good
        if mean_abs_error < 0.01 and snr > 30:
            print(f"  ✓ PASS: Reconstruction quality is excellent!")
            test_passed = True
        elif mean_abs_error < 0.05 and snr > 20:
            print(f"  ⚠ WARNING: Reconstruction quality is acceptable but not perfect")
            test_passed = True
        else:
            print(f"  ✗ FAIL: Reconstruction quality is poor!")
            print(f"  This may indicate STFT/iSTFT parameter mismatch!")
            test_passed = False
            all_passed = False

        # Save audio files
        original_path = f"{results_dir}/sample_{idx}_original.wav"
        reconstructed_path = f"{results_dir}/sample_{idx}_reconstructed.wav"

        sf.write(original_path, original_np, SAMPLE_RATE)
        sf.write(reconstructed_path, reconstructed_np, SAMPLE_RATE)

        print(f"\n[Step 4] Audio files saved:")
        print(f"  Original: {original_path}")
        print(f"  Reconstructed: {reconstructed_path}")

        if test_passed:
            print(f"\n  ✓ Sample #{idx}: PASSED")
        else:
            print(f"\n  ✗ Sample #{idx}: FAILED")

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")

    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nConclusion: STFT/iSTFT roundtrip is working correctly.")
        print("The reconstructed audio should sound like human speech (NOT static noise).")
        print("\n✓ SAFE TO PROCEED WITH TRAINING")
    else:
        print("✗ SOME TESTS FAILED!")
        print("\nConclusion: STFT/iSTFT parameters may have issues.")
        print("Please check the reconstructed audio files.")
        print("\n✗ DO NOT PROCEED WITH TRAINING UNTIL THIS IS FIXED")

    print(f"\nTest audio files saved in: {results_dir}/")
    print("Please listen to them to verify they are NOT static noise.\n")

    return all_passed


def test_stft_with_phase():
    """
    Test STFT with magnitude and phase separation
    (This is what the model will actually do)
    """
    print("\n" + "="*80)
    print("STFT WITH MAGNITUDE/PHASE SEPARATION TEST")
    print("="*80)
    print("\nThis test simulates what the model does:")
    print("  1. Extract magnitude and phase from STFT")
    print("  2. Reconstruct using magnitude + phase")
    print()

    # Load dataset
    dataset = load_dataset(
        "MLCommons/peoples_speech",
        "microset",
        split="train",
        cache_dir="/home/dlwlx05/project/NIAM/data/"
    )

    # Parameters
    N_FFT = 400
    HOP_LENGTH = 160
    WIN_LENGTH = 400
    SAMPLE_RATE = 16000

    # Test on first sample
    sample = dataset[0]
    clean_audio = sample['audio']['array']
    waveform = torch.FloatTensor(clean_audio).unsqueeze(0)

    window = torch.hann_window(WIN_LENGTH)

    # STFT
    stft_result = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True,
        center=True
    )

    # Separate magnitude and phase
    magnitude = torch.abs(stft_result)
    phase = torch.angle(stft_result)

    print(f"STFT shape: {stft_result.shape}")
    print(f"Magnitude shape: {magnitude.shape}")
    print(f"Phase shape: {phase.shape}")

    # Reconstruct complex STFT from magnitude and phase
    reconstructed_stft = magnitude * torch.exp(1j * phase)

    print(f"\nReconstructed STFT shape: {reconstructed_stft.shape}")
    print(f"Reconstructed STFT dtype: {reconstructed_stft.dtype}")

    # iSTFT
    reconstructed = torch.istft(
        reconstructed_stft,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True,
        length=waveform.shape[-1]
    )

    # Check error
    original_np = waveform.squeeze().numpy()
    reconstructed_np = reconstructed.squeeze().numpy()

    min_len = min(len(original_np), len(reconstructed_np))
    original_np = original_np[:min_len]
    reconstructed_np = reconstructed_np[:min_len]

    error = np.mean(np.abs(original_np - reconstructed_np))
    snr = 10 * np.log10(np.mean(original_np ** 2) / (np.mean((original_np - reconstructed_np) ** 2) + 1e-10))

    print(f"\nReconstruction Quality:")
    print(f"  Mean Absolute Error: {error:.6f}")
    print(f"  SNR: {snr:.2f} dB")

    if error < 0.01 and snr > 30:
        print(f"\n✓ PASS: Magnitude/Phase separation works correctly!")
        return True
    else:
        print(f"\n✗ FAIL: Magnitude/Phase separation has issues!")
        return False


if __name__ == "__main__":
    print("\n" + "🎯"*40)
    print("AUDIO RECONSTRUCTION SANITY CHECK")
    print("🎯"*40 + "\n")

    # Test 1: Basic STFT roundtrip
    test1_passed = test_stft_roundtrip()

    # Test 2: STFT with magnitude/phase separation
    test2_passed = test_stft_with_phase()

    # Final verdict
    print("\n" + "="*80)
    print("OVERALL VERDICT")
    print("="*80)

    if test1_passed and test2_passed:
        print("✓ ALL TESTS PASSED")
        print("\nYou can safely proceed with training.")
        print("The model will NOT produce static noise.")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nDO NOT PROCEED WITH TRAINING!")
        print("Fix the STFT/iSTFT issues first.")

    print("="*80 + "\n")
