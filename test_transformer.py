#!/usr/bin/env python3
"""
Quick test script for Transformer model
Tests if the model can run without errors
Much faster than full training
"""
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from train_transformer_cpu import TransformerEnhancer, PositionalEncoding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("\n" + "="*60)
print("Test 1: Create TransformerEnhancer model")
print("="*60)

try:
    model = TransformerEnhancer(
        n_mels=80,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created successfully")
    print(f"Total parameters: {total_params:,}")
    print("PASS")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("Test 2: Forward pass with dummy data")
print("="*60)

try:
    # Create dummy mel-spectrogram
    # Shape: (batch, n_mels, time)
    batch_size = 2
    n_mels = 80
    time_steps = 100
    
    dummy_input = torch.randn(batch_size, n_mels, time_steps).to(device)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    assert output.shape == dummy_input.shape, "Shape mismatch"
    print("PASS")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("Test 3: Backward pass (gradient computation)")
print("="*60)

try:
    # Create dummy input with gradient
    dummy_input = torch.randn(2, 80, 100, requires_grad=True).to(device)
    
    # Forward pass
    output = model(dummy_input)
    
    # Compute loss
    target = torch.randn_like(output)
    loss = torch.nn.functional.mse_loss(output, target)
    
    # Backward pass
    loss.backward()
    
    print(f"Loss: {loss.item():.6f}")
    print(f"Input gradient shape: {dummy_input.grad.shape}")
    print("PASS")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("Test 4: PositionalEncoding")
print("="*60)

try:
    pos_enc = PositionalEncoding(d_model=256, max_len=1000)
    x = torch.randn(2, 100, 256)  # (batch, seq_len, d_model)
    output = pos_enc(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Shape mismatch"
    print("PASS")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED")
print("="*60)
print("\nYour Transformer model is working correctly!")
print("\nNext steps:")
print("1. Run: python train_transformer_cpu.py")
print("   (This will download data and start training)")
print("2. After training works, add your Neuro-Inspired optimizer")