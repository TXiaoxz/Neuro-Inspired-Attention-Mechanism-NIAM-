"""
NIAM - Neuro-Inspired Attention Mechanism
Simplified version using only convolutions to avoid dimension issues

Components:
1. Selective Attention Module
2. Frequency Tuning Layer
3. Temporal Focus Mechanism
4. Noise Adaptation Controller

Target: +12-15 dB SI-SNR improvement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveAttentionModule(nn.Module):
    """
    Module 1: Selective Attention
    Uses convolutions only - no Linear layers
    """
    def __init__(self, hidden_dim):
        super().__init__()
        
        # SNR estimation using 1D convolutions
        self.snr_estimator = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Attention weight generator
        self.attention_generator = nn.Sequential(
            nn.Conv1d(hidden_dim + 1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, hidden_dim, time)
        Returns:
            attended: (batch, hidden_dim, time)
        """
        # Estimate SNR
        snr = self.snr_estimator(x)  # (batch, 1, time)
        
        # Generate attention weights
        attn_input = torch.cat([x, snr], dim=1)  # (batch, hidden_dim+1, time)
        attention_weights = self.attention_generator(attn_input)  # (batch, hidden_dim, time)
        
        # Apply attention
        attended = x * attention_weights
        
        return attended


class FrequencyTuningLayer(nn.Module):
    """
    Module 2: Frequency Tuning Layer
    Multi-band frequency decomposition
    """
    def __init__(self, hidden_dim, num_bands=3):
        super().__init__()
        self.num_bands = num_bands
        
        base_dim = hidden_dim // num_bands
        remainder = hidden_dim % num_bands
        
        # Multi-scale filters
        self.low_freq_filter = nn.Conv1d(hidden_dim, base_dim, kernel_size=7, padding=3)
        self.mid_freq_filter = nn.Conv1d(hidden_dim, base_dim, kernel_size=5, padding=2)
        self.high_freq_filter = nn.Conv1d(hidden_dim, base_dim + remainder, kernel_size=3, padding=1)
        
        # Normalization
        self.norm_low = nn.BatchNorm1d(base_dim)
        self.norm_mid = nn.BatchNorm1d(base_dim)
        self.norm_high = nn.BatchNorm1d(base_dim + remainder)
        
    def forward(self, x):
        """
        Args:
            x: (batch, hidden_dim, time)
        Returns:
            tuned: (batch, hidden_dim, time)
        """
        low = F.relu(self.norm_low(self.low_freq_filter(x)))
        mid = F.relu(self.norm_mid(self.mid_freq_filter(x)))
        high = F.relu(self.norm_high(self.high_freq_filter(x)))
        
        tuned = torch.cat([low, mid, high], dim=1)
        return tuned


class TemporalFocusMechanism(nn.Module):
    """
    Module 3: Temporal Focus Mechanism
    Temporal masking effect
    """
    def __init__(self, hidden_dim, window_size=5):
        super().__init__()
        
        # Energy estimation
        self.energy_estimator = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Temporal context
        self.temporal_conv = nn.Conv1d(1, 1, kernel_size=window_size, padding=window_size // 2)
        
        # Focus weight generator
        self.focus_generator = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, hidden_dim, time)
        Returns:
            focused: (batch, hidden_dim, time)
        """
        # Estimate energy
        energy = self.energy_estimator(x)  # (batch, 1, time)
        
        # Temporal context
        context = self.temporal_conv(energy)  # (batch, 1, time)
        
        # Focus mask
        threshold = 0.5
        focus_mask = (context > threshold).float()
        
        # Generate focus weights
        focus_weights = self.focus_generator(x)  # (batch, hidden_dim, time)
        
        # Apply focus
        focused = x * focus_weights * focus_mask
        
        return focused


class NoiseAdaptationController(nn.Module):
    """
    Module 4: Noise Adaptation Controller
    Adaptive noise cancellation
    """
    def __init__(self, hidden_dim):
        super().__init__()
        
        # Noise profile estimator
        self.noise_profiler = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_dim, hidden_dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 4, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Adaptive filter
        self.adaptive_filter = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, hidden_dim, time)
        Returns:
            adapted: (batch, hidden_dim, time)
        """
        noise_profile = self.noise_profiler(x)  # (batch, hidden_dim, 1)
        filtered = self.adaptive_filter(x)
        adapted = filtered * noise_profile
        
        return adapted


class NIAM(nn.Module):
    """
    Complete NIAM with all 4 modules
    All operations use convolutions - no Linear layers
    """
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 4 bio-inspired modules
        self.selective_attention = SelectiveAttentionModule(hidden_dim)
        self.frequency_tuning = FrequencyTuningLayer(hidden_dim, num_bands=3)
        self.temporal_focus = TemporalFocusMechanism(hidden_dim, window_size=5)
        self.noise_adaptation = NoiseAdaptationController(hidden_dim)
        
        print("="*70)
        print("NIAM (Neuro-Inspired Attention Mechanism) Initialized")
        print("="*70)
        print("Module 1: Selective Attention Module - ENABLED")
        print("Module 2: Frequency Tuning Layer - ENABLED")
        print("Module 3: Temporal Focus Mechanism - ENABLED")
        print("Module 4: Noise Adaptation Controller - ENABLED")
        print("Target: +12-15 dB SI-SNR improvement")
        print("="*70)
        
    def forward(self, x):
        """
        Args:
            x: (batch, hidden_dim, time) or (batch, time, hidden_dim)
        Returns:
            output: Same shape as input
        """
        # Standardize to (batch, hidden_dim, time)
        # Assumption: hidden_dim is fixed (256), time_steps can vary
        # If dim[1] == hidden_dim, it's already correct format
        # If dim[2] == hidden_dim, need to transpose
        need_transpose_back = False
        if x.dim() == 3:
            # Check if it's (batch, time, hidden_dim) format
            if x.shape[2] == self.hidden_dim and x.shape[1] != self.hidden_dim:
                x = x.transpose(1, 2)
                need_transpose_back = True
        
        # Now x is guaranteed to be (batch, hidden_dim, time)
        residual = x
        
        # Module 1: Selective Attention
        x = self.selective_attention(x)
        x = x + residual * 0.5
        residual = x
        
        # Module 2: Frequency Tuning
        x = self.frequency_tuning(x)
        x = x + residual * 0.5
        residual = x
        
        # Module 3: Temporal Focus
        x = self.temporal_focus(x)
        x = x + residual * 0.5
        residual = x
        
        # Module 4: Noise Adaptation
        x = self.noise_adaptation(x)
        x = x + residual * 0.5
        
        # Convert back if needed
        if need_transpose_back:
            x = x.transpose(1, 2)
        
        return x


# Test
if __name__ == "__main__":
    print("Testing NIAM module...")
    
    batch_size = 2
    hidden_dim = 256
    time_steps = 100
    
    # Test input
    x = torch.randn(batch_size, hidden_dim, time_steps)
    print(f"\nInput shape: {x.shape}")
    
    # Create NIAM
    niam = NIAM(hidden_dim=hidden_dim)
    
    # Forward pass
    output = niam(x)
    print(f"Output shape: {output.shape}")
    
    # Check shape
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print("\nShape check: PASS")
    
    # Test gradient
    loss = output.mean()
    loss.backward()
    print("Gradient check: PASS")
    
    # Count parameters
    total_params = sum(p.numel() for p in niam.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test with different input format
    print("\nTesting with (batch, time, hidden_dim) format...")
    x2 = torch.randn(batch_size, time_steps, hidden_dim)
    output2 = niam(x2)
    assert output2.shape == x2.shape, "Format auto-detection failed"
    print("Format auto-detection: PASS")
    
    print("\n" + "="*70)
    print("NIAM module test completed successfully!")
    print("Ready for integration into Transformer")
    print("="*70)