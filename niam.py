"""
Neuro-Inspired Attention Mechanism (NIAM)
Complete implementation matching project requirements

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
    Inspired by cocktail party effect - focus on target while suppressing noise
    """
    def __init__(self, hidden_dim):
        super().__init__()
        
        # SNR estimation network
        self.snr_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Attention weight generator
        self.attention_generator = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, hidden_dim, time)
        Returns:
            attended: Features with selective attention applied (batch, hidden_dim, time)
            attention_weights: Attention map (batch, time, hidden_dim)
        """
        # Input is (batch, hidden_dim, time)
        batch_size, hidden_dim, time_steps = x.shape
        
        # Transpose for linear layers: (batch, time, hidden_dim)
        x_t = x.transpose(1, 2)
        
        # Estimate SNR for each time-frequency point
        snr = self.snr_estimator(x_t)  # (batch, time, 1)
        
        # Generate attention weights based on SNR
        attn_input = torch.cat([x_t, snr], dim=-1)  # (batch, time, hidden_dim+1)
        attention_weights = self.attention_generator(attn_input)  # (batch, time, hidden_dim)
        
        # Apply selective attention
        attended = x_t * attention_weights  # (batch, time, hidden_dim)
        
        # Convert back to (batch, hidden_dim, time)
        attended = attended.transpose(1, 2)
        
        return attended, attention_weights


class FrequencyTuningLayer(nn.Module):
    """
    Module 2: Frequency Tuning Layer
    Inspired by cochlear frequency selectivity
    Decomposes input into multiple frequency bands and processes separately
    """
    def __init__(self, hidden_dim, num_bands=3):
        super().__init__()
        self.num_bands = num_bands
        
        # Calculate dimensions for each band
        base_dim = hidden_dim // num_bands
        remainder = hidden_dim % num_bands
        
        # Multi-scale frequency filters (different kernel sizes for different frequencies)
        self.low_freq_filter = nn.Conv1d(
            hidden_dim, base_dim,
            kernel_size=7, padding=3
        )  # Low frequency: larger receptive field
        
        self.mid_freq_filter = nn.Conv1d(
            hidden_dim, base_dim,
            kernel_size=5, padding=2
        )  # Mid frequency: medium receptive field
        
        self.high_freq_filter = nn.Conv1d(
            hidden_dim, base_dim + remainder,
            kernel_size=3, padding=1
        )  # High frequency: small receptive field
        
        # Learnable frequency band weights
        self.band_weights = nn.Parameter(torch.ones(num_bands) / num_bands)
        
        # Normalization for each band
        self.norm_low = nn.BatchNorm1d(base_dim)
        self.norm_mid = nn.BatchNorm1d(base_dim)
        self.norm_high = nn.BatchNorm1d(base_dim + remainder)
        
    def forward(self, x):
        """
        Args:
            x: (batch, hidden_dim, time)
        Returns:
            tuned: Frequency-tuned features (batch, hidden_dim, time)
        """
        # Process each frequency band
        low = F.relu(self.norm_low(self.low_freq_filter(x)))
        mid = F.relu(self.norm_mid(self.mid_freq_filter(x)))
        high = F.relu(self.norm_high(self.high_freq_filter(x)))
        
        # Apply learned weights to each band
        low = low * self.band_weights[0]
        mid = mid * self.band_weights[1]
        high = high * self.band_weights[2]
        
        # Concatenate all bands
        tuned = torch.cat([low, mid, high], dim=1)
        
        return tuned


class TemporalFocusMechanism(nn.Module):
    """
    Module 3: Temporal Focus Mechanism
    Inspired by temporal masking and forward masking effects
    Suppresses weak signals around strong signals
    """
    def __init__(self, hidden_dim, window_size=5):
        super().__init__()
        self.window_size = window_size
        
        # Energy estimation
        self.energy_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Temporal context aggregation
        self.temporal_conv = nn.Conv1d(
            1, 1,
            kernel_size=window_size,
            padding=window_size // 2
        )
        
        # Focus weight generator using convolution (no Linear layer issues)
        self.focus_generator = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, hidden_dim, time)
        Returns:
            focused: Temporally focused features (batch, hidden_dim, time)
        """
        # Estimate energy at each time point
        x_t = x.transpose(1, 2)  # (batch, time, hidden_dim)
        energy = self.energy_estimator(x_t)  # (batch, time, 1)
        
        # Aggregate temporal context
        energy_t = energy.transpose(1, 2)  # (batch, 1, time)
        context = self.temporal_conv(energy_t)  # (batch, 1, time)
        
        # Determine focus regions (strong signals suppress weak neighbors)
        threshold = 0.5
        focus_mask = (context > threshold).float()
        
        # Generate focus weights (input is already (batch, hidden_dim, time))
        focus_weights = self.focus_generator(x)
        
        # Apply temporal focus with masking
        focused = x * focus_weights * focus_mask
        
        return focused


class NoiseAdaptationController(nn.Module):
    """
    Module 4: Noise Adaptation Controller
    Dynamically adapts to noise characteristics
    Estimates noise profile and adjusts processing accordingly
    """
    def __init__(self, hidden_dim):
        super().__init__()
        
        # Noise profile estimator (using conv to avoid dimension issues)
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
            adapted: Noise-adapted features (batch, hidden_dim, time)
        """
        # Estimate global noise profile
        noise_profile = self.noise_profiler(x)  # (batch, hidden_dim, 1)
        
        # Apply adaptive filtering
        filtered = self.adaptive_filter(x)
        
        # Modulate by noise profile
        adapted = filtered * noise_profile
        
        return adapted


class NIAM(nn.Module):
    """
    Complete Neuro-Inspired Attention Mechanism (NIAM)
    Integrates all 4 components with residual connections
    
    Target: +12-15 dB SI-SNR improvement
    """
    def __init__(self, hidden_dim, enable_all=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Module 1: Selective Attention
        self.selective_attention = SelectiveAttentionModule(hidden_dim)
        
        # Module 2: Frequency Tuning
        self.frequency_tuning = FrequencyTuningLayer(hidden_dim, num_bands=3)
        
        # Module 3: Temporal Focus
        self.temporal_focus = TemporalFocusMechanism(hidden_dim, window_size=5)
        
        # Module 4: Noise Adaptation
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
            x: Input features (batch, hidden_dim, time) or (batch, time, hidden_dim)
        Returns:
            output: Enhanced features with same shape as input
        """
        # Standardize to (batch, hidden_dim, time)
        if x.dim() == 3:
            if x.shape[1] > x.shape[2]:  # (batch, time, hidden_dim)
                x = x.transpose(1, 2)
                need_transpose_back = True
            else:
                need_transpose_back = False
        else:
            need_transpose_back = False
        
        # Now x is guaranteed to be (batch, hidden_dim, time)
        residual = x
        
        # Module 1: Selective Attention with residual (reduced weight)
        attended, attn_weights = self.selective_attention(x)
        x = attended + residual * 0.1  # Changed from 0.5 to 0.1
        residual = x
        
        # Module 2: Frequency Tuning with residual (reduced weight)
        tuned = self.frequency_tuning(x)
        x = tuned + residual * 0.1  # Changed from 0.5 to 0.1
        residual = x
        
        # Module 3: Temporal Focus with residual (reduced weight)
        focused = self.temporal_focus(x)
        x = focused + residual * 0.1  # Changed from 0.5 to 0.1
        residual = x
        
        # Module 4: Noise Adaptation with residual (reduced weight)
        adapted = self.noise_adaptation(x)
        x = adapted + residual * 0.1  # Changed from 0.5 to 0.1
        
        # Safety check: clip extreme values and check for NaN
        x = torch.clamp(x, min=-10.0, max=10.0)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN or Inf detected in NIAM output, replacing with residual")
            x = residual
        
        # Convert back to original format if needed
        if need_transpose_back:
            x = x.transpose(1, 2)
        
        return x


# Quick test
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
    assert output.shape == x.shape, "Shape mismatch!"
    print("\nShape check: PASS")
    
    # Test gradient
    loss = output.mean()
    loss.backward()
    print("Gradient check: PASS")
    
    # Count parameters
    total_params = sum(p.numel() for p in niam.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n" + "="*70)
    print("NIAM module test completed successfully!")
    print("Ready for integration into Transformer")
    print("="*70)