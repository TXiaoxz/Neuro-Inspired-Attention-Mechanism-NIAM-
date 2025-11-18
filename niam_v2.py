"""
NIAM v2 - Improved Neuro-Inspired Attention Mechanism
Changes from v1:
1. Residual weight: 0.2 (balanced between 0.1 and 0.5)
2. Added LayerNorm for training stability
3. Learnable residual weights
4. Better gradient flow

Target: +12-15 dB SI-SNR improvement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveAttentionModule(nn.Module):
    """Module 1: Selective Attention"""
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.snr_estimator = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.attention_generator = nn.Sequential(
            nn.Conv1d(hidden_dim + 1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x: (batch, hidden_dim, time)
        snr = self.snr_estimator(x)
        attn_input = torch.cat([x, snr], dim=1)
        attention_weights = self.attention_generator(attn_input)
        attended = x * attention_weights
        
        # Apply LayerNorm (need to transpose)
        attended = attended.transpose(1, 2)  # (batch, time, hidden_dim)
        attended = self.norm(attended)
        attended = attended.transpose(1, 2)  # (batch, hidden_dim, time)
        
        return attended


class FrequencyTuningLayer(nn.Module):
    """Module 2: Frequency Tuning Layer"""
    def __init__(self, hidden_dim, num_bands=3):
        super().__init__()
        self.num_bands = num_bands
        
        base_dim = hidden_dim // num_bands
        remainder = hidden_dim % num_bands
        
        self.low_freq_filter = nn.Conv1d(hidden_dim, base_dim, kernel_size=7, padding=3)
        self.mid_freq_filter = nn.Conv1d(hidden_dim, base_dim, kernel_size=5, padding=2)
        self.high_freq_filter = nn.Conv1d(hidden_dim, base_dim + remainder, kernel_size=3, padding=1)
        
        self.norm_low = nn.BatchNorm1d(base_dim)
        self.norm_mid = nn.BatchNorm1d(base_dim)
        self.norm_high = nn.BatchNorm1d(base_dim + remainder)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        low = F.relu(self.norm_low(self.low_freq_filter(x)))
        mid = F.relu(self.norm_mid(self.mid_freq_filter(x)))
        high = F.relu(self.norm_high(self.high_freq_filter(x)))
        
        tuned = torch.cat([low, mid, high], dim=1)
        
        # Apply LayerNorm
        tuned = tuned.transpose(1, 2)
        tuned = self.output_norm(tuned)
        tuned = tuned.transpose(1, 2)
        
        return tuned


class TemporalFocusMechanism(nn.Module):
    """Module 3: Temporal Focus Mechanism"""
    def __init__(self, hidden_dim, window_size=5):
        super().__init__()
        
        self.energy_estimator = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.temporal_conv = nn.Conv1d(1, 1, kernel_size=window_size, padding=window_size // 2)
        
        self.focus_generator = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        energy = self.energy_estimator(x)
        context = self.temporal_conv(energy)
        
        # Soft threshold instead of hard (better gradient flow)
        focus_mask = torch.sigmoid((context - 0.5) * 5)
        
        focus_weights = self.focus_generator(x)
        focused = x * focus_weights * focus_mask
        
        # Apply LayerNorm
        focused = focused.transpose(1, 2)
        focused = self.norm(focused)
        focused = focused.transpose(1, 2)
        
        return focused


class NoiseAdaptationController(nn.Module):
    """Module 4: Noise Adaptation Controller"""
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.noise_profiler = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_dim, hidden_dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 4, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.adaptive_filter = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        noise_profile = self.noise_profiler(x)
        filtered = self.adaptive_filter(x)
        adapted = filtered * noise_profile
        
        # Apply LayerNorm
        adapted = adapted.transpose(1, 2)
        adapted = self.norm(adapted)
        adapted = adapted.transpose(1, 2)
        
        return adapted


class NIAM(nn.Module):
    """
    NIAM v2 - Improved version
    Key improvements:
    1. Learnable residual weights (initialized to 0.2)
    2. LayerNorm after each module
    3. Soft thresholding for better gradients
    """
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 4 bio-inspired modules
        self.selective_attention = SelectiveAttentionModule(hidden_dim)
        self.frequency_tuning = FrequencyTuningLayer(hidden_dim, num_bands=3)
        self.temporal_focus = TemporalFocusMechanism(hidden_dim, window_size=5)
        self.noise_adaptation = NoiseAdaptationController(hidden_dim)
        
        # Learnable residual weights (initialized to 0.2)
        self.residual_weight_1 = nn.Parameter(torch.tensor(0.2))
        self.residual_weight_2 = nn.Parameter(torch.tensor(0.2))
        self.residual_weight_3 = nn.Parameter(torch.tensor(0.2))
        self.residual_weight_4 = nn.Parameter(torch.tensor(0.2))
        
        print("="*70)
        print("NIAM v2 (Improved) Initialized")
        print("="*70)
        print("Module 1: Selective Attention Module - ENABLED")
        print("Module 2: Frequency Tuning Layer - ENABLED")
        print("Module 3: Temporal Focus Mechanism - ENABLED")
        print("Module 4: Noise Adaptation Controller - ENABLED")
        print("Improvements: LayerNorm, Learnable residual weights (init=0.2)")
        print("Target: +12-15 dB SI-SNR improvement")
        print("="*70)
        
    def forward(self, x):
        # Handle input format
        need_transpose_back = False
        if x.dim() == 3:
            if x.shape[2] == self.hidden_dim and x.shape[1] != self.hidden_dim:
                x = x.transpose(1, 2)
                need_transpose_back = True
        
        # Now x is (batch, hidden_dim, time)
        residual = x
        
        # Module 1: Selective Attention
        x = self.selective_attention(x)
        x = x + residual * self.residual_weight_1
        residual = x
        
        # Module 2: Frequency Tuning
        x = self.frequency_tuning(x)
        x = x + residual * self.residual_weight_2
        residual = x
        
        # Module 3: Temporal Focus
        x = self.temporal_focus(x)
        x = x + residual * self.residual_weight_3
        residual = x
        
        # Module 4: Noise Adaptation
        x = self.noise_adaptation(x)
        x = x + residual * self.residual_weight_4
        
        # Safety check
        x = torch.clamp(x, min=-10.0, max=10.0)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN/Inf detected, using residual")
            x = residual
        
        # Convert back if needed
        if need_transpose_back:
            x = x.transpose(1, 2)
        
        return x


# Test
if __name__ == "__main__":
    print("Testing NIAM v2...")
    
    batch_size = 2
    hidden_dim = 256
    time_steps = 100
    
    x = torch.randn(batch_size, hidden_dim, time_steps)
    print(f"\nInput shape: {x.shape}")
    
    niam = NIAM(hidden_dim=hidden_dim)
    output = niam(x)
    print(f"Output shape: {output.shape}")
    
    assert output.shape == x.shape
    print("\n✓ NIAM v2 test passed!")
    
    # Show learnable weights
    print(f"\nLearnable residual weights:")
    print(f"  Module 1: {niam.residual_weight_1.item():.3f}")
    print(f"  Module 2: {niam.residual_weight_2.item():.3f}")
    print(f"  Module 3: {niam.residual_weight_3.item():.3f}")
    print(f"  Module 4: {niam.residual_weight_4.item():.3f}")
    
    total_params = sum(p.numel() for p in niam.parameters())
    print(f"\nTotal parameters: {total_params:,}")