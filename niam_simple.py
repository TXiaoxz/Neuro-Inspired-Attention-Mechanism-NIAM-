"""
NIAM Simple - Simplified Neuro-Inspired Attention Mechanism
去掉了可能导致"电流声"的模块

Only keeps:
1. Selective Attention (核心功能)
2. Frequency Tuning (多尺度处理)

Removed:
- Temporal Focus (硬阈值可能导致电流声)
- Noise Adaptation (可能过度抑制)

Target: Stable improvement over baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveAttentionModule(nn.Module):
    """Module 1: Selective Attention - 保留"""
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
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        snr = self.snr_estimator(x)
        attn_input = torch.cat([x, snr], dim=1)
        attention_weights = self.attention_generator(attn_input)
        attended = x * attention_weights
        
        # LayerNorm
        attended = attended.transpose(1, 2)
        attended = self.norm(attended)
        attended = attended.transpose(1, 2)
        
        return attended


class FrequencyTuningLayer(nn.Module):
    """Module 2: Frequency Tuning - 保留"""
    def __init__(self, hidden_dim, num_bands=3):
        super().__init__()
        
        base_dim = hidden_dim // num_bands
        remainder = hidden_dim % num_bands
        
        self.low_freq_filter = nn.Conv1d(hidden_dim, base_dim, kernel_size=7, padding=3)
        self.mid_freq_filter = nn.Conv1d(hidden_dim, base_dim, kernel_size=5, padding=2)
        self.high_freq_filter = nn.Conv1d(hidden_dim, base_dim + remainder, kernel_size=3, padding=1)
        
        self.norm_low = nn.BatchNorm1d(base_dim)
        self.norm_mid = nn.BatchNorm1d(base_dim)
        self.norm_high = nn.BatchNorm1d(base_dim + remainder)
        
        self.output_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        low = F.relu(self.norm_low(self.low_freq_filter(x)))
        mid = F.relu(self.norm_mid(self.mid_freq_filter(x)))
        high = F.relu(self.norm_high(self.high_freq_filter(x)))
        
        tuned = torch.cat([low, mid, high], dim=1)
        
        tuned = tuned.transpose(1, 2)
        tuned = self.output_norm(tuned)
        tuned = tuned.transpose(1, 2)
        
        return tuned


class NIAM(nn.Module):
    """
    NIAM Simple - Only 2 modules
    去掉了Temporal Focus和Noise Adaptation
    这两个模块可能导致音频失真
    """
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Only 2 modules (removed problematic ones)
        self.selective_attention = SelectiveAttentionModule(hidden_dim)
        self.frequency_tuning = FrequencyTuningLayer(hidden_dim, num_bands=3)
        
        # Fixed residual weight = 0.3 (balanced)
        self.residual_weight = 0.3
        
        print("="*70)
        print("NIAM Simple Initialized")
        print("="*70)
        print("Module 1: Selective Attention - ENABLED")
        print("Module 2: Frequency Tuning - ENABLED")
        print("Module 3: Temporal Focus - DISABLED (may cause artifacts)")
        print("Module 4: Noise Adaptation - DISABLED (may over-suppress)")
        print(f"Residual weight: {self.residual_weight}")
        print("Target: Stable improvement without artifacts")
        print("="*70)
        
    def forward(self, x):
        # Handle input format
        need_transpose_back = False
        if x.dim() == 3:
            if x.shape[2] == self.hidden_dim and x.shape[1] != self.hidden_dim:
                x = x.transpose(1, 2)
                need_transpose_back = True
        
        residual = x
        
        # Module 1: Selective Attention
        x = self.selective_attention(x)
        x = x + residual * self.residual_weight
        residual = x
        
        # Module 2: Frequency Tuning
        x = self.frequency_tuning(x)
        x = x + residual * self.residual_weight
        
        # Safety check
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        if need_transpose_back:
            x = x.transpose(1, 2)
        
        return x


if __name__ == "__main__":
    print("Testing NIAM Simple...")
    
    x = torch.randn(2, 256, 100)
    niam = NIAM(hidden_dim=256)
    output = niam(x)
    
    assert output.shape == x.shape
    print(f"\n✓ Test passed! Shape: {output.shape}")
    
    total_params = sum(p.numel() for p in niam.parameters())
    print(f"Parameters: {total_params:,} (fewer than full NIAM)")