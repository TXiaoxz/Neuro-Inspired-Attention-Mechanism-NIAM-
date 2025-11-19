"""
NIAM Test Version: Simple + Temporal Focus
用于测试Temporal Focus模块是否有问题

包含：
- Selective Attention ✓
- Frequency Tuning ✓
- Temporal Focus ✓ (软阈值版)
- Noise Adaptation ✗
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveAttentionModule(nn.Module):
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
        attended = attended.transpose(1, 2)
        attended = self.norm(attended)
        attended = attended.transpose(1, 2)
        return attended


class FrequencyTuningLayer(nn.Module):
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


class TemporalFocusMechanism(nn.Module):
    """改进版：使用更温和的软阈值"""
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
        
        # 更温和的软阈值（乘数从5降到2）
        focus_mask = torch.sigmoid((context - 0.3) * 2)  # 阈值从0.5降到0.3
        
        focus_weights = self.focus_generator(x)
        focused = x * focus_weights * focus_mask
        
        focused = focused.transpose(1, 2)
        focused = self.norm(focused)
        focused = focused.transpose(1, 2)
        return focused


class NIAM(nn.Module):
    """Simple + Temporal Focus"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.selective_attention = SelectiveAttentionModule(hidden_dim)
        self.frequency_tuning = FrequencyTuningLayer(hidden_dim, num_bands=3)
        self.temporal_focus = TemporalFocusMechanism(hidden_dim, window_size=5)
        
        self.residual_weight = 0.3
        
        print("="*70)
        print("NIAM Test: Simple + Temporal Focus")
        print("="*70)
        print("Module 1: Selective Attention - ENABLED")
        print("Module 2: Frequency Tuning - ENABLED")
        print("Module 3: Temporal Focus - ENABLED (soft threshold)")
        print("Module 4: Noise Adaptation - DISABLED")
        print("="*70)
        
    def forward(self, x):
        need_transpose_back = False
        if x.dim() == 3:
            if x.shape[2] == self.hidden_dim and x.shape[1] != self.hidden_dim:
                x = x.transpose(1, 2)
                need_transpose_back = True
        
        residual = x
        
        # Module 1
        x = self.selective_attention(x)
        x = x + residual * self.residual_weight
        residual = x
        
        # Module 2
        x = self.frequency_tuning(x)
        x = x + residual * self.residual_weight
        residual = x
        
        # Module 3
        x = self.temporal_focus(x)
        x = x + residual * self.residual_weight
        
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        if need_transpose_back:
            x = x.transpose(1, 2)
        
        return x


if __name__ == "__main__":
    x = torch.randn(2, 256, 100)
    niam = NIAM(hidden_dim=256)
    output = niam(x)
    print(f"✓ Test passed! Shape: {output.shape}")