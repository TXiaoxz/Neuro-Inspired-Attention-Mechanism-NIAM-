"""
NIAM Test Version: Simple + Noise Adaptation
用于测试Noise Adaptation模块是否有问题

包含：
- Selective Attention ✓
- Frequency Tuning ✓
- Temporal Focus ✗
- Noise Adaptation ✓ (改进版)
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


class NoiseAdaptationController(nn.Module):
    """改进版：防止过度抑制"""
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
        
        # 添加一个最小值，防止完全抑制
        self.min_scale = 0.3
        
    def forward(self, x):
        noise_profile = self.noise_profiler(x)
        
        # 确保noise_profile不会太小
        noise_profile = noise_profile * (1 - self.min_scale) + self.min_scale
        
        filtered = self.adaptive_filter(x)
        adapted = filtered * noise_profile
        
        adapted = adapted.transpose(1, 2)
        adapted = self.norm(adapted)
        adapted = adapted.transpose(1, 2)
        return adapted


class NIAM(nn.Module):
    """Simple + Noise Adaptation"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.selective_attention = SelectiveAttentionModule(hidden_dim)
        self.frequency_tuning = FrequencyTuningLayer(hidden_dim, num_bands=3)
        self.noise_adaptation = NoiseAdaptationController(hidden_dim)
        
        self.residual_weight = 0.3
        
        print("="*70)
        print("NIAM Test: Simple + Noise Adaptation")
        print("="*70)
        print("Module 1: Selective Attention - ENABLED")
        print("Module 2: Frequency Tuning - ENABLED")
        print("Module 3: Temporal Focus - DISABLED")
        print("Module 4: Noise Adaptation - ENABLED (min_scale=0.3)")
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
        
        # Module 4 (skip Module 3)
        x = self.noise_adaptation(x)
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