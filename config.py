"""
Project Configuration
=====================
Central configuration for all paths in the project.

Usage:
    from config import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, CHECKPOINTS_DIR

When migrating to a new machine:
    1. Update DATA_DIR to point to your dataset location
    2. PROJECT_ROOT, RESULTS_DIR, CHECKPOINTS_DIR will auto-update
"""

import os

# ============================================================================
# Project Paths (AUTO-UPDATED - uses relative paths)
# ============================================================================

# Project root directory (auto-detected from this file's location)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Results and checkpoints (inside project)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, 'niam_v2_refined', 'checkpoints')


# ============================================================================
# Data Paths (MANUAL - update when migrating to new machine)
# ============================================================================

# Dataset cache directory
# ** CHANGE THIS when moving to a new machine **
DATA_DIR = '/home/dlwlx05/project/NIAM/data'

# Clean SA dataset cache (for training)
CLEAN_SA_CACHE = os.path.join(DATA_DIR, 'clean_sa')

# General HuggingFace dataset cache
HF_CACHE_DIR = DATA_DIR


# ============================================================================
# Model Paths
# ============================================================================

# Default model checkpoint paths
NIAM_V2_REFINED_MODEL = os.path.join(
    CHECKPOINTS_DIR,
    'transformer_niam_v2_refined_best.pt'
)


# ============================================================================
# Helper Function
# ============================================================================

def get_model_path(model_name='niam_v2_refined'):
    """
    Get model checkpoint path by name.

    Args:
        model_name: One of ['niam_v2_refined', 'niam_v2', 'transformer', 'rnn']

    Returns:
        str: Full path to model checkpoint
    """
    model_paths = {
        'niam_v2_refined': NIAM_V2_REFINED_MODEL,
        'niam_v2': os.path.join(RESULTS_DIR, 'niam_v2', 'checkpoints', 'transformer_niam_v2_best.pt'),
    }

    return model_paths.get(model_name, NIAM_V2_REFINED_MODEL)


def print_config():
    """Print current configuration"""
    print("\n" + "="*80)
    print("Current Configuration")
    print("="*80)
    print(f"PROJECT_ROOT:     {PROJECT_ROOT}")
    print(f"DATA_DIR:         {DATA_DIR}")
    print(f"RESULTS_DIR:      {RESULTS_DIR}")
    print(f"CHECKPOINTS_DIR:  {CHECKPOINTS_DIR}")
    print(f"MODEL PATH:       {NIAM_V2_REFINED_MODEL}")
    print("="*80 + "\n")


if __name__ == "__main__":
    print_config()
