"""
Audio utility functions for the MLSP project.
"""

import soundfile as sf
import io


def decode_audio(sample):
    """
    Decode audio from Hugging Face dataset sample.

    Args:
        sample: Dataset sample with 'audio' field containing 'bytes'

    Returns:
        audio_array: Numpy array of audio data
    """
    audio_bytes = sample['audio']['bytes']
    audio_array, sr = sf.read(io.BytesIO(audio_bytes))
    return audio_array
