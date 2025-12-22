import dataclasses

import crepe
import numpy as np
from pathlib import Path

from scipy.io import wavfile
from resampy import resample


DATA_PATH = Path('../')
DATA_GLOB = '*.wav'

TARGET_SAMPLE_RATE = 16_000
SAMPLES_PER_CHUNK = 1_024

@dataclasses.dataclass
class TestDataTuple:
    audio_samples: np.ndarray
    frequency: np.ndarray
    confidence: np.ndarray

def generate_input_and_output_data(wav_file: Path) -> TestDataTuple:
    """
    Generates raw audio samples, pitch and confidence predictions from the given file.
    :param wav_file: The wave file to generate predictions for. If it does not have the needed format (mono and 16 kHz sampling rate),
    :return: A TestDataTuple with the binary data.
    """
    # Taken from core.py of CREPE:
    sr, audio = wavfile.read(wav_file)
    if len(audio.shape) == 2:
        audio = audio.mean(1)  # make mono
    audio = audio.astype(np.float32)

    if sr != TARGET_SAMPLE_RATE:
        audio = resample(audio, sr, TARGET_SAMPLE_RATE)

    _, frequency, confidence, __ = crepe.predict(
        audio=audio,
        sr=sr,
        model_capacity='full',
        center=False,
        step_size=int((SAMPLES_PER_CHUNK / TARGET_SAMPLE_RATE) * 1000),
    )

    return TestDataTuple(audio_samples=audio, frequency=frequency, confidence=confidence)

def save_as_npy(original_file: Path, name: str, data: np.ndarray) -> None:
    np.save(Path(original_file.parent / f'{original_file.stem}_{name}.npy'), data)

def main():
    for file in DATA_PATH.glob(DATA_GLOB):
        test_data_tuple = generate_input_and_output_data(file)
        save_as_npy(original_file=file, name='samples', data=test_data_tuple.audio_samples)
        save_as_npy(original_file=file, name='frequencies', data=test_data_tuple.frequency)
        save_as_npy(original_file=file, name='confidences', data=test_data_tuple.confidence)


if __name__ == '__main__':
    main()