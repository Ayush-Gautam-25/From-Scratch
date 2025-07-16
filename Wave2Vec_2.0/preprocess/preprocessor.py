import torchaudio
import torchaudio.transforms as T

class AudioPreprocessor:
    def __init__(self, target_sr=16000, normalize=True, max_duration=None):
        """
        Args:
            target_sr (int): Target sampling rate (e.g., 16000 Hz)
            normalize (bool): Whether to normalize waveform to [-1, 1]
            max_duration (float): If set, trims audio to this duration in seconds
        """
        self.target_sr = target_sr
        self.normalize = normalize
        self.max_duration = max_duration

    def __call__(self, path):
        # Load audio
        waveform, sr = torchaudio.load(path)

        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # Normalize to [-1, 1]
        if self.normalize:
            waveform = waveform / waveform.abs().max()

        # Truncate to max duration (in seconds)
        if self.max_duration:
            max_len = int(self.target_sr * self.max_duration)
            waveform = waveform[:, :max_len]

        return waveform.squeeze(0)  # return shape: [T]
