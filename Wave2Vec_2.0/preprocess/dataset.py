from torch.utils.data import Dataset
from pathlib import Path
import torchaudio

class LibriSpeechDataset(Dataset):
    def __init__(self, root_dir, preprocessor=None, ext=".flac", max_files=None):
        """
        Args:
            root_dir (str): Path to LibriSpeech root (e.g., train-clean-100)
            preprocessor (AudioPreprocessor): Callable to preprocess audio
            ext (str): File extension ('.flac' or '.wav')
            max_files (int): Load only first N files (for debugging)
        """
        self.root_dir = Path(root_dir)
        self.ext = ext
        self.preprocessor = preprocessor

        self.audio_paths = list(self.root_dir.rglob(f"*{ext}"))
        if max_files:
            self.audio_paths = self.audio_paths[:max_files]

        self.transcript_map = self._load_transcripts()

    def _load_transcripts(self):
        transcript_map = {}
        for txt_file in self.root_dir.rglob("*.trans.txt"):
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        uid, transcript = parts
                        transcript_map[uid] = transcript
        return transcript_map

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        uid = path.stem
        
        waveform = self.preprocessor(path) if self.preprocessor else torchaudio.load(path)[0].squeeze(0)

        transcript = self.transcript_map.get(uid, "")

        return {
            "audio": waveform,       # Tensor [T]
            "text": transcript,      # str
            "path": str(path)
        }

