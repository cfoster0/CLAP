import glob
import torch
from pathlib import Path

import lm_dataformat as lmd
from itertools import cycle, islice, chain
from einops import rearrange

import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, ConcatDataset, IterableDataset


class CaptionedAudioMetadataset(IterableDataset):
    def __init__(self, path_pairs, lazy=False):
        self.datasets = [
            CaptionedAudioDataset(captions_path, spectrograms_path, lazy=lazy)
            for (captions_path, spectrograms_path) in path_pairs
        ]

    def __iter__(self):
        def roundrobin(datasets):
            num_active = len(datasets)
            nexts = cycle(iter(it).__next__ for it in datasets)
            while num_active:
                try:
                    for next in nexts:
                        yield next()
                except StopIteration:
                    # Remove the iterator we just exhausted from the cycle.
                    num_active -= 1
                    nexts = cycle(islice(nexts, num_active))

        iterator = roundrobin(self.datasets)

        return iterator


class CaptionedAudioDataset(IterableDataset):
    def __init__(self, captions_path, spectrograms_path, lazy=False):
        self.lazy = lazy
        if self.lazy:
            # Warning: The lazy path does not check whether the cpation metadata
            # links it to the spectrogram. It assumes that the specrogram data,
            # read from the files from the path in sorted order, loaded in as
            # tensors, follows the exact same ordering as the LMD-encoded captions.
            self.captions = lmd.Reader(captions_path).stream_data(get_meta=False)
            self.spectrograms = SpectrogramLazyDataset(spectrograms_path)
        else:
            self.captions = lmd.Reader(captions_path).stream_data(get_meta=True)
            self.spectrograms = SpectrogramDataset(spectrograms_path)

    def __iter__(self):
        if self.lazy:
            iterator = (
                (tokenize(text), spectrogram)
                for ((text, _), spectrogram) in zip(self.captions, self.spectrograms)
            )
        else:
            iterator = (
                (tokenize(text), self.spectrograms[meta["index"]])
                for (text, meta) in self.captions
            )
        return iterator


class SpectrogramDataset(Dataset):
    def __init__(self, path):
        self.shard_paths = sorted(glob.glob(f"{path}/*.pt"))
        self.data = ConcatDataset(
            [SpectrogramDatasetShard(shard_path) for shard_path in self.shard_paths]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SpectrogramLazyDataset(IterableDataset):
    def __init__(self, path):
        self.shard_paths = sorted(glob.glob(f"{path}/*.pt"))

    def __iter__(self):
        def lazy_shard_loader():
            for shard_path in self.shard_paths:
                self.shard_data = SpectrogramDatasetShard(shard_path)
                for example in self.shard_data:
                    yield example

        return lazy_shard_loader()


class SpectrogramDatasetShard(Dataset):
    def __init__(self, path):
        self.dataset_shard = TensorDataset(torch.load(path))

    def __len__(self):
        # Layout is [examples, frames, channels]
        return len(self.dataset_shard)

    def __getitem__(self, idx):
        return self.dataset_shard[idx]


class PairTextSpectrogramDataset(Dataset):
    def __init__(self, folder, max_audio_len=2048, max_text_len=256):
        self.paths = [path for path in Path(folder).glob("*.pt")]
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        max_audio_len, max_text_len = self.max_audio_len, self.max_text_len

        path = self.paths[idx]
        data = torch.load(path)

        audio, text = data["audio"], data["text"]
        audio = audio[:max_audio_len]
        text = text[:max_text_len]

        audio_mask = torch.ones(audio.shape[:-1]).bool()
        text_mask = torch.ones_like(text).bool()

        return audio, audio_mask, text, text_mask


def pair_text_spectrogram_dataset_collate_fn(batch):
    audios = [el[0] for el in batch]
    texts = [el[2] for el in batch]
    #max_audio_len = max([audio.shape[0] for audio in audios]) # Should probably pad to a fixed length
    max_audio_len = 2048
    #max_text_len = max([text.shape[0] for text in texts]) # Should probably pad to a fixed length
    max_text_len = 256

    padded_batch = []
    for audio, audio_mask, text, text_mask in batch:
        audio_len = audio.shape[0]
        text_len = text.shape[0]
        audio_pad_len = max_audio_len - audio_len
        text_pad_len = max_text_len - text_len

        if audio_pad_len > 0:
            audio = F.pad(audio, (0, 0, audio_pad_len, 0), value=0.0)
            audio_mask = F.pad(audio_mask, (audio_pad_len, 0), value=False)

        if text_pad_len > 0:
            text = F.pad(text, (text_pad_len, 0), value = 0.)
            text_mask = F.pad(text_mask, (text_pad_len, 0), value = False)
        
        # Add trailing dimension of 1, since mono audio
        audio = rearrange(audio, "t c -> t c ()")

        padded_batch.append((audio, audio_mask, text, text_mask))

    output = tuple(map(lambda t: torch.stack(t).numpy(), zip(*padded_batch)))
    return output


def tokenize(text, pad_to=256):
    # Padding token is 0, the null byte
    tokens = torch.zeros(pad_to, dtype=torch.uint8)
    # Truncate to context window size on the right if need be
    for i, byte in enumerate(text.encode("utf-8")):
        if i < pad_to:
            tokens[i] = int(byte)
        else:
            break
    return torch.tensor(tokens)


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))
