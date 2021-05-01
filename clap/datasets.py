import glob
import torch
import lm_dataformat as lmd
from itertools import cycle, islice, chain
from torch.utils.data import Dataset, TensorDataset, ConcatDataset, IterableDataset

class CaptionedAudioMetadataset(IterableDataset):
    def __init__(self, path_pairs):
        self.datasets = [CaptionedAudioDataset(captions_path, spectrograms_path) for (captions_path, spectrograms_path) in path_pairs)]

    def __iter__(self):
        iterator = roundrobin(self.datasets)

class CaptionedAudioDataset(IterableDataset):
    def __init__(self, captions_path, spectrograms_path, lazy=False):
        self.lazy = lazy
        if self.lazy:
            # Warning: The lazy path does not check whether the cpation metadata
            # links it to the spectrogram. It assumes that the specrogram data,
            # read from the files from the path in sorted order, loaded in as
            # tensors, follows the exact same ordering as the LMD-encoded captions.
            self.captions = lmd.Reader(captions_path).stream_data(get_meta=False)
            self.spectrograms = SpectrogramLazyDataset(spectrogram_path)
        else:
            self.captions = lmd.Reader(captions_path).stream_data(get_meta=True)
            self.spectrograms = SpectrogramDataset(spectrogram_path)

    def __iter__(self):
        if self.lazy:
            iterator = ((tokenize(text), spectrogram), for ((text, _), spectrogram) in zip(self.captions, self.spectrograms))
        else:
            iterator = ((tokenize(text), self.spectrograms[meta['index']]), for (text, meta) in self.captions)
        return iterator

class SpectrogramDataset(Dataset):
    def __init__(self, path):
        self.shard_paths = sorted(glob.glob(f"{path}/*.pt"))
        self.data = ConcatDataset([SpectrogramDatasetShard(shard_path) for shard_path in self.shard_paths])

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
        # Layout is [examples, channels, frames]
        return len(self.dataset_shard)

    def __getitem__(self, idx):
        return self.dataset_shard[idx]

def tokenize(text, pad_to=256):
    # Padding token is 0, the null byte
    tokens = torch.zeros(pad_to, torch.uint8)
    # Truncate to context window size on the right if need be
    [tokens[i] = int(byte) for (i, byte) in enumerate(text.encode('utf-8')) if i < pad_to]
    return tokens


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