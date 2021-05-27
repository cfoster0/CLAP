import glob
import torch
import tensorflow as tf
from pathlib import Path

from itertools import cycle, islice, chain
from einops import rearrange

import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, ConcatDataset, IterableDataset


class PairTextSpectrogramTFRecords(object):
    def __init__(self, local_or_gcs_path, batch_size, prefetch_size=0, mel_bins=80, max_audio_len=2048, max_text_len=256):
        self.mel_bins = mel_bins
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len

        files = tf.data.TFRecordDataset.list_files(local_or_gcs_path + '/*.tfrecord', shuffle=False)
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(self.deserialize_tf_record)
        dataset = dataset.padded_batch(batch_size, padded_shapes={
            'audio': (max_audio_len, mel_bins),
            'text': (max_text_len),
        })
        dataset = dataset.prefetch(prefetch_size)
        dataset = dataset.as_numpy_iterator()
        self.dataset = dataset

    def files(self):
        return self.files
    
    def __iter__(self):
        return self.dataset

    def deserialize_tf_record(self, record):
        tfrecord_format = {
            'spectrogram': tf.io.FixedLenSequenceFeature((self.mel_bins,), dtype=tf.float32, allow_missing=True),
            'text': tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
        }

        features_tensor = tf.io.parse_single_example(record, tfrecord_format)
        return features_tensor

    @staticmethod
    def write(spectrograms, captions, fname="data.tfrecord"):
        tfrecord_writer = tf.io.TFRecordWriter(fname)
        for (spectrogram, caption) in zip(spectrograms, captions):
            example = tf.train.Example(features=tf.train.Features(feature={
                'audio': tf.train.Feature(float_list=tf.train.FloatList(value=spectrogram.flatten())),
                'text': tf.train.Feature(int64_list=tf.train.Int64List(value=[*caption.encode('utf-8')])),
            }))
            tfrecord_writer.write(example.SerializeToString())

        tfrecord_writer.close()
    

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
    max_audio_len = 2048
    max_text_len = 256

    padded_batch = []
    for audio, audio_mask, text, text_mask in batch:
        audio_len = audio.shape[0]
        text_len = text.shape[0]
        audio_pad_len = max_audio_len - audio_len
        if audio_pad_len < 0:
            raise ValueError("Audio clip too long")
        text_pad_len = max_text_len - text_len
        if text_pad_len < 0:
            raise ValueError("Text caption too long")

        if audio_pad_len > 0:
            audio = F.pad(audio, (0, 0, audio_pad_len, 0), value=0.0)
            audio_mask = F.pad(audio_mask, (audio_pad_len, 0), value=False)

        if text_pad_len > 0:
            text = F.pad(text, (text_pad_len, 0), value=0.0)
            text_mask = F.pad(text_mask, (text_pad_len, 0), value=False)

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
