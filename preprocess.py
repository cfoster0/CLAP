import csv
from clap.datasets import tokenize

import torch
import torchaudio

# constants

MAX_TOKEN_LENGTH = 256
DATA_DIR = './data'
NUM_MEL = 80
TSV_FILE_NAME = 'subset.tsv'

# helpers

def tsv_to_dict(path):
    with open(path) as fd:
        rd = csv.DictReader(fd, delimiter = "\t", quotechar = '"')
        return [row for row in rd]

# script

voice_clips = tsv_to_dict(f'{DATA_DIR}/{TSV_FILE_NAME}')

for clip in voice_clips:
    filename = clip['path']
    text = clip['sentence']

    waveform, sample_rate = torchaudio.load(f"{DATA_DIR}/clips/{filename}", normalization = True)

    output = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels = NUM_MEL)(waveform)[0]
    tokenized = torch.tensor([int(byte) for i, byte in enumerate(text.encode('utf-8'))], dtype = torch.uint8)

    save_path = f"{DATA_DIR}/{filename}.pt"

    torch.save({
        'audio': output.t(),
        'text': tokenized
    }, save_path)
