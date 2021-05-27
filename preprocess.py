import csv
from clap.datasets import PairTextSpectrogramTFRecords

import torchaudio

# constants

DATA_DIR = "./data"
NUM_MEL = 80
TSV_FILE_NAME = "subset.tsv"

# helpers


def tsv_to_dict(path):
    with open(path) as fd:
        rd = csv.DictReader(fd, delimiter="\t", quotechar='"')
        return [row for row in rd]


# script

voice_clips = tsv_to_dict(f"{DATA_DIR}/{TSV_FILE_NAME}")

def extract_spectrogram(filename):
    waveform, sample_rate = torchaudio.load(f"{DATA_DIR}/clips/{filename}")

    output = torchaudio.transforms.MelSpectrogram(
        sample_rate, n_mels=NUM_MEL, f_min=0, f_max=8000
    )(waveform)[0]

    return output.t().numpy()

spectrograms = (extract_spectrogram(clip['path']) for clip in voice_clips)
captions = (clip['sentence'] for clip in voice_clips)
save_path = DATA_DIR + '/data.tfrecord'
PairTextSpectrogramTFRecords.write(spectrograms, captions, fname=save_path)
