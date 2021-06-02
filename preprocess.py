import csv
from clap.datasets import PairTextSpectrogramTFRecords
from omegaconf import DictConfig, OmegaConf
import hydra

import torchaudio


# helpers


def tsv_to_dict(path):
    with open(path) as fd:
        rd = csv.DictReader(fd, delimiter="\t", quotechar='"')
        return [row for row in rd]


# script

@hydra.main(config_path="configs")
def preprocess(cfg: DictConfig) -> None:
    data_folder = hydra.utils.get_original_cwd() + "/" + cfg.preprocessing.dataset.data_folder

    voice_clips = tsv_to_dict(f"{data_folder}/{cfg.preprocessing.dataset.tsv_filename}")

    def extract_spectrogram(filename):
        waveform, sample_rate = torchaudio.load(f"{data_folder}/clips/{filename}")

        output = torchaudio.transforms.MelSpectrogram(
            sample_rate, n_mels=cfg.preprocessing.dataset.mel_bins, f_min=0, f_max=8000
        )(waveform)[0]

        return output.t().numpy()

    spectrograms = (extract_spectrogram(clip['path']) for clip in voice_clips)
    captions = (clip['sentence'] for clip in voice_clips)
    save_path = data_folder + '/data.tfrecord'
    PairTextSpectrogramTFRecords.write(spectrograms, captions, fname=save_path)

if __name__ == "__main__":
    preprocess()