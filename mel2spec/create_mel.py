import multiprocessing as mp
import os
from argparse import ArgumentParser

import librosa
import numpy as np
import soundfile as sf
from numpy import ndarray
from tqdm import tqdm
import torch

import sys
from pathlib import Path

from torch.utils.data import DataLoader

from hparams import hp

sys.path.insert(0, '../')
sys.path.insert(0, '../tacotron2')
from tacotron2.layers import TacotronSTFT

def get_mel(sftf, audio):
##    import pdb; pdb.set_trace()

##    audio_norm = audio / hp.max_wav_value
    audio_norm = audio.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    return melspec

def do_sftf(sftf, speech):
    torch_speech = torch.from_numpy(speech).float()
    stft_speech = get_mel(stft, torch_speech)
    stft_speech = stft_speech.numpy()
    return stft_speech

def save_feature(stft, i_speech: int, s_path_speech: str, speech: ndarray) -> tuple:
    spec_clean = np.ascontiguousarray(librosa.stft(speech, **hp.kwargs_stft))
    mag_clean = np.ascontiguousarray(np.abs(spec_clean)[..., np.newaxis])
    
    stft_speech = do_sftf(stft, speech)
    mel_spec_clean = np.ascontiguousarray(stft_speech)
    mel_mag_clean = np.ascontiguousarray(np.abs(mel_spec_clean)[..., np.newaxis])

    mdict =  dict(mag_clean=mag_clean,
                 mel_mag_clean = mel_mag_clean,
                 path_speech=s_path_speech,
                 length=len(speech),
                 )
        
    return mdict

def createLin(loader: DataLoader, logdir: Path, num_snr:int ):
    """ Evaluate the performance of the model.

    :param loader: DataLoader to use.
    :param logdir: path of the result files.
    :param epoch:
    """

    def save_feature(num_snr, i_speech: int, s_path_speech: str, speech: ndarray) -> tuple:
        spec_clean = np.ascontiguousarray(librosa.stft(speech, **hp.kwargs_stft))
        mag_clean = np.ascontiguousarray(np.abs(spec_clean)[..., np.newaxis])

        signal_power = np.mean(np.abs(speech)**2)
        list_dict = []
        list_snr_db = []
        for _ in enumerate(range(num_snr)):
            snr_db = -6*np.random.rand()
            list_snr_db.append(snr_db)
            snr = librosa.db_to_power(snr_db)
            noise_power = signal_power / snr
            noisy = speech + np.sqrt(noise_power) * np.random.randn(len(speech))
            spec_noisy = librosa.stft(noisy, **hp.kwargs_stft)
            spec_noisy = np.ascontiguousarray(spec_noisy)

            list_dict.append(
                dict(spec_noisy=spec_noisy,
                    spec_clean=spec_clean,
                    mag_clean=mag_clean,
                    path_speech=s_path_speech,
                    length=len(speech),
                    )
            )
        return list_snr_db, list_dict

    os.makedirs(Path(logdir), exist_ok=True)

    cnt = 0

    pbar = tqdm(loader, desc='mel2create', postfix='[0]', dynamic_ncols=True)

    form= '{:05d}_mel2spec_{:+.2f}dB.npz' 
    for data in pbar:
        paths = data['path_speech']

        for p in range(len(paths)):
            path_speech= paths[p]

            speech = sf.read(str(path_speech))[0].astype(np.float32)

            list_snr_db, list_dict = save_feature(num_snr, cnt, str(path_speech), speech)
            cnt = cnt + 1
            for snr_db, dict_result in zip(list_snr_db, list_dict):
                np.savez(logdir / form.format(cnt, snr_db),
                        **dict_result,
                        )
    return 


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('kind_data', choices=('TRAIN', 'TEST'))
    args = hp.parse_argument(parser, print_argument=False)
    args.kind_data = args.kind_data.lower()

    path_speech_folder = hp.dict_path['wav_path']

    flist_speech = (list(path_speech_folder.glob('**/*.WAV')) +
                    list(path_speech_folder.glob('**/*.wav')))

    print("building module")
    stft = TacotronSTFT(filter_length=hp.filter_length,
                                hop_length=hp.l_hop,
                                win_length=hp.win_length,
                                n_mel_channels=hp.mel_freq,
                                sampling_rate=hp.sampling_rate,
                                mel_fmin=hp.mel_fmin, mel_fmax=hp.mel_fmax)
    print("finished building module")


    if hp.n_data > 0:
        flist_speech = flist_speech[:hp.n_data]

    path_mel = hp.dict_path['mel_train']
    path_mel2 = hp.dict_path['mel_valid']
    path_mel3 = hp.dict_path['mel_test']


    os.makedirs(path_mel, exist_ok=True)
    os.makedirs(path_mel2, exist_ok=True)
    os.makedirs(path_mel3, exist_ok=True)

    pool = mp.Pool(
        processes=mp.cpu_count()//2 - 1,
        initializer=lambda: np.random.seed(os.getpid()),
    )
    results = []

    form="{:05d}_mel%d.npz" % hp.mel_freq

    validation_every = int(100*hp.train_ratio)

    pbar = tqdm(flist_speech, dynamic_ncols=True)
    for i_speech, path_speech in enumerate(pbar):
        speech = sf.read(str(path_speech))[0].astype(np.float32)
        mdict = save_feature(stft, i_speech, str(path_speech), speech)
        
        if i_speech % 1000 == 0:
            np.savez(path_mel3 / form.format(i_speech),
                        **mdict,
                        )
        elif i_speech % 100 <= validation_every:
            np.savez(path_mel / form.format(i_speech),
                        **mdict,
                        )
        else:
            np.savez(path_mel2 / form.format(i_speech),
                        **mdict,
                        )
