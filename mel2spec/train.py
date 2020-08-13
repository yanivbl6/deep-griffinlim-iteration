import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.io as scio
import torch
import torch.optim.lr_scheduler as lr_scheduler
from numpy import ndarray
from torch import Tensor, nn
from torch.optim import Adam, SGD



from torch.utils.data import DataLoader



# from torchsummary import summary
from tqdm import tqdm

import soundfile as sf


from dataset import ComplexSpecDataset
from melmodel import melGen

import sys
sys.path.insert(0, '../')
from hparams import hp
from tbwriter import CustomWriter
from utils import AverageMeter, arr2str, draw_spectrogram, print_to_file, calc_using_eval_module, count_parameters
from optimizers.radam import RAdam
from optimizers.novograd import NovoGrad

from time import time

import librosa
import lws

import os

import torch.nn.functional as F


def ms(stime = None):
    if stime is None:
        return int(time() * 1000)
    return (int(time() * 1000) - stime)

def create_mel_filterbank(*args, **kwargs):
    return librosa.filters.mel(*args, **kwargs)

def gen_filter(k):
    K=  torch.ones(1,1,k,1 )
    K.requires_grad= False
    return K

class Trainer:

    def __init__(self, path_state_dict=''):

        ##import pdb; pdb.set_trace()

        self.writer: Optional[CustomWriter] = None

        meltrans = create_mel_filterbank( hp.fs, hp.n_fft, fmin=hp.mel_fmin, fmax=hp.mel_fmax, n_mels=hp.mel_freq)

        self.model = melGen(self.writer, hp.n_freq, meltrans, hp.mel_generator)
        count_parameters(self.model)

        self.module = self.model

        self.lws_processor = lws.lws(hp.n_fft, hp.l_hop, mode='speech', perfectrec=False)

        self.prev_stoi_scores = {}
        self.base_stoi_scores = {}

        if hp.crit == "l1":
            self.criterion = nn.L1Loss(reduction='none')
        elif hp.crit == "l2":
            self.criterion = nn.L2Loss(reduction='none')
        else:
            print("Loss not implemented")
            return None

        self.criterion2 = nn.L1Loss(reduction='none')


        self.f_specs=  {0: [(5, 2),(15,5)],
                        1: [(5, 2)],
                        2: [(3 ,1)],
                        3: [(3 ,1),(5, 2 )],
                        4: [(3 ,1),(5, 2 ), ( 7,3 )  ],
                        5: [(15 ,5)],
                        6: [(3 ,1),(5, 2 ), ( 7,3 ), (15,5), (25,10)],
                        7: [(1 ,1)],
                        8: [(1 ,1), (3 ,1), (5, 2 ),(15 ,5),  ( 7,3 ),  (25,10), (9,4), (20,5), (5,3)   ]
                        }[hp.loss_mode]
                        


        self.filters = [gen_filter(k) for  k,s in self.f_specs]

        if hp.optimizer == "adam":
            self.optimizer = Adam(self.model.parameters(),
                                lr=hp.learning_rate,
                                weight_decay=hp.weight_decay,
                                )
        elif hp.optimizer == "sgd":
            self.optimizer = SGD(self.model.parameters(),
                                lr=hp.learning_rate,
                                weight_decay=hp.weight_decay,
                                )
        elif hp.optimizer == "radam":
            self.optimizer = RAdam(self.model.parameters(),
                                lr=hp.learning_rate,
                                weight_decay=hp.weight_decay,
                                )
        elif hp.optimizer == "novograd":
            self.optimizer = NovoGrad(self.model.parameters(), 
                                    lr=hp.learning_rate, 
                                    weight_decay=hp.weight_decay
                                    )
        elif hp.optimizer == "sm3":
            raise NameError('sm3 not implemented')
        else:
            raise NameError('optimizer not implemented')


        self.__init_device(hp.device)

        if  hp.optimizer == "novograd":
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, 11907*3 ,1e-4)
        else:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, **hp.scheduler)

        self.max_epochs = hp.n_epochs


        self.valid_eval_sample: Dict[str, Any] = dict()

        # len_weight = hp.repeat_train
        # self.loss_weight = torch.tensor(
        #     [1./i for i in range(len_weight, 0, -1)],
        # )
        # self.loss_weight /= self.loss_weight.sum()

        # Load State Dict
        if path_state_dict:
            st_model, st_optim, st_sched = torch.load(path_state_dict, map_location=self.in_device)
            try:
                self.module.load_state_dict(st_model)
                self.optimizer.load_state_dict(st_optim)
                self.scheduler.load_state_dict(st_sched)
            except:
                raise Exception('The model is different from the state dict.')


        path_summary = hp.logdir / 'summary.txt'
        if not path_summary.exists():
            # print_to_file(
            #     path_summary,
            #     summary,
            #     (self.model, hp.dummy_input_size),
            #     dict(device=self.str_device[:4])
            # )
            with path_summary.open('w') as f:
                f.write('\n')
            with (hp.logdir / 'hparams.txt').open('w') as f:
                f.write(repr(hp))

    def __init_device(self, device):
        """

        :type device: Union[int, str, Sequence]
        :type out_device: Union[int, str, Sequence]
        :return:
        """


        # device type: List[int]
        if type(device) == int:
            device = [device]
        elif type(device) == str:
            if device[0] == 'a':
                device = [x for x in range(torch.cuda.device_count())]
            else:
                device = [int(d.replace('cuda:', '')) for d in device.split(",")]
            print("Used devices = %s" % device)
        else:  # sequence of devices
            if type(device[0]) != int:
                device = [int(d.replace('cuda:', '')) for d in device]
        self.num_workers = len(device)
        if len(device) > 1:
            self.model = nn.DataParallel(self.model, device_ids=device)

        self.in_device = torch.device(f'cuda:{device[0]}')
        torch.cuda.set_device(self.in_device)

        self.model.cuda()
        self.criterion.cuda()
        self.criterion2.cuda()
        self.filters = [f.cuda() for f in self.filters]

    def preprocess(self, data: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        # B, F, T, C
        y = data['y']
        y = y.cuda()

        return y

    @torch.no_grad()
    def postprocess(self, output: Tensor, residual: Tensor, Ts: ndarray, idx: int,
                    dataset: ComplexSpecDataset) -> Dict[str, ndarray]:
        dict_one = dict(out=output, res=residual)
        for key in dict_one:
            if dict_one[key] is None:
                continue
            one = dict_one[key][idx, :, :, :Ts[idx]]
            one = one.permute(1, 2, 0).contiguous()  # F, T, 2

            one = one.cpu().numpy().view(dtype=np.complex64)  # F, T, 1
            dict_one[key] = one

        return dict_one

    def calc_loss(self, x: Tensor, y: Tensor, T_ys: Sequence[int], crit) -> Tensor:
        """
        out_blocks: B, depth, C, F, T
        y: B, C, F, T
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            loss_no_red = crit(x, y)

        loss_blocks = torch.zeros(x.shape[1], device=y.device)

        tot =0 
        for T, loss_batch in zip(T_ys, loss_no_red):
            tot += T
            loss_blocks += torch.sum(loss_batch[..., :T])
        loss_blocks = loss_blocks / tot

        if len(loss_blocks) == 1:
            loss = loss_blocks.squeeze()
        else:
            loss = loss_blocks @ self.loss_weight
        return loss

    def calc_loss_smooth(self, _x: Tensor, _y: Tensor, T_ys: Sequence[int], filter, stride: int ,pad: int = 0) -> Tensor:
        """
        out_blocks: B, depth, C, F, T
        y: B, C, F, T
        """

        crit = self.criterion 
        x = F.conv2d(_x, filter, stride = stride)
        y = F.conv2d(_y, filter, stride = stride)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            loss_no_red = crit(x, y)

        loss_blocks = torch.zeros(x.shape[1], device=y.device)

        tot =0 
        for T, loss_batch in zip(T_ys, loss_no_red):
            tot += T
            loss_blocks += torch.sum(loss_batch[..., :T])
        loss_blocks = loss_blocks / tot

        if len(loss_blocks) == 1:
            loss = loss_blocks.squeeze()
        else:
            loss = loss_blocks @ self.loss_weight
        return loss

    def calc_loss_smooth2(self, _x: Tensor, _y: Tensor, T_ys: Sequence[int], kern: int , stride: int ,pad: int = 0) -> Tensor:
        """
        out_blocks: B, depth, C, F, T
        y: B, C, F, T
        """

        crit = self.criterion 

        x = F.max_pool2d(_x, (kern, 1), stride = stride ) 
        y = F.max_pool2d(_y, (kern, 1), stride = stride ) 

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            loss_no_red = crit(x, y)

        loss_blocks = torch.zeros(x.shape[1], device=y.device)

        tot =0 
        for T, loss_batch in zip(T_ys, loss_no_red):
            tot += T
            loss_blocks += torch.sum(loss_batch[..., :T])
        loss_blocks = loss_blocks / tot

        if len(loss_blocks) == 1:
            loss1 = loss_blocks.squeeze()
        else:
            loss1 = loss_blocks @ self.loss_weight

        x = F.max_pool2d(-1*_x, (kern, 1), stride = stride ) 
        y = F.max_pool2d(-1*_y, (kern, 1), stride = stride ) 

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            loss_no_red = crit(x, y)

        loss_blocks = torch.zeros(x.shape[1], device=y.device)

        tot =0 
        for T, loss_batch in zip(T_ys, loss_no_red):
            tot += T
            loss_blocks += torch.sum(loss_batch[..., :T])
        loss_blocks = loss_blocks / tot

        if len(loss_blocks) == 1:
            loss2 = loss_blocks.squeeze()
        else:
            loss2 = loss_blocks @ self.loss_weight

        loss = loss1 + loss2
        return loss





    @torch.no_grad()
    def should_stop(self, loss_valid, epoch):
        if epoch == self.max_epochs - 1:
            return True
        self.scheduler.step(loss_valid)
        # if self.scheduler.t_epoch == 0:  # if it is restarted now
        #     # if self.loss_last_restart < loss_valid:
        #     #     return True
        #     if self.loss_last_restart * hp.threshold_stop < loss_valid:
        #         self.max_epochs = epoch + self.scheduler.restart_period + 1
        #     self.loss_last_restart = loss_valid




    def train(self, loader_train: DataLoader, loader_valid: DataLoader,
              logdir: Path, first_epoch=0):

        os.makedirs(Path(logdir), exist_ok=True)
        self.writer = CustomWriter(str(logdir), group='train', purge_step=first_epoch)

        # Start Training
        step = 0

        loss_valid = self.validate(loader_valid, logdir, 0)
        l2_factor = hp.l2_factor
        
        num_filters = len(self.filters)

        for epoch in range(first_epoch, hp.n_epochs):
            self.writer.add_scalar('meta/lr', self.optimizer.param_groups[0]['lr'], epoch)
            pbar = tqdm(loader_train,
                        desc=f'epoch {epoch:3d}', postfix='[]', dynamic_ncols=True)
            avg_loss1 = AverageMeter(float)
            avg_loss2 = AverageMeter(float)

            avg_loss_tot = AverageMeter(float)
            avg_losses = [AverageMeter(float) for _ in range(num_filters) ]
            losses = [None] * num_filters

            avg_grad_norm = AverageMeter(float)

            for i_iter, data in enumerate(pbar):
                # get data
                ##import pdb; pdb.set_trace()
                y = self.preprocess(data)

                x_mel = self.model.spec_to_mel(y) 

                T_ys = data['T_ys']
                # forward
                x = self.model(x_mel) 

                y_mel = self.model.spec_to_mel(x)  


                step = step + 1

                loss1 = self.calc_loss(x    , y    , T_ys, self.criterion)
                loss2 = self.calc_loss(x_mel, y_mel, T_ys, self.criterion2)
                loss = loss1+ l2_factor*loss2

                # for i,f in enumerate(self.filters):
                #     s = self.f_specs[i][1]
                #     losses[i] = self.calc_loss_smooth(x,y,T_ys,f, s )
                #     loss = loss + losses[i]

                for i,(k,s) in enumerate(self.f_specs):
                    losses[i] = self.calc_loss_smooth2(x,y,T_ys,k, s )
                    loss = loss + losses[i]
            
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                           hp.thr_clip_grad)

                self.optimizer.step()

                # print
                avg_loss1.update(loss1.item(), len(T_ys))
                avg_loss2.update(loss2.item(), len(T_ys))
                avg_loss_tot.update(loss.item(), len(T_ys))

                for j,l in enumerate(losses):
                    avg_losses[j].update(l.item(), len(T_ys))

                
                pbar.set_postfix_str(f'{avg_loss1.get_average():.1e}')
                avg_grad_norm.update(grad_norm)

                if i_iter % 25 == 0:
                    self.writer.add_scalar('loss/loss1_train', avg_loss1.get_average(), epoch*len(loader_train)+ i_iter)
                    self.writer.add_scalar('loss/loss2_train', avg_loss2.get_average(), epoch*len(loader_train)+ i_iter)

                    for j, avg_loss in enumerate(avg_losses):
                        k = self.f_specs[j][0]
                        s = self.f_specs[j][1]
                        self.writer.add_scalar(f'loss/losses_{k}_{s}_train', avg_loss.get_average(), epoch*len(loader_train)+ i_iter)
                    self.writer.add_scalar('loss/loss_total_train', avg_loss_tot.get_average(), epoch*len(loader_train)+ i_iter)

                    self.writer.add_scalar('loss/grad', avg_grad_norm.get_average(), epoch*len(loader_train) +  i_iter)

                    avg_loss1 = AverageMeter(float)
                    avg_loss2 = AverageMeter(float)
                    avg_loss_tot = AverageMeter(float)
                    avg_losses = [AverageMeter(float) for _ in range(num_filters) ]
                    avg_grad_norm = AverageMeter(float)


            # Validation
            # loss_valid = self.validate(loader_valid, logdir, epoch)
            loss_valid = self.validate(loader_valid, logdir, epoch+1)

            # save loss & model
            if epoch % hp.period_save_state == hp.period_save_state - 1:
                torch.save(
                    (self.module.state_dict(),
                     self.optimizer.state_dict(),
                     self.scheduler.state_dict(),
                     ),
                    logdir / f'{epoch+1}.pt'
                )

            # Early stopping
            if self.should_stop(loss_valid, epoch):
                break

        self.writer.close()

    def audio_from_mag_spec(self, mag_spec):
        mag_spec = mag_spec.astype(np.float64)
        spec_lws = self.lws_processor.run_lws(np.transpose(mag_spec))
        magspec_inv = self.lws_processor.istft(spec_lws)[:, np.newaxis, np.newaxis]
        magspec_inv = magspec_inv.astype('float32')
        return magspec_inv


    @torch.no_grad()
    def validate(self, loader: DataLoader, logdir: Path, epoch: int):
        """ Evaluate the performance of the model.

        :param loader: DataLoader to use.
        :param logdir: path of the result files.
        :param epoch:
        """
        self.model.eval()

        num_filters = len(self.filters)
        avg_stoi = AverageMeter(float)
        avg_stoi_norm = AverageMeter(float)
        avg_stoi_base = AverageMeter(float)


        avg_loss1 = AverageMeter(float)
        avg_lozz1 = AverageMeter(float)
        avg_loss2 = AverageMeter(float)
        avg_lozz2 = AverageMeter(float)

        avg_loss_tot = AverageMeter(float)
        avg_losses = [AverageMeter(float) for _ in range(num_filters) ]
        losses = [None] * num_filters

        pbar = tqdm(loader, desc='validate ', postfix='[0]', dynamic_ncols=True)
        num_iters = len(pbar)

        for i_iter, data in enumerate(pbar):

            ##import pdb; pdb.set_trace()
            y = self.preprocess(data)  # B, C, F, T
            x_mel = self.model.spec_to_mel(y) 

            z = self.model.mel_pseudo_inverse(x_mel)

            T_ys = data['T_ys']

            x = self.model(x_mel)  # B, C, F, T
            y_mel = self.model.spec_to_mel(x)     
            z_mel = self.model.spec_to_mel(y)

            loss1 = self.calc_loss(x, y, T_ys, self.criterion)
            lozz1 = self.calc_loss(z, y, T_ys, self.criterion)

            loss2 = self.calc_loss(x_mel, y_mel, T_ys, self.criterion2)
            lozz2 = self.calc_loss(z_mel, x_mel, T_ys, self.criterion2)

            loss = loss1 + loss2*hp.l2_factor

            # for i,f in enumerate(self.filters):
            #     s = self.f_specs[i][1]
            #     losses[i] = self.calc_loss_smooth(x,y,T_ys,f, s )
            #     loss = loss + losses[i]

            for i,(k,s) in enumerate(self.f_specs):
                losses[i] = self.calc_loss_smooth2(x,y,T_ys,k, s )
                loss = loss + losses[i]

            avg_loss1.update(loss1.item(), len(T_ys))
            avg_lozz1.update(lozz1.item(), len(T_ys))
            avg_loss2.update(loss2.item(), len(T_ys))
            avg_lozz2.update(lozz2.item(), len(T_ys))
            avg_loss_tot.update(loss.item(), len(T_ys))

            for j,l in enumerate(losses):
                avg_losses[j].update(l.item(), len(T_ys))

            # print
            pbar.set_postfix_str(f'{avg_loss1.get_average():.1e}')

            ## STOI evaluation with LWS
            for p in range(min(hp.num_stoi// num_iters,len(T_ys))):

                _x = x[p,0,:,:T_ys[p]].cpu()
                _y = y[p,0,:,:T_ys[p]].cpu()
                _z = z[p,0,:,:T_ys[p]].cpu()

                audio_x = self.audio_from_mag_spec(np.abs(_x.numpy()))
                y_wav = data['wav'][p]

                stoi_score= self.calc_stoi(y_wav, audio_x)
                avg_stoi.update(stoi_score)

                if not i_iter in self.prev_stoi_scores:
                    audio_y = self.audio_from_mag_spec(_y.numpy())
                    audio_z = self.audio_from_mag_spec(_z.numpy())

                    self.prev_stoi_scores[i_iter] = self.calc_stoi(y_wav, audio_y)
                    self.base_stoi_scores[i_iter] = self.calc_stoi(y_wav, audio_z)

                avg_stoi_norm.update( stoi_score / self.prev_stoi_scores[i_iter])
                avg_stoi_base.update( stoi_score / self.base_stoi_scores[i_iter])

            # write summary
            ## if i_iter < 4:
            if False: ## stoi is good enough until tests
                x = x[0,0,:,:T_ys[0]].cpu()
                y = y[0,0,:,:T_ys[0]].cpu()
                z = z[0,0,:,:T_ys[0]].cpu()

                ##import pdb; pdb.set_trace()

                if i_iter == 3 and hp.request_drawings:
                    ymin = y[y > 0].min()
                    vmin, vmax = librosa.amplitude_to_db(np.array((ymin, y.max())))
                    kwargs_fig = dict(vmin=vmin, vmax=vmax)
                    fig_x = draw_spectrogram(x, **kwargs_fig)


                    ##self.add_figure(f'{self.group}Audio{idx}/0_Noisy_Spectrum', fig_x, step)
                    self.writer.add_figure(f'Audio{i_iter}/1_DNN_Output', fig_x, epoch)

                    if epoch ==0:
                        fig_y = draw_spectrogram(y, **kwargs_fig)
                        fig_z = draw_spectrogram(z, **kwargs_fig)
                        self.writer.add_figure(f'Audio{i_iter}/0_Pseudo_Inverse', fig_z, epoch)
                        self.writer.add_figure(f'Audio{i_iter}/2_Real_Spectrogram', fig_y, epoch)

                else:
                    audio_x = self.audio_from_mag_spec(np.abs(x.numpy()))


                    x_scale = np.abs(audio_x).max() / 0.5



                    self.writer.add_audio(f'Audio{i_iter}/1_DNN_Output',
                                torch.from_numpy(audio_x / x_scale),
                                epoch,
                                sample_rate=hp.sampling_rate)
                    if epoch ==0:

                        audio_y = self.audio_from_mag_spec(y.numpy())
                        audio_z = self.audio_from_mag_spec(z.numpy())
                        
                        z_scale = np.abs(audio_z).max() / 0.5
                        y_scale = np.abs(audio_y).max() / 0.5

                        self.writer.add_audio(f'Audio{i_iter}/0_Pseudo_Inverse',
                                    torch.from_numpy(audio_z / z_scale),
                                    epoch,
                                    sample_rate=hp.sampling_rate)


                        self.writer.add_audio(f'Audio{i_iter}/2_Real_Spectrogram',
                                    torch.from_numpy(audio_y / y_scale),
                                    epoch,
                                    sample_rate=hp.sampling_rate)


        self.writer.add_scalar(f'valid/loss', avg_loss1.get_average(), epoch)
        self.writer.add_scalar(f'valid/baseline', avg_lozz1.get_average(), epoch)
        self.writer.add_scalar(f'valid/melinv_loss', avg_loss2.get_average(), epoch)
        self.writer.add_scalar(f'valid/melinv_baseline', avg_lozz2.get_average(), epoch)
        self.writer.add_scalar(f'valid/STOI', avg_stoi.get_average(), epoch )
        self.writer.add_scalar(f'valid/STOI_normalized', avg_stoi_norm.get_average(), epoch )
        self.writer.add_scalar(f'valid/STOI_improvement', avg_stoi_base.get_average(), epoch )

        for j, avg_loss in enumerate(avg_losses):
            k = self.f_specs[j][0]
            s = self.f_specs[j][1]
            self.writer.add_scalar(f'valid/losses_{k}_{s}', avg_loss.get_average(), epoch)
        self.writer.add_scalar('valid/loss_total', avg_loss_tot.get_average(), epoch)

        self.model.train()

        return avg_loss1.get_average()

    def calc_stoi(self, y_wav, audio):

        audio_len = min(y_wav.shape[0], audio.shape[0]  )
        measure = calc_using_eval_module(y_wav[:audio_len], audio[:audio_len,0,0])
        return measure['STOI']

    @torch.no_grad()
    def test(self, loader: DataLoader, logdir: Path):
        """ Evaluate the performance of the model.

        :param loader: DataLoader to use.
        :param logdir: path of the result files.
        :param epoch:
        """
        self.model.eval()

        os.makedirs(Path(logdir), exist_ok=True)
        self.writer = CustomWriter(str(logdir), group='test')

        ##import pdb; pdb.set_trace()
        num_filters = len(self.filters)

        avg_loss1 = AverageMeter(float)
        avg_lozz1 = AverageMeter(float)
        avg_loss2 = AverageMeter(float)
        avg_lozz2 = AverageMeter(float)

        avg_loss_tot = AverageMeter(float)
        avg_losses = [AverageMeter(float) for _ in range(num_filters) ]
        losses = [None] * num_filters

        cnt = 0
        for i_iter, data in enumerate(loader):

            ##import pdb; pdb.set_trace()
            y = self.preprocess(data)  # B, C, F, T
            x_mel = self.model.spec_to_mel(y) 

            z = self.model.mel_pseudo_inverse(x_mel)

            T_ys = data['T_ys']
            x = self.model(x_mel)  # B, C, F, T
            y_mel = self.model.spec_to_mel(x)     
            z_mel = self.model.spec_to_mel(y)

            loss1 = self.calc_loss(x, y, T_ys, self.criterion)
            lozz1 = self.calc_loss(z, y, T_ys, self.criterion)

            loss2 = self.calc_loss(x_mel, y_mel, T_ys, self.criterion2)
            lozz2 = self.calc_loss(z_mel, x_mel, T_ys, self.criterion2)

            loss = loss1 + loss2*hp.l2_factor

            # for i,f in enumerate(self.filters):
            #     s = self.f_specs[i][1]
            #     losses[i] = self.calc_loss_smooth(x,y,T_ys,f, s )
            #     loss = loss + losses[i]

            for i,(k,s) in enumerate(self.f_specs):
                losses[i] = self.calc_loss_smooth2(x,y,T_ys,k, s )
                loss = loss + losses[i]

            avg_loss1.update(loss1.item(), len(T_ys))
            avg_lozz1.update(lozz1.item(), len(T_ys))
            avg_loss2.update(loss2.item(), len(T_ys))
            avg_lozz2.update(lozz2.item(), len(T_ys))
            avg_loss_tot.update(loss.item(), len(T_ys))

            for j,l in enumerate(losses):
                avg_losses[j].update(l.item(), len(T_ys))

            # print
            ##pbar.set_postfix_str(f'{avg_loss1.get_average():.1e}')

            # write summary

            pbar = tqdm(range(len(T_ys)), desc='validate_bath', postfix='[0]', dynamic_ncols=True)

            for p in pbar:
                _x = x[p,0,:,:T_ys[p]].cpu()
                _y = y[p,0,:,:T_ys[p]].cpu()
                _z = z[p,0,:,:T_ys[p]].cpu()
                y_wav = data['wav'][p]

                ymin = _y[_y > 0].min()
                vmin, vmax = librosa.amplitude_to_db(np.array((ymin, _y.max())))
                kwargs_fig = dict(vmin=vmin, vmax=vmax)


                if hp.request_drawings:
                    fig_x = draw_spectrogram(_x, **kwargs_fig)
                    self.writer.add_figure(f'Audio/1_DNN_Output', fig_x, cnt)
                    fig_y = draw_spectrogram(_y, **kwargs_fig)
                    fig_z = draw_spectrogram(_z, **kwargs_fig)
                    self.writer.add_figure(f'Audio/0_Pseudo_Inverse', fig_z, cnt)
                    self.writer.add_figure(f'Audio/2_Real_Spectrogram', fig_y, cnt)

                audio_x = self.audio_from_mag_spec(np.abs(_x.numpy()))
                x_scale = np.abs(audio_x).max() / 0.5

                self.writer.add_audio(f'LWS/1_DNN_Output',
                            torch.from_numpy(audio_x / x_scale),
                            cnt,
                            sample_rate=hp.sampling_rate)

                audio_y = self.audio_from_mag_spec(_y.numpy())
                audio_z = self.audio_from_mag_spec(_z.numpy())
                
                z_scale = np.abs(audio_z).max() / 0.5
                y_scale = np.abs(audio_y).max() / 0.5

                self.writer.add_audio(f'LWS/0_Pseudo_Inverse',
                            torch.from_numpy(audio_z / z_scale),
                            cnt,
                            sample_rate=hp.sampling_rate)


                self.writer.add_audio(f'LWS/2_Real_Spectrogram',
                            torch.from_numpy(audio_y / y_scale),
                            cnt,
                            sample_rate=hp.sampling_rate)

                ##import pdb; pdb.set_trace()

                stoi_scores = {'0_Pseudo_Inverse'       : self.calc_stoi(y_wav, audio_z),
                               '1_DNN_Output'           : self.calc_stoi(y_wav, audio_x),
                               '2_Real_Spectrogram'     : self.calc_stoi(y_wav, audio_y)}

                self.writer.add_scalars(f'LWS/STOI', stoi_scores, cnt )
                # self.writer.add_scalar(f'STOI/0_Pseudo_Inverse_LWS', self.calc_stoi(y_wav, audio_z) , cnt)
                # self.writer.add_scalar(f'STOI/1_DNN_Output_LWS', self.calc_stoi(y_wav, audio_x) , cnt)
                # self.writer.add_scalar(f'STOI/2_Real_Spectrogram_LWS', self.calc_stoi(y_wav, audio_y) , cnt)
                cnt = cnt + 1

        # self.writer.add_scalar(f'valid/loss', avg_loss1.get_average(), epoch)
        # self.writer.add_scalar(f'valid/baseline', avg_lozz1.get_average(), epoch)
        # self.writer.add_scalar(f'valid/melinv_loss', avg_loss2.get_average(), epoch)
        # self.writer.add_scalar(f'valid/melinv_baseline', avg_lozz2.get_average(), epoch)

        # for j, avg_loss in enumerate(avg_losses):
        #     k = self.f_specs[j][0]
        #     s = self.f_specs[j][1]
        #     self.writer.add_scalar(f'valid/losses_{k}_{s}', avg_loss.get_average(), epoch)
        # self.writer.add_scalar('valid/loss_total', avg_loss_tot.get_average(), epoch)

        self.model.train()

        return 

    @torch.no_grad()
    def inspect(self, loader: DataLoader, logdir: Path):
        """ Evaluate the performance of the model.

        :param loader: DataLoader to use.
        :param logdir: path of the result files.
        :param epoch:
        """
        self.model.eval()

        os.makedirs(Path(logdir), exist_ok=True)
        self.writer = CustomWriter(str(logdir), group='test')

        ##import pdb; pdb.set_trace()
        num_filters = len(self.filters)

        avg_loss1 = AverageMeter(float)
        avg_lozz1 = AverageMeter(float)
        avg_loss2 = AverageMeter(float)
        avg_lozz2 = AverageMeter(float)

        avg_loss_tot = AverageMeter(float)
        avg_losses = [AverageMeter(float) for _ in range(num_filters) ]
        avg_losses_base = [AverageMeter(float) for _ in range(num_filters) ]
        losses = [None] * num_filters
        losses_base = [None] * num_filters

        cnt = 0

        pbar = tqdm(enumerate(loader), desc='loss inspection', dynamic_ncols=True)

        for i_iter, data in pbar:

            ##import pdb; pdb.set_trace()
            y = self.preprocess(data)  # B, C, F, T
            x_mel = self.model.spec_to_mel(y) 

            z = self.model.mel_pseudo_inverse(x_mel)

            T_ys = data['T_ys']
            x = self.model(x_mel)  # B, C, F, T
            y_mel = self.model.spec_to_mel(x)     
            z_mel = self.model.spec_to_mel(y)

            loss1 = self.calc_loss(x, y, T_ys, self.criterion)
            lozz1 = self.calc_loss(z, y, T_ys, self.criterion)

            loss2 = self.calc_loss(x_mel, y_mel, T_ys, self.criterion2)
            lozz2 = self.calc_loss(z_mel, x_mel, T_ys, self.criterion2)

            loss = loss1 + loss2*hp.l2_factor

            # for i,f in enumerate(self.filters):
            #     s = self.f_specs[i][1]
            #     losses[i] = self.calc_loss_smooth(x,y,T_ys,f, s )
            #     loss = loss + losses[i]

            for i,(k,s) in enumerate(self.f_specs):
                losses[i] = self.calc_loss_smooth2(x,y,T_ys,k, s )
                losses_base[i] = self.calc_loss_smooth2(y,y,T_ys,k, s )

                loss = loss + losses[i]
            avg_loss1.update(loss1.item(), len(T_ys))
            avg_lozz1.update(lozz1.item(), len(T_ys))
            avg_loss2.update(loss2.item(), len(T_ys))
            avg_lozz2.update(lozz2.item(), len(T_ys))
            avg_loss_tot.update(loss.item(), len(T_ys))

            for j,l in enumerate(losses):
                avg_losses[j].update(l.item(), len(T_ys))
                
            for j,l in enumerate(losses_base):
                avg_losses_base[j].update(l.item(), len(T_ys))                
            # print
            ##pbar.set_postfix_str(f'{avg_loss1.get_average():.1e}')

            # write summary

            if 0:
                for p in range(len(T_ys)):
                    _x = x[p,0,:,:T_ys[p]].cpu()
                    _y = y[p,0,:,:T_ys[p]].cpu()
                    _z = z[p,0,:,:T_ys[p]].cpu()
                    y_wav = data['wav'][p]

                    ymin = _y[_y > 0].min()
                    vmin, vmax = librosa.amplitude_to_db(np.array((ymin, _y.max())))
                    kwargs_fig = dict(vmin=vmin, vmax=vmax)


                    if hp.request_drawings:
                        fig_x = draw_spectrogram(_x, **kwargs_fig)
                        self.writer.add_figure(f'Audio/1_DNN_Output', fig_x, cnt)
                        fig_y = draw_spectrogram(_y, **kwargs_fig)
                        fig_z = draw_spectrogram(_z, **kwargs_fig)
                        self.writer.add_figure(f'Audio/0_Pseudo_Inverse', fig_z, cnt)
                        self.writer.add_figure(f'Audio/2_Real_Spectrogram', fig_y, cnt)

                    audio_x = self.audio_from_mag_spec(np.abs(_x.numpy()))
                    x_scale = np.abs(audio_x).max() / 0.5

                    self.writer.add_audio(f'LWS/1_DNN_Output',
                                torch.from_numpy(audio_x / x_scale),
                                cnt,
                                sample_rate=hp.sampling_rate)

                    audio_y = self.audio_from_mag_spec(_y.numpy())
                    audio_z = self.audio_from_mag_spec(_z.numpy())
                    
                    z_scale = np.abs(audio_z).max() / 0.5
                    y_scale = np.abs(audio_y).max() / 0.5

                    self.writer.add_audio(f'LWS/0_Pseudo_Inverse',
                                torch.from_numpy(audio_z / z_scale),
                                cnt,
                                sample_rate=hp.sampling_rate)


                    self.writer.add_audio(f'LWS/2_Real_Spectrogram',
                                torch.from_numpy(audio_y / y_scale),
                                cnt,
                                sample_rate=hp.sampling_rate)

                    ##import pdb; pdb.set_trace()

                    stoi_scores = {'0_Pseudo_Inverse'       : self.calc_stoi(y_wav, audio_z),
                                '1_DNN_Output'           : self.calc_stoi(y_wav, audio_x),
                                '2_Real_Spectrogram'     : self.calc_stoi(y_wav, audio_y)}

                    self.writer.add_scalars(f'LWS/STOI', stoi_scores, cnt )
                    # self.writer.add_scalar(f'STOI/0_Pseudo_Inverse_LWS', self.calc_stoi(y_wav, audio_z) , cnt)
                    # self.writer.add_scalar(f'STOI/1_DNN_Output_LWS', self.calc_stoi(y_wav, audio_x) , cnt)
                    # self.writer.add_scalar(f'STOI/2_Real_Spectrogram_LWS', self.calc_stoi(y_wav, audio_y) , cnt)
                    cnt = cnt + 1

        for j, avg_loss in enumerate(avg_losses):
            k = self.f_specs[j][0]
            s = self.f_specs[j][1]
            self.writer.add_scalar(f'inspect/losses_breakdown', avg_loss.get_average(), j)

        for j, avg_loss in enumerate(avg_losses_base):
            k = self.f_specs[j][0]
            s = self.f_specs[j][1]
            self.writer.add_scalar(f'inspect/losses_base_breakdown', avg_loss.get_average(), j)

        for j, avg_loss in enumerate(avg_losses):
            avg_loss2 = avg_losses_base[j]
            k = self.f_specs[j][0]
            s = self.f_specs[j][1]
            self.writer.add_scalar(f'inspect/losses_normalized_breakdown', avg_loss2.get_average() / avg_loss.get_average() , j)


        # self.writer.add_scalar(f'valid/loss', avg_loss1.get_average(), epoch)
        # self.writer.add_scalar(f'valid/baseline', avg_lozz1.get_average(), epoch)
        # self.writer.add_scalar(f'valid/melinv_loss', avg_loss2.get_average(), epoch)
        # self.writer.add_scalar(f'valid/melinv_baseline', avg_lozz2.get_average(), epoch)

        # for j, avg_loss in enumerate(avg_losses):
        #     k = self.f_specs[j][0]
        #     s = self.f_specs[j][1]
        #     self.writer.add_scalar(f'valid/losses_{k}_{s}', avg_loss.get_average(), epoch)
        # self.writer.add_scalar('valid/loss_total', avg_loss_tot.get_average(), epoch)

        self.model.train()

        return 


    @torch.no_grad()
    def infer(self, loader: DataLoader, logdir: Path):
        """ Evaluate the performance of the model.

        :param loader: DataLoader to use.
        :param logdir: path of the result files.
        :param epoch:
        """
        def save_feature(num_snr, i_speech: int, s_path_speech: str, speech: ndarray, mag_mel2spec) -> tuple:
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
                        speech=speech,
                        spec_clean=spec_clean,
                        mag_clean=mag_mel2spec,
                        path_speech=s_path_speech,
                        length=len(speech),
                        )
                )
            return list_snr_db, list_dict


        self.model.eval()

        os.makedirs(Path(logdir), exist_ok=True)

        ##import pdb; pdb.set_trace()
        cnt = 0

        pbar = tqdm(loader, desc='mel2inference', postfix='[0]', dynamic_ncols=True)

        form= '{:05d}_mel2spec_{:+.2f}dB.npz' 
        num_snr = hp.num_snr
        for i_iter, data in enumerate(pbar):

            ##import pdb; pdb.set_trace()
            y = self.preprocess(data)  # B, C, F, T
            x_mel = self.model.spec_to_mel(y) 

            T_ys = data['T_ys']
            x = self.model(x_mel)  # B, C, F, T

            for p in range(len(T_ys)):
                _x = x[p,0,:,:T_ys[p]].unsqueeze(2).cpu().numpy()
                ##import pdb; pdb.set_trace()
                speech = data['wav'][p].numpy()

                list_snr_db, list_dict = save_feature(num_snr, cnt, data['path_speech'][p] , speech, _x)
                cnt = cnt + 1
                for snr_db, dict_result in zip(list_snr_db, list_dict):
                    np.savez(logdir / form.format(cnt, snr_db),
                            **dict_result,
                            )
        self.model.train()

        return 
