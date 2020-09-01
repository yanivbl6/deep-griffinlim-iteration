import torch
import torch.nn as nn

from .istft import InverseSTFT

from lib.modules.optimizations import *

import math

from collections import OrderedDict

def str2act(txt, param= None):
    return {"sigmoid": nn.Sigmoid(), "relu": nn.ReLU(), "none": nn.Sequential() , "lrelu": nn.LeakyReLU(param), "selu": nn.SELU() }[txt.lower()]

class ConvGLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(7, 7), padding=None, batchnorm=False, act="sigmoid", stride = None):
        super().__init__()
        if not padding:
            padding = (kernel_size[0] // 2 , kernel_size[1] // 2 )
        if stride is None:
            self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size, padding=padding)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size, padding=padding, stride= stride)
        self.weight = self.conv.weight
        self.bias = self.conv.bias

        if batchnorm:
            self.conv = nn.Sequential(
                self.conv,
                nn.BatchNorm2d(out_ch * 2)
            )
        self.sigmoid = str2act(act)
        
    def forward(self, x):
        x = self.conv(x)
        ch = x.shape[1]
        x = x[:, :ch//2, ...] * self.sigmoid(x[:, ch//2:, ...])
        return x


class DeGLI_DEQ(nn.Module):
    def __init__(self, writer, wnorm: bool, num_branches: int, base_channels: int, ratio2head: int, fuse_method: str,
                 droprate: float, final_multiplier: int, pretrain_steps:int, f_thres:int, b_thres:int, num_layers:int,
                 ch_hidden: int, k1:int, k2:int, p2:int):

        super().__init__()

        ## parameters ---------------------------
        self.writer = writer
        self.wnorm = wnorm
        self.num_branches = num_branches
        self.num_blocks = [1] * self.num_branches

class DeGLI_DNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        k_x1, k_y1, k_x2 ,k_y2 = self.parse(**config)

        ch_hidden = self.ch_hidden

        self.convglu_first = ConvGLU(6, ch_hidden, kernel_size=(k_y1, k_x1), batchnorm=True, act=self.act)
        self.two_convglus = nn.Sequential(
            ConvGLU(ch_hidden, ch_hidden, batchnorm=True, act=self.act, kernel_size=(k_y2, k_x2)),
            ConvGLU(ch_hidden, ch_hidden, act=self.act, kernel_size=(k_y2, k_x2))
        )
        self.convglu_last = ConvGLU(ch_hidden, ch_hidden , act=self.act)


        self.conv = nn.Conv2d(ch_hidden, 2, kernel_size=(k_y2, k_x2), padding=( (k_y2-1)//2 ,  (k_x2-1)//2  ) )

    def forward(self, x, mag_replaced, consistent, train_step = -1):
        x = torch.cat([x, mag_replaced, consistent], dim=1)
        x = self.convglu_first(x)
        residual = x
        x = self.two_convglus(x)
        x += residual
        x = self.convglu_last(x)
        x = self.conv(x)
        return x

    def parse(self, k_x1: int = 11, k_y1: int = 11, k_x2: int = 7, k_y2: int = 7, num_channel: int=16,  act = "sigmoid"):
        self.ch_hidden = num_channel

        self.act = act.lower()
        return (k_x1, k_y1, k_x2 ,k_y2)


class DeGLI_ED(nn.Module):
    def __init__(self, n_freq, config):
        super().__init__()

        self.parse(**config)

        layer_specs = [
            6, # encoder_1: [batch, 128, 128, 1] => [batch, 128, 128, ngf]
            self.widening, # encoder_1: [batch, 128, 128, 1] => [batch, 128, 128, ngf]
            self.widening * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            self.widening * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            self.widening * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            self.widening * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            self.widening * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            self.widening * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            self.widening * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]

        layer_specs = layer_specs[0:self.n_layers+1]


        self.encoders = nn.ModuleList()

        conv, pad = self._gen_conv(layer_specs[0] ,layer_specs[1], convGlu = self.convGlu, rounding_needed  = True)
        self.encoders.append(nn.Sequential(pad, conv))
        
        last_ch = layer_specs[1]

        for i,ch_out in enumerate(layer_specs[2:]):
            d = OrderedDict()
            d['act'] = str2act(self.act,self.lamb)
            gain  = math.sqrt(2.0/(1.0+self.lamb**2))
            gain = gain / math.sqrt(2)  ## for naive signal propagation with residual w/o bn

            conv, pad  = self._gen_conv(last_ch ,ch_out, gain = gain, convGlu = self.convGlu, kernel_size = self.k_xy)
            if not pad is None:
                d['pad'] = pad
            d['conv'] = conv

            if self.use_batchnorm:
                d['bn']  = nn.BatchNorm2d(ch_out)

            encoder_block = nn.Sequential(d)
            self.encoders.append(encoder_block)
            last_ch = ch_out

        layer_specs.reverse()
        self.decoders = nn.ModuleList()
        kernel_size = 4
        for i,ch_out in enumerate(layer_specs[1:]):

            d = OrderedDict()
            d['act'] = str2act(self.act2,self.lamb)
            gain  =  math.sqrt(2.0/(1.0+self.lamb**2))
            gain = gain / math.sqrt(2) 

            if i == len(layer_specs)-2:
                 kernel_size = 5
                 ch_out = 2
            conv = self._gen_deconv(last_ch, ch_out , gain = gain, k= kernel_size)
            d['conv'] = conv

            # if i < self.num_dropout and self.droprate > 0.0:
            #     d['dropout'] = nn.Dropout(self.droprate)

            if self.use_batchnorm and i < self.n_layers-1:
                d['bn']  = nn.BatchNorm2d(ch_out)

            decoder_block = nn.Sequential(d)
            self.decoders.append(decoder_block)
            last_ch = ch_out * 2

        if self.use_linear_finalizer:
            init_alpha = 0.001
            self.linear_finalizer = nn.Parameter(torch.ones(n_freq) * init_alpha , requires_grad = True)

    def parse(self, layers:int, k_x:int, k_y:int, s_x:int, s_y:int, widening:int,use_bn: bool, lamb: float, linear_finalizer:bool, convGlu: bool, act: str, act2 : str, glu_bn:bool) :
        self.n_layers = layers
        self.k_xy = (k_y, k_x)
        self.s_xy = (s_y, s_x)
        self.widening = widening
        self.use_batchnorm = use_bn
        self.lamb = lamb
        self.use_linear_finalizer = linear_finalizer
        self.convGlu = convGlu
        self.act = act
        self.act2 = act2
        self.glu_bn = glu_bn
    def forward(self, x, mag_replaced, consistent, train_step = -1):
        
        
        ##import pdb; pdb.set_trace()
        x = torch.cat([x, mag_replaced, consistent], dim=1)

        encoders_output = []

        for i,encoder in enumerate(self.encoders):
            x = encoder(x)
            encoders_output.append(x)

        for i,decoder in enumerate(self.decoders[:-1]):
            x = decoder(x)
            x = torch.cat([x, encoders_output[-(i+2)]], dim=1)

        x = self.decoders[-1](x) 

        if self.use_linear_finalizer:
            x_perm = x.permute(0,1,3,2)
            x = torch.mul(x_perm,  self.linear_finalizer) 
            x = x.permute(0,1,3,2)

        return x

    def _gen_conv(self, in_ch,  out_ch, strides = (2, 1), kernel_size = (5,3), gain = math.sqrt(2), convGlu = False, rounding_needed= False):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        ky,kx = kernel_size
        p1x = (kx-1)//2
        p2x = kx-1 - p1x
        p1y = (ky-1)//2
        p2y = ky-1 - p1y

        if rounding_needed:
            pad_counts = (p1x,p2x,p1y-1 , p2y)
            pad = torch.nn.ReplicationPad2d(pad_counts)
        else:
            pad = None

        if convGlu:
            conv =  ConvGLU(in_ch, out_ch, kernel_size=kernel_size, stride = strides, batchnorm=self.glu_bn , padding=(0,0), act= "sigmoid")
        else:
            if pad is None:
                conv =  nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride = strides, padding = (p1y, p1x) )
            else:
                conv =  nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride = strides , padding=0)

        w = conv.weight
        k = w.size(1) * w.size(2) * w.size(3)
        conv.weight.data.normal_(0.0, gain / math.sqrt(k) )
        nn.init.constant_(conv.bias,0.01)
        return conv, pad 

    def _gen_deconv(self, in_ch,  out_ch, strides = (2, 1), k = 4, gain = math.sqrt(2), p =1 ):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]

        conv =  nn.ConvTranspose2d(in_ch, out_ch, kernel_size= (k,3), stride = strides, padding_mode='zeros',padding = (p,1), dilation  = 1)

        w = conv.weight
        k = w.size(1) * w.size(2) * w.size(3)
        conv.weight.data.normal_(0.0, gain / math.sqrt(k) )
        nn.init.constant_(conv.bias,0.01)

        return conv

def replace_magnitude(x, mag):
    phase = torch.atan2(x[:, 1:], x[:, :1])  # imag, real
    return torch.cat([mag * torch.cos(phase), mag * torch.sin(phase)], dim=1)


class DeGLI(nn.Module):
    def __init__(self, writer, model_config,  model_type ,  n_freq:int, use_fp16:bool , n_fft: int, hop_length: int, depth:int, out_all_block:bool):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.out_all_block = out_all_block
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        self.istft = InverseSTFT(n_fft, hop_length=self.hop_length, window=self.window.data)

        model_type = model_type.lower()
        if model_type == "vanilla":
            self.dnns = nn.ModuleList([DeGLI_DNN(model_config) for _ in range(depth)])
        elif model_type == "ed":
            self.dnns = nn.ModuleList([DeGLI_ED( n_freq ,model_config) for _ in range(depth)])

        # self.use_fp16 = use_fp16

        # if self.use_fp16:
        #     for dnn in self.dnns:
        #         dnn = dnn.half()

    def stft(self, x):
        return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window)

    def forward(self, x, mag, max_length=None, repeat=1, train_step = -1):
        if isinstance(max_length, torch.Tensor):
            max_length = max_length.item()

        out_repeats = []
        for ii in range(repeat):
            for dnn in self.dnns:
                # B, 2, F, T
                mag_replaced = replace_magnitude(x, mag)

                # B, F, T, 2
                waves = self.istft(mag_replaced.permute(0, 2, 3, 1), length=max_length)
                consistent = self.stft(waves)

                # B, 2, F, T
                consistent = consistent.permute(0, 3, 1, 2)
                # if self.use_fp16:
                #     residual = dnn(x.half() , mag_replaced.half(), consistent.half(), train_step = train_step).float()
                # else:
                residual = dnn(x , mag_replaced, consistent, train_step = train_step)
                
                x = consistent - residual
            if self.out_all_block:
                out_repeats.append(x)

        if self.out_all_block:
            out_repeats = torch.stack(out_repeats, dim=1)
        else:
            out_repeats = x.unsqueeze(1)

        final_out = replace_magnitude(x, mag)

        return out_repeats, final_out, residual

    def plain_gla(self, x, mag, max_length=None, repeat=1, train_step = -1):
        if isinstance(max_length, torch.Tensor):
            max_length = max_length.item()

        out_repeats = []
        for _ in range(repeat):
            for _ in self.dnns:
                # B, 2, F, T
                mag_replaced = replace_magnitude(x, mag)

                # B, F, T, 2
                waves = self.istft(mag_replaced.permute(0, 2, 3, 1), length=max_length)
                consistent = self.stft(waves)

                # B, 2, F, T
                x = consistent.permute(0, 3, 1, 2)
            if self.out_all_block:
                out_repeats.append(x)

        if self.out_all_block:
            out_repeats = torch.stack(out_repeats, dim=1)
        else:
            out_repeats = x.unsqueeze(1)

        final_out = replace_magnitude(x, mag)
        return out_repeats, final_out