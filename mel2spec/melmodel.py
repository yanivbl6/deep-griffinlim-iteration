import torch
import torch.nn as nn


import math

from collections import OrderedDict

from numpy.linalg import pinv

def str2act(txt, param= None):
    return {"sigmoid": nn.Sigmoid(), "relu": nn.ReLU(), "none": nn.Sequential() , "lrelu": nn.LeakyReLU(param), "selu": nn.SELU() }[txt.lower()]


class melGen(nn.Module):

    def __init__(self, writer, n_freq  ,meltrans , melgen_config):
        super().__init__()

        self.meltrans = nn.Parameter(torch.transpose(torch.tensor(meltrans, dtype=torch.float),0,1), requires_grad = False) ## maybe turn grad on?
        self.meltrans_inv = nn.Parameter(torch.transpose(torch.tensor(pinv(meltrans), dtype=torch.float),0,1), requires_grad = False) ## maybe turn grad on?

        self.parse(writer,   **melgen_config)



        layer_specs = [
            1, # encoder_1: [batch, 128, 128, 1] => [batch, 128, 128, ngf]
            self.ngf, # encoder_1: [batch, 128, 128, 1] => [batch, 128, 128, ngf]
            self.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            self.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            self.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            self.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            self.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            self.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            self.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
            self.ngf * 8, # encoder_9: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]

        layer_specs = layer_specs[0:self.n_layers+1]

        n_time = self.subseq_len

        self.encoders = nn.ModuleList()

        conv, pad = self._gen_conv(layer_specs[0] ,layer_specs[1])
        self.encoders.append(nn.Sequential(pad, conv))
        
        last_ch = layer_specs[1]
        #n_time = n_time /2

        for i,ch_out in enumerate(layer_specs[2:]):
            d = OrderedDict()
            d['act'] = str2act(self.act1,self.lamb)
            gain  = math.sqrt(2.0/(1.0+self.lamb**2))
            gain = gain / math.sqrt(2)  ## for naive signal propagation with residual + bn

            #if n_time > 1:
            conv, pad  = self._gen_conv(last_ch ,ch_out, gain = gain)
            #    n_time /= 2
            # else:
            #     n_stride1_layers += 1
            #     conv,pad = self._gen_conv(last_ch ,ch_out,strides = (1,2), gain = gain)
            d['pad'] = pad
            d['conv'] = conv

            if self.use_batchnorm:
                d['bn']  = nn.BatchNorm2d(ch_out)

            encoder_block = nn.Sequential(d)
            self.encoders.append(encoder_block)
            last_ch = ch_out

        layer_specs.reverse()
        self.decoders = nn.ModuleList()
        for i,ch_out in enumerate(layer_specs[1:]):

            d = OrderedDict()
            d['act'] = str2act(self.act2,self.lamb)
            gain  =  math.sqrt(2.0/(1.0+self.lamb**2))
            gain = gain / math.sqrt(2) 
            
            # if i < n_stride1_layers:
            #     conv = self._gen_deconv(last_ch, ch_out,strides = (1,2), gain= gain)
            # else:
 
            kernel_size = 4 if i < len(layer_specs)-2 else 5
            conv = self._gen_deconv(last_ch, ch_out , gain = gain, k= kernel_size)
            ##d['pad'] = pad
            d['conv'] = conv

            if i < self.num_dropout and self.droprate > 0.0:
                d['dropout'] = nn.Dropout(self.droprate)

            if self.use_batchnorm and i < self.n_layers-1:
                d['bn']  = nn.BatchNorm2d(ch_out)

            decoder_block = nn.Sequential(d)
            self.decoders.append(decoder_block)
            last_ch = ch_out * 2

        init_alpha = 0.001
        self.linear_finalizer = nn.Parameter(torch.ones(n_freq) * init_alpha , requires_grad = True)
    
        if self.pre_final_lin:
            self.linear_pre_final = nn.Parameter(torch.ones(self.ngf*2, n_freq//2) , requires_grad = True)


    def mel_pseudo_inverse(self,x):
        return torch.tensordot(x,self.meltrans_inv, dims=[[2],[0]]).permute(0,1,3,2)

    def spec_to_mel(self,x):
        return torch.tensordot(x,self.meltrans, dims=[[2],[0]]).permute(0,1,3,2)

    def forward(self, x):
        x = self.mel_pseudo_inverse(x)
        # print("input:")
        # print(x.shape)

        x_in = x


        encoders_output = []
        ##import pdb; pdb.set_trace()

        for i,encoder in enumerate(self.encoders):
            x = encoder(x)
            encoders_output.append(x)
            ##print("encoder %d output:" % i)
            # ##print('XXX' * i)
            # print(x.shape)

        for i,decoder in enumerate(self.decoders[:-1]):
            x = decoder(x)            
            # print("decoder %d output:" % (i+1))
            # print(x.shape)
            x = torch.cat([x, encoders_output[-(i+2)]], dim=1)

        if self.pre_final_lin:
            x_perm = x.permute(0,3,1,2)
            x = torch.mul(x_perm,  self.linear_pre_final).permute(0,2,3,1)

        x = self.decoders[-1](x) 
        ##x = torch.tensordot(x, self.linear_finalizer, dims = [[2] ,[0]]).permute(0,1,3,2)
        x_perm = x.permute(0,1,3,2)
        x = torch.mul(x_perm,  self.linear_finalizer) 
        x = x.permute(0,1,3,2)

        x = x + x_in ##add to input 
        # print("output:")
        # print(x.shape)
        return x

    def _gen_conv(self, in_ch,  out_ch, strides = (2, 1), kernel_size = (5,3), gain = math.sqrt(2), pad = (1,1,1,2)):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        if self.separable_conv:
            conv = None
            pad = None
            print("separable_conv Not implemented")
        else:
            pad = torch.nn.ReplicationPad2d(pad)
            conv =  nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride = strides , padding=0)

        w = conv.weight
        k = w.size(1) * w.size(2) * w.size(3)
        conv.weight.data.normal_(0.0, gain / math.sqrt(k) )
        nn.init.constant_(conv.bias,0.01)
        return conv, pad 

    def _gen_deconv(self, in_ch,  out_ch, strides = (2, 1), k = 4, gain = math.sqrt(2), p =1 ):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        if self.separable_conv:
            conv = None
            print("separable_conv Not implemented")
        else:
            conv =  nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(k,3), stride = strides, padding_mode='zeros',padding = (p,1), dilation  = 1)

        w = conv.weight
        k = w.size(1) * w.size(2) * w.size(3)
        conv.weight.data.normal_(0.0, gain / math.sqrt(k) )
        nn.init.constant_(conv.bias,0.01)

        return conv

    def parse(self, writer, layers:int, audio_fs:int , subseq_len:int, ngf:int, ndf:int, separable_conv:bool, use_batchnorm:bool, lamb:float, droprate:float,  num_dropout:int, pre_final_lin: bool, act1: str, act2:str):
        self.writer = writer
        self.n_layers = layers
        self.audio_fs = audio_fs
        self.subseq_len = subseq_len
        self.ngf = ngf
        self.ndf = ndf
        self.separable_conv = separable_conv
        self.use_batchnorm = use_batchnorm
        self.lamb = lamb
        self.droprate = droprate
        self.num_dropout = num_dropout
        self.pre_final_lin = pre_final_lin
        self.act1 = act1
        self.act2 = act2

