import torch
import torch.nn as nn

from .istft import InverseSTFT

from lib.modules.deq2d import *
from lib.models.mdeq_forward_backward import MDEQWrapper
from lib.modules.optimizations import *

from lib.models.mdeq import Bottleneck
from lib.models.mdeq_core import *

import math

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
        self.num_channels = [ base_channels*(2**x) for x in range(self.num_branches)]
        self.head_channels = [x//ratio2head for x in self.num_channels]
        self.block_type = BasicBlock
        self.fuse_method = fuse_method
        self.droprate = droprate
        self.init_chansize = self.num_channels[0]
        self.final_chansize = self.num_channels[-1]*final_multiplier
        self.pretrain_steps = pretrain_steps
        self.f_thres = f_thres
        self.b_thres = b_thres
        self.num_layers = num_layers ## for pretrain

        self.convglu_first = ConvGLU(6, ch_hidden, kernel_size=(k1, k1), batchnorm=True)
        self.fullstage = MDEQModule(self.num_branches, self.block_type, self.num_blocks, self.num_channels, self.fuse_method, dropout=self.droprate)
        self.fullstage_copy = copy.deepcopy(self.fullstage)

        for param in self.fullstage_copy.parameters():
            param.requires_grad_(False)
        self.deq = MDEQWrapper(self.fullstage, self.fullstage_copy)

        
        if self.wnorm:
            self.fullstage._wnorm()
            
        for param in self.fullstage_copy.parameters():
            param.requires_grad_(False)

        self.iodrop = VariationalHidDropout2d(0.0)

        self.convglu_last = ConvGLU(2*ch_hidden, ch_hidden)
        self.conv = nn.Conv2d(ch_hidden, 2, kernel_size=(k2, k2), padding=(p2, p2))

        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(self.num_channels)
            
        self.init_weights('')
        


    def _make_head(self, pre_stage_channels):
        """
        Create a classification head that:
           - Increase the number of features in each resolution 
           - Downsample higher-resolution equilibria to the lowest-resolution and concatenate
           - Pass through a final FC layer for classification
        """
        head_block = Bottleneck
        d_model = self.init_chansize
        head_channels = self.head_channels
        
        # Increasing the number of channels on each resolution when doing classification. 
        incre_modules = []
        for i, channels  in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block, channels, head_channels[i], blocks=1, stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
            
        # Downsample the high-resolution streams to perform classification
        downsamp_modules = []
        for i in range(len(pre_stage_channels)-1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i+1] * head_block.expansion

            downsamp_module = nn.Sequential(conv3x3(in_channels, out_channels, stride=2, bias=True),
                                            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                                            nn.ReLU(inplace=True))
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        # Final FC layers
        final_layer = nn.Sequential(nn.Conv2d(head_channels[len(pre_stage_channels)-1] * head_block.expansion,
                                              self.final_chansize,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0),
                                    nn.BatchNorm2d(self.final_chansize, momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True))
        return incre_modules, downsamp_modules, final_layer

    def forward(self, x , mag_replaced, consistent, train_step=-1):

        dev = x.device
        x = torch.cat([x, mag_replaced, consistent], dim=1)
        x = self.convglu_first(x)
        ## x = self.two_convglus(x) replace

        x_list = [x]
        for i in range(1, self.num_branches):
            bsz, _, H, W = x_list[-1].shape
            x_list.append(torch.zeros(bsz, self.num_channels[i], H//2, W//2).to(dev))   # ... and the rest are all zeros
            
        z_list = [torch.zeros_like(elem) for elem in x_list]
        
        # For variational dropout mask resetting and weight normalization re-computations
        self.fullstage._reset(z_list)
        self.fullstage_copy._copy(self.fullstage)
        
        # Multiscale Deep Equilibrium!
        if 0 <= train_step < self.pretrain_steps:
            for layer_ind in range(self.num_layers):
                z_list = self.fullstage(z_list, x_list)
        else:
            if train_step == self.pretrain_steps:
                torch.cuda.empty_cache()
            z_list = self.deq(z_list, x_list, threshold=self.f_thres, train_step=train_step, writer=self.writer)
        y_list = self.iodrop(z_list)
        y = self.incre_modules[0](y_list[0])

        # Classification Head
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i+1](y_list[i+1]) + self.downsamp_modules[i](y)



        x = self.convglu_last(y)
        x = self.conv(x)


        return x

    def init_weights(self, pretrained='',):
        """
        Model initialization. If pretrained weights are specified, we load the weights.
        """
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

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

        conv, pad = self._gen_conv(layer_specs[0] ,layer_specs[1], convGlu = self.convGlu)
        self.encoders.append(nn.Sequential(pad, conv))
        
        last_ch = layer_specs[1]

        for i,ch_out in enumerate(layer_specs[2:]):
            d = OrderedDict()
            d['act'] = str2act(self.act,self.lamb)
            gain  = math.sqrt(2.0/(1.0+self.lamb**2))
            gain = gain / math.sqrt(2)  ## for naive signal propagation with residual w/o bn

            conv, pad  = self._gen_conv(last_ch ,ch_out, gain = gain, convGlu = self.convGlu, kernel_size = self.k_xy)

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

    def _gen_conv(self, in_ch,  out_ch, strides = (2, 1), kernel_size = (5,3), gain = math.sqrt(2), convGlu = False):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        ky,kx = kernel_size
        p1x = (kx-1)//2
        p2x = kx-1 - p1x
        p1y = (ky-1)//2
        p2y = ky-1 - p1y
        pad = (p1x,p2x,p1y-1 , p2y)

        pad = torch.nn.ReplicationPad2d(pad)
        if convGlu:
            conv =  ConvGLU(in_ch, out_ch, kernel_size=kernel_size, stride = strides, batchnorm=self.glu_bn , padding=(0,0), act= "sigmoid")
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