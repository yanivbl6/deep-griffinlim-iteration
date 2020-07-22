import torch
import torch.nn as nn

from .istft import InverseSTFT

from lib.modules.deq2d import *
from lib.models.mdeq_forward_backward import MDEQWrapper
from lib.modules.optimizations import *

from lib.models.mdeq import Bottleneck
from lib.models.mdeq_core import *


from model.degli import ConvGLU


def replace_magnitude(x, mag):
    phase = torch.atan2(x[:, 1:], x[:, :1])  # imag, real
    return torch.cat([mag * torch.cos(phase), mag * torch.sin(phase)], dim=1)





class DeqGLI(nn.Module):
    def __init__(self, writer, dedeq_config, n_freq: int , n_fft: int,  hop_length: int, depth:int, out_all_block:bool):
        super().__init__()



        self.n_fft = n_fft
        self.hop_length = hop_length
        self.out_all_block = out_all_block
        self.parse(writer, **dedeq_config)

        base_pad = (self.k1 - 1) // 2
        extra_pad1 = (self.freq_embedding - n_freq) // 2
        extra_pad2 = (self.freq_embedding - n_freq) - extra_pad1

        self.pad0 = torch.nn.ReplicationPad2d((0,0,base_pad+extra_pad2,base_pad+extra_pad1))
        self.conv_first = nn.Conv2d(3, self.ch_hidden, kernel_size=(self.k1, 1), padding=(0, 0))

        #-------------------------- START: Deep Equalibrium model initialization -------------------- #
        self.fullstage = MDEQModule(self.num_branches, self.block_type, self.num_blocks, self.num_channels, self.fuse_method, dropout=self.droprate)
        self.fullstage_copy = copy.deepcopy(self.fullstage)

        for param in self.fullstage_copy.parameters():
            param.requires_grad_(False)
        self.deq = MDEQWrapper(self.fullstage, self.fullstage_copy)

        
        if self.wnorm:
            self.fullstage._wnorm()
            
        for param in self.fullstage_copy.parameters():
            param.requires_grad_(False)

        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(self.num_channels)
        self.iodrop = VariationalHidDropout2d(0.0)

        #--------------------------- END: Deep Equalibrium model initialization --------------------- #

        base_pad = (self.k2 - 1) // 2
        extra_pad1 = (self.freq_embedding - n_freq) // 2
        extra_pad2 = (self.freq_embedding - n_freq) - extra_pad1

        self.pad1 = torch.nn.ReplicationPad2d((0,0,base_pad-extra_pad2,base_pad-extra_pad1))
        self.conv_last = nn.Conv2d(2*self.ch_hidden, 2, kernel_size=(self.k2, 1) , padding=(0, 0))

        self.init_weights()

    def init_weights(self, pretrained=''):
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

    def parse(self,writer, wnorm: bool, num_branches: int, base_channels: int, ratio2head: int, fuse_method: str,
                 droprate: float, final_multiplier: int, pretrain_steps:int, f_thres:int, b_thres:int, num_layers:int,
                 ch_hidden: int, k1:int, k2:int, p2:int, freq_embedding:int):

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
        self.ch_hidden = ch_hidden
        self.k1 = k1
        self.k2 = k2
        self.p2 = p2
        self.freq_embedding = freq_embedding

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

    def forward(self, x, mag, max_length=None, repeat=1, train_step = -1):
        if isinstance(max_length, torch.Tensor):
            max_length = max_length.item()

        dev = x.device
        x = torch.cat([mag, x], dim=1)

        x = self.pad0(x)

        x = self.conv_first(x)

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

        x = self.pad1(y)

        x = self.conv_last(x)

        final_out = replace_magnitude(x, mag)

        return x.unsqueeze(1), final_out, None


    def _forward(self, x, mag, max_length=None, repeat=1, train_step = -1):
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

