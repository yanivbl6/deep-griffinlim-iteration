
# noinspection PyUnresolvedReferences
##import matlab.engine

import os
import shutil
from argparse import ArgumentError, ArgumentParser

from torch.utils.data import DataLoader

from dataset import ComplexSpecDataset
from hparams1 import hp
from train import Trainer
from pathlib import Path

from os import listdir


parser = ArgumentParser()

parser.add_argument('-l','--list',action='store_true')

parser.add_argument('-n','--network',type=str)
parser.add_argument('-m','--mel2spec',type=str)
parser.add_argument('-d','--device',type=int, default=0)
parser.add_argument('--dest',type=str, default="../result/inference")

parser.add_argument('--network_results',type=str, default="../result/ngc_degli")
parser.add_argument('--mel2spec_results',type=str, default="../result/mel2spec")
parser.add_argument('-p','--perf', action='store_true')

parser.add_argument('-b','--batch_size', type=int, default=16)

args = parser.parse_args()

##import pdb; pdb.set_trace()
if args.list:
    print('-'*30)
    print("Available Networks:")
    for f in listdir(args.network_results):
        full_path = "%s/%s" % (args.network_results,f)
        if not os.path.isdir(full_path):  
            continue
        checkpoints = []
        full_path_train = "%s/train" % full_path

        if not os.path.exists(full_path_train):  
            continue
        for e in listdir(full_path_train):
            if e.__str__()[-2:] == "pt":
                checkpoints.append(int(e.split('.')[0]))

        if len(checkpoints) > 0:
            checkpoints.sort()
            print("%s : %s" % (f,checkpoints.__str__()))
    print('-'*30)
    print("Available Mel2Spec infered data:")

    for f in listdir(args.mel2spec_results):
        full_path = "%s/%s" % (args.mel2spec_results,f)
        if not os.path.isdir(full_path):  
            continue
        checkpoints = []
        for e in listdir(full_path):
            if e.split('_')[0] == "infer":
                checkpoints.append(int(e.split('_')[1]))
        if len(checkpoints) > 0:
            checkpoints.sort()
            print("%s : %s" % (f,checkpoints.__str__()))
    print('-'*30)

if not args.network is None:
    net_split = args.network.split(":")
    networkDir = net_split[0]
    networkEpoch = net_split[1]

    if args.perf:
        sub = "perf"
    else:
        sub = "quality"

    if not  args.mel2spec is None:
        mel_split = args.mel2spec.split(":")
        mel2specDir = mel_split[0]
        mel2specEpoch = mel_split[1]
        mel_dest = f"{args.mel2spec_results}/{mel2specDir}/infer_{mel2specEpoch}"
        full_dest= f"{args.dest}/{sub}/{networkDir}_E{networkEpoch}_mel2spec_{mel2specDir}_E{mel2specEpoch}"
    else:
        mel_dest = f"~/deep-griffinlim-iteration/mel2spec/baseline_data"
        full_dest= f"{args.dest}/{sub}/{networkDir}_E{networkEpoch}_baseline"

    os.makedirs(args.dest, exist_ok=True)

    command = "test"
    if args.perf:
        full_dest = full_dest + "_B%d" % args.batch_size
        command = "perf"
    cmd=f"python main.py --{command} --device {args.device} --from {networkEpoch} --logdir {args.network_results}/{networkDir} --path_feature {mel_dest} --dest_test {full_dest} --batch_size {args.batch_size}"

    print(cmd)