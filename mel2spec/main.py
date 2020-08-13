""" Train or Test DNN

Usage:
```
    python main.py {--train, --test={seen, unseen}}
                   [--DF {IV, DirAC}]
                   [--room_train ROOM_TRAIN]
                   [--room_test ROOM_TEST]
                   [--logdir LOGDIR]
                   [--n_epochs MAX_EPOCH]
                   [--from START_EPOCH]
                   [--device DEVICES] [--out_device OUT_DEVICE]
                   [--batch_size B]
                   [--learning_rate LR]
                   [--weight_decay WD]
                   [--model_name MODEL]
```

More parameters are in `hparams.py`.
- specify `--train` or `--test {seen, unseen}`.
- DF: "IV" for using spatially-averaged intensity, "DirAC" for using direction vector.
- ROOM_TRAIN: room used to train
- ROOM_TEST: room used to test
- LOGDIR: log directory
- MAX_EPOCH: maximum epoch
- START_EPOCH: start epoch (Default: -1)
- DEVICES, OUT_DEVICE, B, LR, WD, MODEL: read `hparams.py`.
"""
# noinspection PyUnresolvedReferences
##import matlab.engine

import os
import shutil
from argparse import ArgumentError, ArgumentParser
from torch.utils.data import DataLoader

from dataset import ComplexSpecDataset
from hparams import hp
from train import Trainer

from create_mel import createLin

tfevents_fname = 'events.out.tfevents.*'
form_overwrite_msg = 'The folder "{}" already has tfevent files. Continue? [y/n]\n'

parser = ArgumentParser()

parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--inspect', action='store_true')

parser.add_argument('--infer', action='store_true')
parser.add_argument('--from', type=int, default=-1, dest='epoch', metavar='EPOCH')

args = hp.parse_argument(parser)
del parser
if not (args.train ^ args.test ^ args.infer ^ args.inspect ) or args.epoch < -1:
    raise ArgumentError

# directory
logdir_train = hp.logdir / 'train'
if (args.train and args.epoch == -1 and
        logdir_train.exists() and list(logdir_train.glob(tfevents_fname))):
    ##ans = input(form_overwrite_msg.format(logdir_train))
    ans = 'y'

    if ans.lower() == 'y':
        shutil.rmtree(logdir_train)
        try:
            os.remove(hp.logdir / 'summary.txt')
            os.remove(hp.logdir / 'hparams.txt')
        except FileNotFoundError:
            pass
    else:
        exit()

if args.test or args.inspect:
    logdir_test = hp.logdir
    foldername_test = f'test_{args.epoch}'
    if hp.n_save_block_outs > 0:
        foldername_test += '_blockouts'
    logdir_test /= foldername_test
    if logdir_test.exists() and list(logdir_test.glob(tfevents_fname)):
        ##ans = input(form_overwrite_msg.format(logdir_test))
        ans ='y'
        if ans.lower().startswith('y'):
            shutil.rmtree(logdir_test)
            os.makedirs(logdir_test)
        else:
            exit()
    os.makedirs(logdir_test, exist_ok=True)

if args.infer:
    logdir_infer = hp.logdir
    foldername_infer = f'infer_{args.epoch}'

    logdir_infer /= foldername_infer
    if logdir_infer.exists() and list(logdir_infer.glob(tfevents_fname)):
        ##ans = input(form_overwrite_msg.format(logdir_test))
        ans ='y'
        if ans.lower().startswith('y'):
            shutil.rmtree(logdir_infer)
            os.makedirs(logdir_infer)
        else:
            exit()
    os.makedirs(logdir_infer, exist_ok=True)

# epoch, state dict
first_epoch = args.epoch + 1
if first_epoch > 0:
    path_state_dict = logdir_train / f'{args.epoch}.pt'
    if not path_state_dict.exists():
        raise FileNotFoundError(path_state_dict)
else:
    path_state_dict = None

# Training + Validation Set


# run
if args.infer and path_state_dict is None:
    trainer = None
    num_workers = 1
else:
    os.makedirs(logdir_train, exist_ok=True)
    trainer = Trainer(path_state_dict)
    num_workers = trainer.num_workers
    
if args.train:
    dataset_train = ComplexSpecDataset('train')
    dataset_valid = ComplexSpecDataset('valid')

    dataset_train.set_needs(**hp.channels)
    ##dataset_valid.set_needs(**hp.channels)
    loader_train = DataLoader(dataset_train,
                              batch_size=hp.batch_size,
                              num_workers=num_workers,
                              collate_fn=dataset_train.pad_collate,
                              pin_memory=(hp.device != 'cpu'),
                              shuffle=True,
                              drop_last=True,
                              )
    loader_valid = DataLoader(dataset_valid,
                              batch_size=hp.batch_size * 2,
                              num_workers=num_workers,
                              collate_fn=dataset_valid.pad_collate,
                              pin_memory=(hp.device != 'cpu'),
                              shuffle=False,
                              drop_last=True,
                              )

    trainer.train(loader_train, loader_valid, logdir_train, first_epoch)
elif args.test: 
    # Test Set
    dataset_test = ComplexSpecDataset('test')
    loader = DataLoader(dataset_test,
                        batch_size=hp.batch_size * 2,
                        num_workers=num_workers,
                        collate_fn=dataset_test.pad_collate,
                        pin_memory=(hp.device != 'cpu'),
                        shuffle=False,
                        )

    # noinspection PyUnboundLocalVariable
    trainer.test(loader, logdir_test)
elif args.inspect: 
    # Test Set
    dataset_test = ComplexSpecDataset('valid')
    loader = DataLoader(dataset_test,
                        batch_size=hp.batch_size * 2,
                        num_workers=num_workers,
                        collate_fn=dataset_test.pad_collate,
                        pin_memory=(hp.device != 'cpu'),
                        shuffle=False,
                        )

    # noinspection PyUnboundLocalVariable
    trainer.inspect(loader, logdir_test)
elif args.infer: 

    tasks = hp.infer_task.split(';')
    print(tasks)
    infer_valid = infer_train = infer_test = False
    for task in tasks:
        task = task.lower()
        if task == "train":
            infer_train = True
        elif task[0] == 'v':
            infer_valid = True
        elif task[0] == 't':
            infer_test = True
        else:
            raise NotImplementedError(task)

    assert (infer_valid or infer_train or  infer_test)
    # Train Set
    if infer_valid:

        dataset =  ComplexSpecDataset('valid')
        ##dataset.set_needs(**hp.channels)

        loader = DataLoader(dataset,
                            batch_size=hp.batch_size * 2,
                            num_workers=num_workers,
                            collate_fn=dataset.pad_collate,
                            pin_memory=(hp.device != 'cpu'),
                            shuffle=False,
                            )

        logdir_infer_valid = logdir_infer / "VALID" 
        os.makedirs(logdir_infer_valid, exist_ok=True)

        if trainer is not None:
            trainer.infer(loader, logdir_infer_valid)
        else:
            createLin(loader, logdir_infer_valid, hp.num_snr)

    if infer_train:
        # Train Set
        dataset = ComplexSpecDataset('train')
        loader = DataLoader(dataset,
                            batch_size=hp.batch_size * 2,
                            num_workers=num_workers,
                            collate_fn=dataset.pad_collate,
                            pin_memory=(hp.device != 'cpu'),
                            shuffle=False,
                            )

        logdir_infer_train = logdir_infer / "TRAIN" 
        os.makedirs(logdir_infer_train, exist_ok=True)

        if trainer is not None:
            trainer.infer(loader, logdir_infer_train)
        else:
            createLin(loader, logdir_infer_train, hp.num_snr)
        # Test Set
    if infer_test:

        dataset = ComplexSpecDataset('test')
        loader = DataLoader(dataset,
                            batch_size=hp.batch_size * 2,
                            num_workers=num_workers,
                            collate_fn=dataset.pad_collate,
                            pin_memory=(hp.device != 'cpu'),
                            shuffle=False,
                            )

        hp.num_snr = 1
        logdir_infer_test = logdir_infer / "TEST" 
        os.makedirs(logdir_infer_test, exist_ok=True)

        if trainer is not None:
            trainer.infer(loader, logdir_infer_test)
        else:
            createLin(loader, logdir_infer_test, 1)


else:
    print("Must use --train, --test or --infer")