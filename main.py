import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import math
import copy
import utils
from utils import *
from arguments import get_args
import importlib
# import comet_ml at the top of your file
# from comet_ml import Experiment
import json

device = 'cuda'

args = get_args()
tstart = time.time()

args.max_params = max(args.max_params, 0)
args.max_mul = max(args.max_mul, 0)


print('=' * 100)
print('Arguments =')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 100)

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# dataloader = importlib.import_module('dataloaders.{}'.format(args.experiment))
# data, taskcla, inputsize = dataloader.get(batch_size=args.batch_size, val_batch_size=args.val_batch_size, seed=args.seed, tasknum=args.tasknum)

# try:
dataloader = importlib.import_module('dataloaders.{}'.format(args.experiment))
data, taskcla, inputsize = dataloader.get(args)
# except:
#     dataloader = importlib.import_module('dataloaders.single_task')
#     data, taskcla, inputsize = dataloader.get(batch_size=args.batch_size, val_batch_size=args.val_batch_size, seed=args.seed, name=args.experiment)

print('Input size =', inputsize, '\nTask info =', taskcla)

approach = importlib.import_module('approaches.{}'.format(args.approach))
appr = approach.Appr(inputsize=inputsize, taskcla=taskcla, args=args)

start_task = args.start_task
if args.resume:
    start_task = appr.resume()

# experiment = Experiment(
#     api_key="YSY2PKZaRYWMWkA9XvW0SnJzF",
#     project_name="sccl",
#     workspace="nguyentriquan",
# )
# experiment.set_name(appr.log_name)
# with open(f'{appr.log_name}.json', 'w') as file:
#     json.dump(experiment.get_key(), file)


utils.print_optimizer_config(appr.optimizer)
print('-' * 100)

acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)

past_ncla = [ncla for t, ncla in taskcla]


for t, ncla in taskcla[start_task:]:
    if t >= args.tasknum: break
    print('*' * 100)
    print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    task = t

    # Train
    if args.experiment == 'split_cifar10_100':
        if 'cifar100' in data[t]['name']:
            appr.sbatch = 32
        else:
            appr.sbatch = 256

    if 'sccl' in args.approach:
        appr.train(task+1, data[t]['train_loader'], data[t]['valid_loader'], data['train_transform'], data['valid_transform'], ncla=ncla)
    else:
        appr.train(task, data[t]['train_loader'], data[t]['valid_loader'], data['train_transform'], data['valid_transform'])
    print('-' * 100)

    # Test
    for u in range(t + 1):
        if 'sccl' in args.approach:
            if args.cil:
                test_loss, test_acc = appr.eval(None, data[u]['test_loader'], data['valid_transform'])
            else:
                appr.model.get_old_parameters(u+1)
                test_loss, test_acc = appr.eval(u+1, data[u]['test_loader'], data['valid_transform'])
        else:
            test_loss, test_acc = appr.eval(u, data[u]['test_loader'], data['valid_transform'])

        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.2f}% <<<'.format(u, data[u]['name'], test_loss, 100 * test_acc))
        acc[t, u] = test_acc
        lss[t, u] = test_loss

    # Save
    print('Avg acc={:5.2f}%'.format(100*sum(acc[t])/(t+1)))
    print('Save at ' + f'../result_data/{appr.log_name}.txt')
    np.savetxt(f'../result_data/{appr.log_name}.txt', acc, '%.4f')

# Done
print('*' * 100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t', end='')
    for j in range(acc.shape[1]):
        print('{:5.2f}% '.format(100 * acc[i, j]), end='')
    print()
print('*' * 100)
print('Done!')

print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
