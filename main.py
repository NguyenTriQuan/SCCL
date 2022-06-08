import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import math
import copy

import utils
from utils import *
from arguments import get_args
from sccl_layer import DynamicLinear, DynamicConv2D, _DynamicLayer

import importlib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]'); sys.exit()

approach = importlib.import_module('approaches.{}'.format(args.approach))

try:
    networks = importlib.import_module('networks.{}_net'.format(args.approach))
except:
    networks = importlib.import_module('networks.net')


# dataloader = importlib.import_module('dataloaders.{}'.format(args.experiment))
# data, taskcla, inputsize = dataloader.get(batch_size=args.batch_size, val_batch_size=args.val_batch_size, seed=args.seed, tasknum=args.tasknum)

try:
    dataloader = importlib.import_module('dataloaders.{}'.format(args.experiment))
    data, taskcla, inputsize = dataloader.get(batch_size=args.batch_size, val_batch_size=args.val_batch_size, seed=args.seed, tasknum=args.tasknum)
except:
    dataloader = importlib.import_module('dataloaders.single_task')
    data, taskcla, inputsize = dataloader.get(batch_size=args.batch_size, val_batch_size=args.val_batch_size, seed=args.seed, name=args.experiment)

print('Input size =', inputsize, '\nTask info =', taskcla)

Net = getattr(networks, args.arch)
print(Net)
if 'sccl' in args.approach:
    net = Net(inputsize).cuda()
else:
    net = Net(inputsize, taskcla).cuda()

# print(utils.print_model_report(net))

appr = approach.Appr(net, args=args)

utils.print_optimizer_config(appr.optimizer)
print('-' * 100)

acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)

past_ncla = [ncla for t, ncla in taskcla]

start_task = args.start_task
if args.resume:
    start_task = appr.resume()


for t, ncla in taskcla[start_task:]:
    if t >= args.tasknum: break
    print('*' * 100)
    print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    task = t

    # Train
    if args.experiment == 'split_cifar10_100' or args.experiment == 'split_cifar10_100_big':
        if 'cifar100' in data[t]['name']:
            appr.sbatch = 32
        else:
            appr.sbatch = 256

    if 'sccl' in args.approach:
        appr.train(task+1, data[t]['train loader'], data[t]['valid loader'], ncla=ncla)
    else:
        appr.train(task, data[t]['train loader'], data[t]['valid loader'])
    print('-' * 100)

    # Test
    for u in range(t + 1):
        if 'sccl' in args.approach:
            if args.cil:
                test_loss, test_acc = appr.eval(None, data[u]['test loader'])
            else:
                test_loss, test_acc = appr.eval(u+1, data[u]['test loader'])
        else:
            test_loss, test_acc = appr.eval(u, data[u]['test loader'])

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
