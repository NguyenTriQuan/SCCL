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
import random
# import comet_ml at the top of your file
# from comet_ml import Experiment, ExistingExperiment
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('../result_data/trained_model/', exist_ok=True)
os.makedirs('../result_data/logger/', exist_ok=True)
args = get_args()
tstart = time.time()

args.max_params = max(args.max_params, 0)
args.max_mul = max(args.max_mul, 0)

print('=' * 100)
print('Arguments =')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 100)
print('hi')
# Seed
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

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
    # with open(f'../result_data/logger/{appr.log_name}.json', 'r') as f:
    #     KEY = json.load(f)

    # appr.logger = ExistingExperiment(
    #     api_key="YSY2PKZaRYWMWkA9XvW0SnJzF",
    #     previous_experiment=KEY
    # )
    start_task = appr.resume()
    print('start from task ', start_task)
# else:
#     appr.logger = Experiment(
#         api_key="YSY2PKZaRYWMWkA9XvW0SnJzF",
#         project_name="sccl",
#         workspace="nguyentriquan",
#     )
#     appr.logger.set_name(appr.log_name)
#     with open(f'../result_data/logger/{appr.log_name}.json', 'w') as f:
#         json.dump(appr.logger.get_key(), f)

print('-' * 100)

acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)

past_ncla = [ncla for t, ncla in taskcla]

if args.approach == 'joint':
    train_loaders = [data[t]['train_loader'] for t, ncla in taskcla]
    # valid_loaders = [data[t]['valid_loader'] for t, ncla in taskcla]
    test_loaders = [data[t]['test_loader'] for t, ncla in taskcla]

    train_transforms = [data[t]['train_transform'] for t, ncla in taskcla]
    valid_transforms = [data[t]['valid_transform'] for t, ncla in taskcla]

    appr.train(train_loaders, test_loaders, train_transforms, valid_transforms)

    test_losses, test_accs = appr.eval(test_loaders, valid_transforms)
    for t in range(len(test_accs)):
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.2f}% <<<'.format(t, data[t]['name'], test_losses[t], 100 * test_accs[t]))
    print('Avg acc={:5.2f}%'.format(100*np.mean(test_accs)))
    sys.exit()

for t, ncla in taskcla[start_task:]:
    if t >= args.tasknum: break
    print('*' * 100)
    print('Task {:2d} ({:s}), Train-{}/Test-{}'.format(t, data[t]['name'], data[t]['train_loader'].dataset.tensors[0].shape[0], data[t]['test_loader'].dataset.tensors[0].shape[0]))
    print('*' * 100)
    task = t
    train_start = time.time()
    # Train
    # appr.train(task, data[t]['train_loader'], data[t]['valid_loader'], data[t]['train_transform'], data[t]['valid_transform'], ncla=ncla)
    appr.train(task, data[t]['train_loader'], data[t]['test_loader'], data[t]['train_transform'], data[t]['valid_transform'], ncla=ncla)

    print('-' * 100)
    print(f'Task {t} training time: {time.time() - train_start} s')
    # Test
    for u in range(t + 1):
        test_loss, test_acc = appr.eval(u, data[u]['test_loader'], data[u]['valid_transform'])
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.2f}% <<<'.format(u, data[u]['name'], test_loss, 100 * test_acc))
        acc[t, u] = test_acc
        lss[t, u] = test_loss
    print('Avg acc={:5.2f}%'.format(100*sum(acc[t])/(t+1)))
    # Save
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
