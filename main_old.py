import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import math
import copy

import utils
from utils import *
from arguments import get_args
from SCCL_layer import DynamicLinear, DynamicConv2D, _DynamicLayer, DynamicBatchNorm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tstart = time.time()

# Arguments

args = get_args()

if args.max_params < 0:
	args.max_params = 0
if args.max_mul < 0:
	args.max_mul = 0

if args.approach == 'sccl':
	lambs = [float(i) for i in args.lamb.split('_')]
	if len(lambs) < args.tasknum:
		lambs = [lambs[-1] if i>=len(lambs) else lambs[i] for i in range(args.tasknum)]

	print('lambs:', lambs)
else:
	args.lamb = float(args.lamb)

if args.approach == 'si':
	log_name = '{}_{}_{}_{}_c_{}_lr_{}_unitN_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, args.approach,args.seed, 
																	args.c, args.lr, args.unitN, args.batch_size, args.nepochs)
elif args.approach == 'ewc' or args.approach == 'rwalk' or args.approach == 'mas':
	log_name = '{}_{}_{}_{}_lamb_{}_mul_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, args.approach,args.seed,
																			args.lamb, args.mul, args.lr, 
																			 args.batch_size, args.nepochs)
elif args.approach == 'ucl' or args.approach == 'baye_hat':
	log_name = '{}_{}_{}_{}_alpha_{}_beta_{:.5f}_ratio_{:.4f}_mul_{}_lr_{}_lr_rho_{}_batch_{}_epoch_{}'.format(
		args.date, args.experiment, args.approach, args.seed, args.alpha, args.beta, args.ratio, 
		args.mul, args.lr, args.lr_rho, args.batch_size, args.nepochs)

elif args.approach == 'ucl_ablation':
	log_name = '{}_{}_{}_{}_{}_alpha_{}_beta_{:.5f}_ratio_{:.4f}_lr_{}_lr_rho_{}_unitN_{}_batch_{}_epoch_{}'.format(
		args.date, args.experiment, args.approach, args.seed, args.ablation, args.alpha, args.beta, args.ratio, 
		args.lr, args.lr_rho, args.unitN, args.batch_size, args.nepochs)

elif args.approach == 'hat' or args.approach == 'hat_expand':
	log_name = '{}_{}_{}_{}_gamma_{}_smax_{}_mul_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, 
																			  args.approach, args.seed,
																			  args.gamma, args.smax, args.mul, args.lr, 
																			  args.batch_size, args.nepochs)

elif args.approach == 'gs':
	log_name = '{}_{}_{}_{}_lamb_{}_mu_{}_rho_{}_eta_{}_mul_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.experiment,
																						  args.approach, args.seed, 
																						  args.lamb, args.mu, args.rho, args.eta, args.mul,
																						  args.lr, args.batch_size, args.nepochs)
elif args.approach == 'baseline':
	log_name = '{}_{}_{}_{}_mul_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, args.approach, args.seed, 
																				args.mul, args.lr, 
																			 args.batch_size, args.nepochs)
elif args.approach == 'finetuning':
	log_name = '{}_{}_{}_{}_mul_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, args.approach, args.seed, 
																				args.mul, args.lr, 
																			 args.batch_size, args.nepochs)

elif args.approach == 'sccl':
	# log_name = '{}_{}_{}_{}_lamb_{}_mul_{}_max_mul_{}_max_params_{}_lr_{}_batch_{}'.format(args.date, args.experiment, args.approach, args.seed,
	# 																		'_'.join([str(lamb) for lamb in lambs]), 
	# 																		args.mul, args.max_mul, args.max_params, args.lr, args.batch_size)
	log_name = ''

print('=' * 100)
print('Arguments =')
for arg in vars(args):
	print('\t' + arg + ':', getattr(args, arg))
print('=' * 100)

########################################################################################################################
# Split
split = True
notMNIST = False
split_experiment = [
	'split_mnist', 
	'split_notmnist', 
	'split_cifar10',
	'split_cifar100',
	'split_cifar100_20',
	'split_cifar10_100',
	'split_pmnist',
	'split_row_pmnist', 
	'split_CUB200',
	'split_tiny_imagenet',
	'split_mini_imagenet', 
	'omniglot',
	'mixture'
]

conv_experiment = [
	'split_cifar10',
	'split_cifar100',
	'split_cifar100_20',
	'split_cifar10_100',
	'split_CUB200',
	'split_tiny_imagenet',
	'split_mini_imagenet', 
	'omniglot',
	'mixture'
]

if args.experiment in split_experiment:
	split = True
if args.experiment == 'split_notmnist':
	notMNIST = True
if args.experiment in conv_experiment:
	args.conv_net = True
	# log_name = log_name + '_conv'
if args.output == '':
	args.output = '../result_data/' + log_name + '.txt'
   
split = True
# args.mul = math.sqrt(args.mul)
# args.max_mul = math.sqrt(args.max_mul)
# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(args.seed)
else:
	print('[CUDA unavailable]'); sys.exit()
# Args -- Experiment
if args.experiment == 'mnist2':
	from dataloaders import mnist2 as dataloader
elif args.experiment == 'pmnist' or args.experiment == 'split_pmnist':
	from dataloaders import pmnist as dataloader
elif args.experiment == 'row_pmnist' or args.experiment == 'split_row_pmnist':
	from dataloaders import row_pmnist as dataloader
elif args.experiment == 'split_mnist':
	from dataloaders import split_mnist as dataloader
elif args.experiment == 'split_notmnist':
	from dataloaders import split_notmnist as dataloader
elif args.experiment == 'split_cifar10':
	from dataloaders import split_cifar10 as dataloader
elif args.experiment == 'split_cifar100' or args.experiment == 'split_cifar100_big':
	from dataloaders import split_cifar100 as dataloader
elif args.experiment == 'split_cifar100_20':
	from dataloaders import cifar100_dataset as dataloader
elif args.experiment == 'split_cifar10_100' or args.experiment == 'split_cifar10_100_big':
	from dataloaders import split_cifar10_100 as dataloader
elif args.experiment == 'split_CUB200':
	from dataloaders import split_CUB200 as dataloader
elif args.experiment == 'split_tiny_imagenet':
	from dataloaders import split_tiny_imagenet as dataloader
elif args.experiment == 'split_mini_imagenet':
	from dataloaders import split_mini_imagenet as dataloader
elif args.experiment == 'omniglot':
	from dataloaders import split_omniglot as dataloader
elif args.experiment == 'mixture':
	from dataloaders import mixture as dataloader
else:
	from dataloaders import single_task as dataloader

# Args -- Approach
if args.approach == 'random':
	from approaches import random as approach
elif args.approach == 'ucl':
	from approaches import ucl as approach
elif args.approach == 'ucl_ablation':
	from approaches import ucl_ablation as approach
elif args.approach == 'baye_hat':
	from core import baye_hat as approach
elif args.approach == 'baye_fisher':
	from core import baye_fisher as approach
elif args.approach == 'sgd':
	from approaches import sgd as approach
elif args.approach == 'sgd-restart':
	from approaches import sgd_restart as approach
elif args.approach == 'sgd-frozen':
	from approaches import sgd_frozen as approach
elif args.approach == 'sgd_with_log':
	from approaches import sgd_with_log as approach
elif args.approach == 'sgd_L2_with_log':
	from approaches import sgd_L2_with_log as approach
elif args.approach == 'lwf':
	from approaches import lwf as approach
elif args.approach == 'lwf_with_log':
	from approaches import lwf_with_log as approach
elif args.approach == 'lfl':
	from approaches import lfl as approach
elif args.approach == 'ewc':
	from approaches import ewc as approach
elif args.approach == 'si':
	from approaches import si as approach
elif args.approach == 'rwalk':
	from approaches import rwalk as approach
elif args.approach == 'mas':
	from approaches import mas as approach
elif args.approach == 'imm-mean':
	from approaches import imm_mean as approach
elif args.approach == 'imm-mode':
	from approaches import imm_mode as approach
elif args.approach == 'progressive':
	from approaches import progressive as approach
elif args.approach == 'pathnet':
	from approaches import pathnet as approach
elif args.approach == 'hat-test':
	from approaches import hat_test as approach
elif args.approach == 'hat':
	from approaches import hat as approach
elif args.approach == 'joint':
	from approaches import joint as approach
elif args.approach == 'gs':
	from approaches import gs as approach
elif args.approach == 'sccl':
	from approaches import SCCL as approach
elif args.approach == 'baseline':
	from approaches import simple as approach
elif args.approach == 'finetuning':
	from approaches import simple as approach
elif args.approach == 'hat_expand':
	from approaches import hat_expand as approach

# Args -- Network

if args.arch == 'vgg8':
	if args.approach == 'hat' or args.approach == 'hat_expand':
		from networks import conv_net_hat as network
	elif args.approach == 'ucl' or args.approach == 'ucl_ablation':
		from networks import conv_net_ucl as network
	elif args.approach == 'gs':
		from networks import conv_net_gs as network
	elif args.approach == 'sccl':
		from networks.SCCL_net import DynamicCNN_CIFAR as Net
	else:
		from networks import conv_net as network

elif args.arch == 'alexnet':
	if args.approach == 'hat' or args.approach == 'hat_expand':
		from networks import alexnet_hat as network
	elif args.approach == 'ucl' or args.approach == 'ucl_ablation':
		from networks import alexnet_ucl as network
	elif args.approach == 'gs':
		from networks import alexnet_gs as network
	elif args.approach == 'sccl':
		from networks.SCCL_net import DynamicAlexnet as Net
	else:
		from networks import alexnet as network

elif args.arch == 'mlp':
	if args.approach == 'hat' or args.approach == 'hat_expand':
		from networks import mlp_hat as network
	elif args.approach == 'ucl' or args.approach == 'ucl_ablation':
		from networks import mlp_ucl as network
	elif args.approach == 'gs':
		from networks import mlp_gs as network
	elif args.approach == 'sccl':
		from networks.SCCL_net import DynamicMLP as Net
	else:
		from networks import mlp as network


elif args.experiment == 'split_cifar100_big' or args.experiment == 'split_cifar10_100_big' or args.experiment == 'split_cifar100_20' or 'split_cifar100_big' in args.experiment or 'split_cifar10_100_big' in args.experiment:
	if args.approach == 'hat' or args.approach == 'hat_expand':
		from networks import conv_net_hat as network
	elif args.approach == 'ucl' or args.approach == 'ucl_ablation':
		from networks import conv_net_ucl as network
	elif args.approach == 'gs':
		from networks import conv_net_gs as network
	elif args.approach == 'sccl':
		from networks.SCCL_net import DynamicVGG as Net
	else:
		from networks import conv_net as network

elif args.experiment == 'split_cifar100' or args.experiment == 'split_cifar10_100' or 'split_cifar100' in args.experiment or 'cifar10' in args.experiment:
	if args.approach == 'hat' or args.approach == 'hat_expand':
		from networks import conv_net_hat as network
	elif args.approach == 'ucl' or args.approach == 'ucl_ablation':
		from networks import conv_net_ucl as network
	elif args.approach == 'gs':
		from networks import conv_net_gs as network
	elif args.approach == 'sccl':
		from networks.SCCL_net import DynamicCNN_CIFAR as Net
	else:
		from networks import conv_net as network

elif args.experiment == 'mixture' or 'mixture' in args.experiment or args.experiment == 'split_mini_imagenet' or args.experiment == 'split_tiny_imagenet' or args.experiment == 'split_CUB200':
	if args.approach == 'hat' or args.approach == 'hat_expand':
		from networks import alexnet_hat as network
	elif args.approach == 'ucl' or args.approach == 'ucl_ablation':
		from networks import alexnet_ucl as network
	elif args.approach == 'gs':
		from networks import alexnet_gs as network
	elif args.approach == 'sccl':
		from networks.SCCL_net import DynamicAlexnet as Net
	else:
		from networks import alexnet as network

elif args.experiment == 'omniglot':
	if args.approach == 'hat' or args.approach == 'hat_expand':
		from networks import conv_net_omniglot_hat as network
	elif args.approach == 'ucl' or args.approach == 'ucl_ablation':
		from networks import conv_net_omniglot_ucl as network
	elif args.approach == 'gs':
		from networks import conv_net_omniglot_gs as network
	elif args.approach == 'sccl':
		from networks.SCCL_net import DynamicCNN_Omniglot as Net
	else:
		from networks import conv_net_omniglot as network
elif args.experiment == 'pmnist' or args.experiment == 'split_mnist':
	if args.approach == 'hat' or args.approach == 'hat_expand':
		from networks import mlp_hat as network
	elif args.approach == 'ucl' or args.approach == 'ucl_ablation':
		from networks import mlp_ucl as network
	elif args.approach == 'gs':
		from networks import mlp_gs as network
	elif args.approach == 'sccl':
		from networks.SCCL_net import DynamicMLP as Net
	else:
		from networks import mlp as network
	

########################################################################################################################
# Load
print('Load data...')

try:
	data, taskcla, inputsize, ids = dataloader.get(seed=args.seed, tasknum=args.tasknum)
except:
	data, taskcla, inputsize = dataloader.get(seed=args.seed, name=args.experiment)
print('Input size =', inputsize, '\nTask info =', taskcla)
# Inits
print('Inits...')
# print (inputsize,taskcla)
torch.set_default_tensor_type('torch.cuda.FloatTensor')


if args.approach == 'ucl' or args.approach == 'ucl_ablation':
	net = network.Net(inputsize, taskcla, args.ratio, mul=args.mul).cuda()
		# net_old = network.Net(inputsize, taskcla, args.ratio).cuda()
	appr = approach.Appr(net, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name, split=split)
elif args.approach == 'sccl':
	net = Net(inputsize, mul=args.mul)
	appr = approach.Appr(net, lamb=args.lamb, thres=args.thres, lr=args.lr, sbatch=args.batch_size, nepochs=args.nepochs, log_name=log_name, optim=args.optimizer)
elif args.approach == 'gs':
	net = network.Net(inputsize, taskcla, mul=args.mul).cuda()
	appr = approach.Appr(net, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name)
else:
	net = network.Net(inputsize, taskcla, mul=args.mul).cuda()
		# net_old = network.Net(inputsize, taskcla).cuda()
	appr = approach.Appr(net, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name, split=split)

	
print(utils.print_model_report(net))

# print(appr.ce)
utils.print_optimizer_config(appr.optimizer)
print('-' * 100)
past_ncla = [ncla for t, ncla in taskcla]

max_params = 0

if args.max_params > 0 and args.approach == 'sccl':
	max_bound = list(appr.model.bound) + [sum(past_ncla)]
	add_in = 0
	for i, m in enumerate(appr.model.DM):
		add_out = int(max_bound[i])
		max_params += m.num_add(add_in, add_out)
		if isinstance(m, DynamicConv2D) and isinstance(appr.model.DM[i+1], DynamicLinear):
			add_in = appr.model.smid * appr.model.smid * add_out
		else:
			add_in = add_out

	# print('SCCL with limited training parameters')
	# print('max_bound', max_bound)
	max_params = max_params*args.max_params
	print('Max params', max_params)

# if (not args.max_params) and args.max_mul == 1.0:
# 	print('SCCL with fixed network capacity')
# if (not args.max_params) and args.max_mul == 0.0:
# 	print('SCCL with unlimited training parameters')

# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)

start_task = 0
check_point = {'model': appr.model, 'phase':3}
if args.resume:
	for t in range(args.tasknum, 0, -1):
		try:
			log_name = '{}_{}_{}_{}_lamb_{}_mul_{}_max_mul_{}_max_params_{}_lr_{}_batch_{}_cil_{}'.format(args.date, args.experiment, args.approach, args.seed,
																			'_'.join([str(lamb) for lamb in lambs[:t]]), 
																			args.mul, args.max_mul, args.max_params, args.lr, args.batch_size, args.cil)
			acc = np.loadtxt(f'../result_data/{log_name}.txt')
			if len(acc) == 0:
				acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
			check_point = torch.load(f'../result_data/trained_model/{log_name}.model')
			appr.model = check_point['model']
			if check_point['phase'] == 3:
				start_task = t
			else:
				start_task = t-1
			print('Resume from task', start_task)

			break
		except:
			continue

	if args.shink >= 0:
		print(f'Shink to task {args.shink}')
		task = args.shink + 1
		shink_name = '{}_{}_{}_{}_lamb_{}_'.format(args.date, args.experiment, args.approach, args.seed, '_'.join([str(lamb) for lamb in lambs[:task]]))
		setting_name = f'_mul_{args.mul}_max_mul_{args.max_mul}_max_params_{args.max_params}_lr_{args.lr}_batch_{args.batch_size}_cil_{args.cil}'
		print('find', shink_name)

		for file_name in os.listdir('../result_data/trained_model/'):
			if shink_name in file_name and setting_name in file_name:
				log_name = file_name.replace('.model', '')
				print('log name', log_name)
				check_point = torch.load(f'../result_data/trained_model/{log_name}.model')
				acc = np.loadtxt(f'../result_data/{log_name}.txt')
				if len(acc) == 0:
					acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
				appr.model = check_point['model']
				start_task = args.shink + 1
				break
		for m in appr.model.DM:
			m.shape_out = m.shape_out[:task+1]
			m.shape_in = m.shape_in[:task+1]

			m.weight.data = m.weight.data[:m.shape_out[task]][:,:m.shape_in[task]].clone()
			m.bias.data = m.bias.data[:m.shape_out[task]].clone()

			m.weight.grad = None
			m.bias.grad = None

			m.in_features, m.out_features = m.weight.shape[1], m.weight.shape[0]

			print(f'shape in: {m.shape_in}, shape out: {m.shape_out}')
			if m.batch_norm is not None:
				m.batch_norm = m.batch_norm[:task+1]

			m.mask_pre = m.mask_pre[:t]

		check_point = {'model': appr.model, 'phase':3}
		log_name = '{}_{}_{}_{}_lamb_{}_mul_{}_max_mul_{}_max_params_{}_lr_{}_batch_{}_cil_{}'.format(args.date, args.experiment, args.approach, args.seed,
																			'_'.join([str(lamb) for lamb in lambs[:args.shink+1]]), 
																			args.mul, args.max_mul, args.max_params, args.lr, args.batch_size, args.cil)
		torch.save(check_point, f'../result_data/trained_model/{log_name}.model')
		np.savetxt(f'../result_data/{log_name}.txt', acc, '%.4f')

	# for m in appr.model.DM:
	# 	print(m.shape_out)
	try:
		for u in range(start_task):
			xtest = data[u]['test']['x'].cuda()
			ytest = data[u]['test']['y'].cuda()
			if args.approach == 'sccl':
				if args.cil:
					ytest += sum(past_ncla[:u])
					test_loss, test_acc = appr.eval(None, xtest, ytest)
				else:
					test_loss, test_acc = appr.eval(u+1, xtest, ytest)
			else:
				test_loss, test_acc = appr.eval(u, xtest, ytest)
			print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.2f}% <<<'.format(u, data[u]['name'], test_loss,
																						  100 * test_acc))
			acc[start_task-1, u] = test_acc
			lss[start_task-1, u] = test_loss

		print('Avg acc={:5.2f}%'.format(100*sum(acc[start_task-1])/(start_task)))
	except:
		pass




for t, ncla in taskcla[start_task:]:
	if t >= args.tasknum: break
	print('*' * 100)
	print('Task {:2d} ({:s})'.format(t, data[t]['name']))
	print('*' * 100)

	if args.approach == 'joint':
		# Get data. We do not put it to GPU
		if t == 0:
			xtrain = data[t]['train']['x']
			ytrain = data[t]['train']['y']
			xvalid = data[t]['valid']['x']
			yvalid = data[t]['valid']['y']
			task_t = t * torch.ones(xtrain.size(0)).int()
			task_v = t * torch.ones(xvalid.size(0)).int()
			task = [task_t, task_v]
		else:
			xtrain = torch.cat((xtrain, data[t]['train']['x']))
			ytrain = torch.cat((ytrain, data[t]['train']['y']))
			xvalid = torch.cat((xvalid, data[t]['valid']['x']))
			yvalid = torch.cat((yvalid, data[t]['valid']['y']))
			task_t = torch.cat((task_t, t * torch.ones(data[t]['train']['y'].size(0)).int()))
			task_v = torch.cat((task_v, t * torch.ones(data[t]['valid']['y'].size(0)).int()))
			task = [task_t, task_v]
	else:
		# Get data
		xtrain = data[t]['train']['x'].cuda()
		xvalid = data[t]['valid']['x'].cuda()
			
		ytrain = data[t]['train']['y'].cuda()
		yvalid = data[t]['valid']['y'].cuda()
		task = t

		# xtrain = torch.cat([data[i]['train']['x'].cuda() for i in range(t+1)], dim=0)
		# xvalid = torch.cat([data[i]['valid']['x'].cuda() for i in range(t+1)], dim=0)

		# ytrain = torch.cat([data[i]['train']['y'].cuda() for i in range(t+1)], dim=0)
		# yvalid = torch.cat([data[i]['valid']['y'].cuda() for i in range(t+1)], dim=0)
			
		# ytrain = torch.cat([data[i]['train']['y'].cuda() + sum(past_ncla[:i]) for i in range(t+1)], dim=0)
		# yvalid = torch.cat([data[i]['valid']['y'].cuda() + sum(past_ncla[:i]) for i in range(t+1)], dim=0)

		# xtrain = [data[i]['train']['x'].cuda() for '3i in range(t+1)], dim=0
		# xvalid = [data[i]['valid']['x'].cuda() for i in range(t+1)], dim=0

		# ytrain = [data[i]['train']['y'].cuda() for i in range(t+1)], dim=0
		# yvalid = [data[i]['valid']['y'].cuda() for i in range(t+1)], dim=0

		# if args.cil:
		# 	print(sum(past_ncla[:t]))
		# 	ytrain += sum(past_ncla[:t])
		# 	yvalid += sum(past_ncla[:t])
		# 	print(yvalid)

	# Train
	if args.experiment == 'split_cifar10_100' or args.experiment == 'split_cifar10_100_big':
		if 'cifar100' in data[t]['name']:
			appr.sbatch = 32
		else:
			appr.sbatch = 256
	if args.approach =='baseline':
		appr.model = network.Net(inputsize, taskcla, mul=args.mul).cuda()

	if args.approach == 'sccl':

		try:
			os.remove(f'../result_data/{log_name}.txt')
			os.remove(f'../result_data/trained_model/{log_name}.model')
		except:
			pass
		log_name = '{}_{}_{}_{}_lamb_{}_mul_{}_max_mul_{}_max_params_{}_lr_{}_batch_{}_cil_{}'.format(args.date, args.experiment, args.approach, args.seed,
																			'_'.join([str(lamb) for lamb in lambs[:t+1]]), 
																			args.mul, args.max_mul, args.max_params, args.lr, args.batch_size, args.cil)
		np.savetxt(f'../result_data/{log_name}.txt', acc, '%.4f')
		torch.save(check_point, f'../result_data/trained_model/{log_name}.model')
		appr.file_name = log_name

		appr.lamb = lambs[t]


		print('lambda', appr.lamb)
		# we could use only train data for better performance, 
		# but for a fair comparison with other baslines we use the same train, valid split

		# if xtrain.shape != xvalid.shape:
		# 	xtrain = torch.cat([xtrain, xvalid], dim=0)
		# 	ytrain = torch.cat([ytrain, yvalid], dim=0)
		print('Train data:', xtrain.shape)
		if args.cil:
			check_point = appr.train(None, xtrain, ytrain, xvalid, yvalid, ncla=ncla, max_mul=args.max_mul, max_params=max_params, check_point=check_point, tasknum=args.tasknum)
		else:
			check_point = appr.train(task+1, xtrain, ytrain, xvalid, yvalid, ncla=ncla, max_mul=args.max_mul, max_params=max_params, check_point=check_point, tasknum=args.tasknum)
	else:
		appr.train(task, xtrain, ytrain, xvalid, yvalid, data, inputsize, taskcla)
	print('-' * 100)

	# Test
	for u in range(t + 1):
		xtest = data[u]['test']['x'].cuda()
		ytest = data[u]['test']['y'].cuda()
		if args.approach == 'sccl':
			if args.cil:
				ytest += sum(past_ncla[:u])
				test_loss, test_acc = appr.eval(None, xtest, ytest)
				# print('>>> Test on task {:2d} - CIL - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss, 100 * test_acc))
			else:
				test_loss, test_acc = appr.eval(u+1, xtest, ytest)
		else:
			test_loss, test_acc = appr.eval(u, xtest, ytest)
		print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.2f}% <<<'.format(u, data[u]['name'], test_loss, 100 * test_acc))
		acc[t, u] = test_acc
		lss[t, u] = test_loss

	print('Avg acc={:5.2f}%'.format(100*sum(acc[t])/(t+1)))

	# Save
	if args.approach == 'sccl':
		print('Save at ' + f'../result_data/{log_name}.txt')
		np.savetxt(f'../result_data/{log_name}.txt', acc, '%.4f')
	else:
		print('Save at ' + args.output)
		np.savetxt(args.output, acc, '%.4f')
	if args.approach != 'sccl':
	    torch.save(appr.model, '../result_data/trained_model/' + log_name + '.model')


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

if hasattr(appr, 'logs'):
	if appr.logs is not None:
		# save task names
		from copy import deepcopy

		appr.logs['task_name'] = {}
		appr.logs['test_acc'] = {}
		appr.logs['test_loss'] = {}
		for t, ncla in taskcla:
			appr.logs['task_name'][t] = deepcopy(data[t]['name'])
			appr.logs['test_acc'][t] = deepcopy(acc[t, :])
			appr.logs['test_loss'][t] = deepcopy(lss[t, :])
		# pickle
		import gzip
		import pickle

		with gzip.open(os.path.join(appr.logpath), 'wb') as output:
			pickle.dump(appr.logs, output, pickle.HIGHEST_PROTOCOL)

########################################################################################################################

