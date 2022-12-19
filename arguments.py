import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Continual')
    # Arguments
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--experiment', default='', type=str, required=True,
                        # choices=['mnist2', 
                        #          'pmnist', 
                        #          'split_pmnist', 
                        #          'row_pmnist', 
                        #          'mixture', 
                        #          'omniglot',
                        #          'split_cifar100_big',
                        #          'split_cifar10_100_big',
                        #          'split_mnist',
                        #          'split_notmnist', 
                        #          'split_row_pmnist', 
                        #          'split_cifar10_100', 
                        #          'split_cifar100',
                        #          'split_cifar100_20',
                        #          'split_CUB200', 
                        #          'split_tiny_imagenet', 
                        #          'split_mini_imagenet', 
                        #          'split_cifar10'], 
                        help='(default=%(default)s)')
    parser.add_argument('--approach', default='', type=str, required=True,
                        # choices=['random', 
                        #          'sgd', 
                        #          'sgd-frozen', 
                        #          'sgd_with_log', 
                        #          'sgd_L2_with_log', 
                        #          'lwf','lwf_with_log', 
                        #          'lfl',
                        #          'ewc', 
                        #          'si', 
                        #          'rwalk', 
                        #          'mas', 
                        #          'ucl', 
                        #          'ucl_ablation', 
                        #          'baye_fisher',
                        #          'baye_hat', 
                        #          'imm-mean', 
                        #          'progressive', 
                        #          'pathnet',
                        #          'imm-mode', 
                        #          'sgd-restart', 
                        #          'joint', 
                        #          'hat', 
                        #          'hat-test',
                        #          'gs',
                        #          'sccl',
                        #          'finetuning',
                        #          'baseline',
                        #          'hat_expand',
                        #          'simple'], 
                        help='(default=%(default)s)')
    parser.add_argument('--optimizer', default='Adam', type=str, required=False,
                        choices=['SGD', 
                                 'SGD_momentum_decay', 
                                 'Adam'], 
                        help='(default=%(default)s)')
    parser.add_argument('--ablation', default='full', type=str, required=False,
                        # choices=['no_ensemble',
                        #         'no_scale',
                        #         'no_scale_ensemble',
                        #         'full'], 
                        help='(default=%(default)s)')
    parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--nepochs', default=100, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--mul', default=1.0, type=float, required=False, help='(default=%(default)d)')
    parser.add_argument('--thres', default=0, type=float, required=False, help='(default=%(default)d)')
    parser.add_argument('--batch_size', default=256, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--val_batch_size', default=256, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.001, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--lr_patience', default=5, type=int, required=False, help='(default=%(default)f)')
    parser.add_argument('--lr_factor', default=3, type=int, required=False, help='(default=%(default)f)')
    parser.add_argument('--lr_rho', default=0, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--ratio', default=1.0, type=float, help='(default=%(default)f)')
    parser.add_argument('--alpha', default=0.01, type=float, help='(default=%(default)f)')
    parser.add_argument('--beta', default=0.03, type=float, help='(default=%(default)f)')
    parser.add_argument('--gamma', default=0.75, type=float, help='(default=%(default)f)')
    parser.add_argument('--ensemble_drop', default=0, type=float, help='(default=%(default)f)')
    parser.add_argument('--smax', default=400, type=float, help='(default=%(default)f)')
    parser.add_argument('--lamb', default='0.0', type=str, help='(default=%(default)f)')
    parser.add_argument('--threshold', default='0.0', type=str, help='(default=%(default)f)')
    parser.add_argument('--c', default='0.9', type=float, help='(default=%(default)f)')
    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--tasknum', default=10, type=int, help='(default=%(default)s)')
    parser.add_argument('--mixture_task', default=None, type=int, help='(default=%(default)s)')
    parser.add_argument('--start_task', default=0, type=int, help='(default=%(default)s)')
    parser.add_argument('--conv-net', action='store_true', default=False, help='Using convolution network')
    parser.add_argument('--rebuttal', action='store_true', default=False, help='Using convolution network')
    parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
    parser.add_argument('--sample', type = int, default=1, help='Using sigma max to split_CUB200                                                                                                                                                        port coefficient')
    parser.add_argument('--rho', type = float, default=-2.783, help='initial rho')
    parser.add_argument('--nu', default=0.1, type=float, help='(default=%(default)f)')
    parser.add_argument('--mu', default=0, type=float, help='groupsparse parameter')
    parser.add_argument('--eta', default=0.8, type=float, help='(default=%(default)f)')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume from check point')
    parser.add_argument('--max_mul', default=0.0, type=float, help='max number of neurons to expand')
    parser.add_argument('--cil', default=False, action='store_true', help='(default=%(default)s)')
    parser.add_argument('--fix', default=False, action='store_true', help='(default=%(default)s)')
    parser.add_argument('--shink', default=-1, type=int, help='delete network of recent tasks, shink back to given task')
    parser.add_argument('--max_params', default=0.0, type=float, help='max number of parameters of SCCL')
    parser.add_argument('--arch', default=None, type=str, help='Architecture')
    parser.add_argument('--min_ratio', default=-1.0, type=float, required=False, help='min prune ratio')
    parser.add_argument('--norm_type', default=None, type=str, required=False, help='normalization layer type')
    parser.add_argument('--affine', default=False, action='store_true', help='(default=%(default)s)')
    parser.add_argument('--augment', default=False, action='store_true', help='(default=%(default)s)')
    parser.add_argument('--prune_method', type=str, default='pgd', help='(default=%(default)s)')
    parser.add_argument('--dropout_method', type=str, default='pcdrop', help='(default=%(default)s)')
    parser.add_argument('--sparsity', default=1000, type=float, required=False, help='(default=%(default)s)')
    parser.add_argument('--factor', default=10, type=float, required=False, help='(default=%(default)s)')
    
    

    args=parser.parse_args()
    return args