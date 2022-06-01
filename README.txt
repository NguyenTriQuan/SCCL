Structure Compression-based Continual Learning

dependencies:
- python			3.7.12
- matplotlib                    3.2.2
- numpy                         1.19.5
- torch                         1.10.0+cu111
- torchvision                   0.11.1+cu111

Reproduce results:

- Permuted MNIST:
python main.py --approach sccl --experiment pmnist --lamb 0.0025 --lr 0.001 --batch_size 256 --nepochs 100 --optimizer Adam --mul 1 --tasknum 10

- Split CIFAR-100:
python main.py --approach sccl --experiment split_cifar100 --lamb 0.015_0.05 --lr 0.001 --batch_size 32 --nepochs 100 --optimizer Adam --mul 1 --tasknum 10

- Split CIAFR-10/100:
python main.py --approach sccl --experiment split_cifar10_100 --lamb 0.01_0.05 --lr 0.001 --batch_size 32 --nepochs 100 --optimizer Adam --mul 1 --tasknum 11

- Mixture:
python main.py --approach sccl --experiment mixture --lamb 0.008_0.02 --lr 0.05 --batch_size 256 --nepochs 150 --optimizer SGD --mul 1 --tasknum 8

More:

- different lambda for each task:
python main.py --approach sccl --experiment mixture --lamb 0.008_0.02_0.01_0.015_0.03_0.02_0.02_0.015 --lr 0.05 --batch_size 256 --nepochs 150 --optimizer SGD --mul 1 --tasknum 8

- SCCL-fix:
python main.py --approach sccl --experiment split_cifar_100 --lamb 0.015_0.05 --lr 0.001 --batch_size 32 --nepochs 100 --optimizer Adam --mul 1 --max_mul 1 --tasknum 10

- Continue training:
python main.py --approach sccl --experiment split_cifar_100 --lamb 0.015_0.05 --lr 0.001 --batch_size 32 --nepochs 100 --optimizer Adam --mul 1 --tasknum 10 --resume

- Continue training from task t if the current task >= t (e.g t=3):
python main.py --approach sccl --experiment split_cifar_100 --lamb 0.015_0.05 --lr 0.001 --batch_size 32 --nepochs 100 --optimizer Adam --mul 1 --tasknum 10 --resume --shink 3

- Limit maximum number of parameters can be expand by xN the number of parameters of the base network (e.g N=3):
python main.py --approach sccl --experiment split_cifar_100 --lamb 0.015_0.05 --lr 0.001 --batch_size 32 --nepochs 100 --optimizer Adam --mul 1 --max_params 3 --tasknum 10

- Limit maximum number of neurons can be expand by xN the number of neurons of the base network (e.g N=3):
python main.py --approach sccl --experiment split_cifar_100 --lamb 0.015_0.05 --lr 0.001 --batch_size 32 --nepochs 100 --optimizer Adam --mul 1 --max_mul 3 --tasknum 10