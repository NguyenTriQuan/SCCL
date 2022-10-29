Structure Compression-based Continual Learning

dependencies:
- python				  3.7.12
- matplotlib                    3.2.2
- numpy                         1.19.5
- torch                         1.10.0+cu111
- torchvision                   0.11.1+cu111

Reproduce results:

- Permuted MNIST:
python main.py --approach sccl --experiment pmnist --lamb 0.0025 --lr 0.001 --batch_size 256 --nepochs 100 --optimizer Adam --mul 1 --tasknum 10

- Split CIFAR-100:
python main.py --approach sccl --experiment split_cifar100 --arch VGG8 --lamb 6_7 --lr 0.001 --batch_size 32 --nepochs 100 --optimizer Adam --tasknum 10

- Split CIAFR-10/100:
python main.py --approach sccl --experiment split_cifar10_100 --lamb 0.01_0.05 --lr 0.001 --batch_size 32 --nepochs 100 --optimizer Adam --mul 1 --tasknum 11

- Mixture:
python main.py --approach sccl --experiment mixture --lamb 0.008_0.02 --lr 0.05 --batch_size 256 --nepochs 150 --optimizer SGD --mul 1 --tasknum 8

