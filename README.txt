Dynamic Architecture and past connection Dropout (DAD)

Reproduce results:

- Permuted MNIST:
python main.py --approach dad --experiment pmnist --arch MLP --lamb 2.3_0 --batch_size 256 --lr 0.001 --optimizer Adam --tasknum 10  --nepochs 100 --ensemble_drop 0

- Split CIFAR-100:
python main.py --approach dad --experiment split_cifar100 --arch VGG8 --lamb 7_2_0.5 --batch_size 32 --lr 0.001 --optimizer Adam --tasknum 10  --nepochs 150 --ensemble_drop 0

- Split CIAFR-10/100:
python main.py --approach dad --experiment split_cifar10_100 --arch VGG8 --lamb 8_2_0.5 --batch_size 32 --lr 0.001 --optimizer Adam --tasknum 11  --nepochs 150 --ensemble_drop 0

- Mixture:
python main.py --approach dad --experiment mixture --arch Alexnet --lamb 30_0 --batch_size 256 --lr 0.01 --optimizer SGD --tasknum 8  --nepochs 150 --ensemble_drop 0

- 10 Split mini ImageNet:
python main.py --approach dad --experiment 10_split_mini_imagenet --arch VGG16_small --lamb 0.18_0.06_0 --batch_size 32 --lr 0.01 --optimizer SGD --tasknum 10  --nepochs 100 --ensemble_drop 0.2 --norm_type affine_track

- 20 split min ImageNet:
python main.py --approach dad --experiment 20_split_mini_imagenet --arch VGG16_small --lamb 0.3_0.1_0 --batch_size 32 --lr 0.01 --optimizer SGD --tasknum 10  --nepochs 100 --ensemble_drop 0.2 --norm_type affine_track
