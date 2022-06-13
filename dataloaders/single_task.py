import os,sys
import os.path
import numpy as np
import torch
import torch.utils.data
from torchvision import datasets,transforms
from sklearn.utils import shuffle
import urllib.request
from PIL import Image
import pickle
import utils
from torch.utils.data import  TensorDataset, DataLoader
########################################################################################################################

def get(name, batch_size, val_batch_size, seed=0):
    data={}
    taskcla=[]
    size=[3,32,32]

    if 'split_cifar100' in name:
        i = name.split('_')[-1]
        data[0] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[0][s]={'x':[],'y':[]}
            data[0][s]['x']=torch.load(os.path.join(os.path.expanduser('../dat/binary_split_cifar100'),
                                                    'data'+i+s+'x.bin'))
            data[0][s]['y']=torch.load(os.path.join(os.path.expanduser('../dat/binary_split_cifar100'),
                                                    'data'+i+s+'y.bin'))

        data[0]['ncla']=len(np.unique(data[0]['train']['y'].numpy()))
        data[0]['name']='cifar100-'+i
        r=np.arange(data[0]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(0.10*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[0]['valid']={}
        data[0]['valid']['x']=data[0]['train']['x'][ivalid].clone()
        data[0]['valid']['y']=data[0]['train']['y'][ivalid].clone()
        data[0]['train']['x']=data[0]['train']['x'][itrain].clone()
        data[0]['train']['y']=data[0]['train']['y'][itrain].clone()
    elif 'cifar10' in name:
        data[0] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[0][s]={'x':[],'y':[]}
            data[0][s]['x']=torch.load(os.path.join(os.path.expanduser('../dat/binary_cifar10'),'data'+s+'x.bin'))
            data[0][s]['y']=torch.load(os.path.join(os.path.expanduser('../dat/binary_cifar10'),'data'+s+'y.bin'))
        data[0]['ncla']=len(np.unique(data[0]['train']['y'].numpy()))
        data[0]['name']='cifar10'
        r=np.arange(data[0]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(0.10*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[0]['valid']={}
        data[0]['valid']['x']=data[0]['train']['x'][ivalid].clone()
        data[0]['valid']['y']=data[0]['train']['y'][ivalid].clone()
        data[0]['train']['x']=data[0]['train']['x'][itrain].clone()
        data[0]['train']['y']=data[0]['train']['y'][itrain].clone()
    elif 'mixture' in name:
        idx = (name.split('_')[-1])
        data[0] = dict.fromkeys(['name','ncla','train','test'])
        if idx=='0':
            data[0]['name']='cifar10'
            data[0]['ncla']=10
        elif idx=='1':
            data[0]['name']='cifar100'
            data[0]['ncla']=100
        elif idx=='2':
            data[0]['name']='mnist'
            data[0]['ncla']=10
        elif idx=='3':
            data[0]['name']='svhn'
            data[0]['ncla']=10
        elif idx=='4':
            data[0]['name']='fashion-mnist'
            data[0]['ncla']=10
        elif idx=='5':
            data[0]['name']='traffic-signs'
            data[0]['ncla']=43
        elif idx=='6':
            data[0]['name']='facescrub'
            data[0]['ncla']=100
        elif idx=='7':
            data[0]['name']='notmnist'
            data[0]['ncla']=10
        else:
            print('ERROR: Undefined data set',idx)
            sys.exit()

        # Load
        for s in ['train','test']:
            data[0][s]={'x':[],'y':[]}
            data[0][s]['x'] = torch.load(os.path.join(os.path.expanduser('../dat/binary_mixture'),'data'+str(idx)+s+'x.bin'))
            data[0][s]['y'] = torch.load(os.path.join(os.path.expanduser('../dat/binary_mixture'),'data'+str(idx)+s+'y.bin'))
        r=np.arange(data[0]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(0.15*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[0]['valid']={}
        data[0]['valid']['x']=data[0]['train']['x'][ivalid].clone()
        data[0]['valid']['y']=data[0]['train']['y'][ivalid].clone()
        data[0]['train']['x']=data[0]['train']['x'][itrain].clone()
        data[0]['train']['y']=data[0]['train']['y'][itrain].clone()

        data[0]['train loader'] = DataLoader(
                    TensorDataset(data[0]['train']['x'], data[0]['train']['y']) , batch_size=batch_size, shuffle=True
                )

        data[0]['valid loader'] = DataLoader(
                    TensorDataset(data[0]['valid']['x'], data[0]['valid']['y']) , batch_size=val_batch_size, shuffle=False
                )

        data[0]['test loader'] = DataLoader(
                    TensorDataset(data[0]['test']['x'], data[0]['test']['y']) , batch_size=val_batch_size, shuffle=False
                )

    
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data, taskcla, size

########################################################################################################################

class FashionMNIST(datasets.MNIST):
    """`Fashion MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.
    """
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]

########################################################################################################################

class TrafficSigns(torch.utils.data.Dataset):
    """`German Traffic Signs <http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory ``Traffic signs`` exists.
        split (string): One of {'train', 'test'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
    """

    def __init__(self, root, train=True,transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.filename = "traffic_signs_dataset.zip"
        self.url = "https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d53ce_traffic-sign-data/traffic-sign-data.zip"
        # Other options for the same 32x32 pickled dataset
        # url="https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip"
        # url_train="https://drive.google.com/open?id=0B5WIzrIVeL0WR1dsTC1FdWEtWFE"
        # url_test="https://drive.google.com/open?id=0B5WIzrIVeL0WLTlPNlR2RG95S3c"

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                self.download()

        training_file = 'lab 2 data/train.p'
        testing_file = 'lab 2 data/test.p'
        if train:
            with open(os.path.join(root,training_file), mode='rb') as f:
                train = pickle.load(f)
            self.data = train['features']
            self.labels = train['labels']
        else:
            with open(os.path.join(root,testing_file), mode='rb') as f:
                test = pickle.load(f)
            self.data = test['features']
            self.labels = test['labels']

        self.data = np.transpose(self.data, (0, 3, 1, 2))
        #print(self.data.shape); sys.exit()

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        import errno
        root = os.path.expanduser(self.root)
        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urllib.request.urlretrieve(self.url, fpath)
        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()


########################################################################################################################

class Facescrub(torch.utils.data.Dataset):
    """Subset of the Facescrub cropped from the official Megaface challenge page: http://megaface.cs.washington.edu/participate/challenge.html, resized to 38x38
    Args:
        root (string): Root directory of dataset where directory ``Traffic signs`` exists.
        split (string): One of {'train', 'test'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
    """

    def __init__(self, root, train=True,transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.filename = "facescrub_100.zip"
        self.url = "https://github.com/nkundiushuti/facescrub_subset/blob/master/data/facescrub_100.zip?raw=true"

        fpath=os.path.join(root,self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                self.download()

        training_file = 'facescrub_train_100.pkl'
        testing_file = 'facescrub_test_100.pkl'
        if train:
            with open(os.path.join(root,training_file),'rb') as f:
                # u = pickle._Unpickler(f)
                # u.encoding = 'latin1'
                # train  = u.load()
                train = pickle.load(f)
            self.data = train['features'].astype(np.uint8)
            self.labels = train['labels'].astype(np.uint8)
            """
            print(self.data.shape)
            print(self.data.mean())
            print(self.data.std())
            print(self.labels.max())
            #"""
        else:
            with open(os.path.join(root,testing_file),'rb') as f:
                # u = pickle._Unpickler(f)
                # u.encoding = 'latin1'
                # test  = u.load()
                test = pickle.load(f)

            self.data = test['features'].astype(np.uint8)
            self.labels = test['labels'].astype(np.uint8)

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        import errno
        root = os.path.expanduser(self.root)

        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urllib.request.urlretrieve(self.url, fpath)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()


########################################################################################################################

class notMNIST(torch.utils.data.Dataset):
    """The notMNIST dataset is a image recognition dataset of font glypyhs for the letters A through J useful with simple neural networks. It is quite similar to the classic MNIST dataset of handwritten digits 0 through 9.
    Args:
        root (string): Root directory of dataset where directory ``Traffic signs`` exists.
        split (string): One of {'train', 'test'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
    """

    def __init__(self, root, train=True,transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.filename = "notmnist.zip"
        self.url = "https://github.com/nkundiushuti/notmnist_convert/blob/master/notmnist.zip?raw=true"

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                self.download()

        training_file = 'notmnist_train.pkl'
        testing_file = 'notmnist_test.pkl'
        if train:
            with open(os.path.join(root,training_file),'rb') as f:
                # u = pickle._Unpickler(f)
                # u.encoding = 'latin1'
                # train  = u.load()
                train = pickle.load(f)
            self.data = train['features'].astype(np.uint8)
            self.labels = train['labels'].astype(np.uint8)
        else:
            with open(os.path.join(root,testing_file),'rb') as f:
                # u = pickle._Unpickler(f)
                # u.encoding = 'latin1'
                # test  = u.load()
                test = pickle.load(f)

            self.data = test['features'].astype(np.uint8)
            self.labels = test['labels'].astype(np.uint8)


    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img[0])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        import errno
        root = os.path.expanduser(self.root)

        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urllib.request.urlretrieve(self.url, fpath)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()


########################################################################################################################