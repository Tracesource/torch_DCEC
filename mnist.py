from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
from six.moves import cPickle as pickle



class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, small=False, full=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.full = full

        if full:
            self.train = True

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.train_data, self.train_labels = torch.load(os.path.join(self.root, self.processed_folder, self.training_file).replace('\\', '/'))
        self.test_data, self.test_labels = torch.load(os.path.join(self.root, self.processed_folder, self.test_file).replace('\\', '/'))

        if full:
            self.train_data = np.concatenate((self.train_data, self.test_data), axis=0)
            self.train_labels = np.concatenate((self.train_labels, self.test_labels), axis=0)

        if small:
            self.train_data = self.train_data[0:1400]
            self.train_labels = self.train_labels[0:1400]
            if not full:
                self.train_data = self.train_data[0:1200]
                self.train_labels = self.train_labels[0:1200]
            self.test_data = self.test_data[0:200]
            self.test_labels = self.test_labels[0:200]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.full:
            img = Image.fromarray(img, mode='L')
        else:
            img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file).replace('\\', '/')) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file).replace('\\', '/'))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        # try:
        #     os.makedirs(os.path.join(self.root, self.raw_folder))
        #     os.makedirs(os.path.join(self.root, self.processed_folder))
        # except OSError as e:
        #     if e.errno == errno.EEXIST:
        #         pass
        #     else:
        #         raise

        # for url in self.urls:
        #     print('Downloading ' + url)
        #     data = urllib.request.urlopen(url)
        #     filename = url.rpartition('/')[2]
        #     file_path = os.path.join(self.root, self.raw_folder, filename)
        #     with open(file_path, 'wb') as f:
        #         f.write(data.read())
        #     with open(file_path.replace('.gz', ''), 'wb') as out_f, \
        #             gzip.GzipFile(file_path) as zip_f:
        #         out_f.write(zip_f.read())
        #     os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images.idx3-ubyte').replace('\\', '/')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels.idx1-ubyte').replace('\\', '/'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images.idx3-ubyte').replace('\\', '/')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels.idx1-ubyte').replace('\\', '/'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file).replace('\\', '/'), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file).replace('\\', '/'), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class CIFAR10(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        # 训练集
        self.train_data,self.train_labels = self.load_CIFAR10_train()
        # 测试集
        self.test_data, self.test_labels = self.load_CIFAR10_test()

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img, mode='L')
        img = Image.fromarray(img, mode='RGB') #if mode in ["1", "L", "I", "P", "F"]:ndmax = 2elif mode == "RGB":ndmax = 3

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    
    def load_CIFAR10_train(self):
        xs = [] # list
        ys = []
        # 训练集batch 1～5
        for b in range(1,6):
            filename = os.path.join(self.root, 'data_batch_%d' % (b, ))
            with open(filename, 'rb') as f:
                datadict = pickle.load(f, encoding='latin1')   # dict类型
                X = datadict['data']        # X, ndarray, 像素值
                Y = datadict['labels']      # Y, list, 标签, 分类
                # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
                X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
                Y = np.array(Y)
                xs.append(X) # 在list尾部添加对象X, x = [..., [X]]
                ys.append(Y)  
                del X, Y  
        t_data = np.concatenate(xs) # [ndarray, ndarray] 合并为一个ndarray
        t_labels = np.concatenate(ys)
        return t_data,t_labels

    def load_CIFAR10_test(self):
        filename = os.path.join(self.root, 'test_batch')
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')   # dict类型
            X = datadict['data']        # X, ndarray, 像素值
            Y = datadict['labels']      # Y, list, 标签, 分类
            
            # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
            # transpose，转置
            # astype，复制，同时指定类型
            X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
            print("x shape:",X.shape)
            Y = np.array(Y)
            return X, Y


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)

