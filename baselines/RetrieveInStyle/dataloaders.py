import os
import torchvision.transforms as tfs
import torch
import torch.utils.data
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms
#from .gaussian_blur import GaussianBlur
#from .gaussian_noise import GaussianNoise


def get_data_loaders(cfgs):
    batch_size = cfgs.get('batch_size', 64)
    num_workers = cfgs.get('num_workers', 4)
    image_size = cfgs.get('image_size', 64)
    contrastive = cfgs.get('contrastive', False)
    crop = cfgs.get('crop', None)

    run_train = cfgs.get('run_train', False)
    train_val_data_dir = cfgs.get('train_val_data_dir', './data')
    run_test = cfgs.get('run_test', False)
    test_data_dir = cfgs.get('test_data_dir', './data/test')

    load_gt_depth = cfgs.get('load_gt_depth', False)
    AB_dnames = cfgs.get('paired_data_dir_names', ['A', 'B'])
    AB_fnames = cfgs.get('paired_data_filename_diff', None)

    train_loader = val_loader = test_loader = None
    if load_gt_depth:
        get_loader = lambda **kargs: get_paired_image_loader(**kargs, batch_size=batch_size, image_size=image_size, crop=crop, AB_dnames=AB_dnames, AB_fnames=AB_fnames)
    else:
        get_loader = lambda **kargs: get_image_loader(**kargs, batch_size=batch_size, image_size=image_size, crop=crop)
        get_ctrain_loader = lambda **kargs: get_contrastive_loader(**kargs, batch_size=batch_size, image_size=image_size, crop=crop)

    if run_train:
        train_data_dir = os.path.join(train_val_data_dir, "train")
        val_data_dir = os.path.join(train_val_data_dir, "val")
        print(train_data_dir)
        assert os.path.isdir(train_data_dir), "Training data directory does not exist: %s" %train_data_dir
        assert os.path.isdir(val_data_dir), "Validation data directory does not exist: %s" %val_data_dir
        print(f"Loading training data from {train_data_dir}")
        if contrastive:
            train_loader = get_ctrain_loader(data_dir=train_data_dir, is_validation=False)
        else:
            fraction = cfgs.get('fraction', 1)
            train_loader = get_loader(data_dir=train_data_dir, is_validation=False, fraction=fraction)
        print(f"Loading validation data from {val_data_dir}")
        if contrastive:
            val_loader = get_ctrain_loader(data_dir=val_data_dir, is_validation=True)
        else:
            val_loader = get_loader(data_dir=val_data_dir, is_validation=True)
    if run_test:
        assert os.path.isdir(test_data_dir), "Testing data directory does not exist: %s" %test_data_dir
        print(f"Loading testing data from {test_data_dir}")
        test_loader = get_loader(data_dir=test_data_dir, is_validation=True)

    return train_loader, val_loader, test_loader


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp')
def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)


## simple image dataset ##
def make_dataset(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                fpath = os.path.join(root, fname)
                images.append(fpath)
    return images


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size=256, crop=None, is_validation=False, fraction=1):
        super(ImageDataset, self).__init__()
        self.root = data_dir
        self.paths = make_dataset(data_dir)
        self.size = len(self.paths)
        self.image_size = image_size
        self.crop = crop
        self.is_validation = is_validation

        fraction = int(fraction * self.size)
        self.paths = self.paths[:fraction]
        self.size = len(self.paths)

    def transform(self, img, hflip=False):
        if self.crop is not None:
            if isinstance(self.crop, int):
                img = tfs.CenterCrop(self.crop)(img)
            else:
                assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *self.crop)
        img = tfs.functional.resize(img, (self.image_size, self.image_size))
        if hflip:
            img = tfs.functional.hflip(img)
        return tfs.functional.to_tensor(img)

    def __getitem__(self, index):
        fpath = self.paths[index % self.size]
        img_name = fpath.split("/")[-1]
        img = Image.open(fpath).convert('RGB')
        hflip = not self.is_validation and np.random.rand()>0.5
        return self.transform(img, hflip=hflip), img_name

    def __len__(self):
        return self.size

    def name(self):
        return 'ImageDataset'

class ContrastiveImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size=256, crop=None, is_validation=False):
        super(ContrastiveImageDataset, self).__init__()
        self.root = data_dir
        self.paths = make_dataset(data_dir)
        self.size = len(self.paths)
        self.image_size = image_size
        self.crop = crop
        self.is_validation = is_validation

    def transform(self, img, hflip=False):
        if self.crop is not None:
            if isinstance(self.crop, int):
                img = tfs.CenterCrop(self.crop)(img)
            else:
                assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *self.crop)
        img = tfs.functional.resize(img, (self.image_size, self.image_size))
        if hflip:
            img = tfs.functional.hflip(img)
        return tfs.functional.to_tensor(img)

    def get_contrastive_transform(self, size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        crop = transforms.CenterCrop(0.8 * size)
        data_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([crop], p=0.8),
                                            transforms.Resize(size),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            #transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(kernel_size=int(0.1 * size)),
                                            transforms.ToTensor(), #])
                                            GaussianNoise(p=1)])
        return data_transforms

    def __getitem__(self, index):
        fpath = self.paths[index % self.size]
        img = Image.open(fpath).convert('RGB')

        # contrastive pair
        contrastive_transform = self.get_contrastive_transform(self.image_size)
        img_transform = tfs.functional.resize(img, (self.image_size, self.image_size))
        img_transform = contrastive_transform(img_transform)

        hflip = not self.is_validation and np.random.rand()>0.5
        img = self.transform(img, hflip=hflip)
        #from torchvision.utils import save_image
        #save_image(img, "img.png")
        #save_image(img_transform, "img_transform.png")
        #exit()
        return img, img_transform

    def __len__(self):
        return self.size

    def name(self):
        return 'ImageDataset'

class LabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size=64, crop=None, is_validation=False, celeb=False, celeb_id=False, limit=None):
        super(LabeledImageDataset, self).__init__()
        self.root = data_dir
        self.paths = make_dataset(data_dir)
        self.images = None
        if limit:
            self.paths = self.paths[:limit]
        self.size = len(self.paths)
        self.image_size = image_size
        self.crop = crop
        self.is_validation = is_validation
        self.celeb_labels = None
        self.celeb_ids = None
        data_dir = data_dir.split("/")[:-1]
        data_dir = "/".join(data_dir)
        if os.path.exists(os.path.join(data_dir, "list_attr_celeba.txt")) and celeb:
            self.celeb_labels = {}
            fp = open(os.path.join(data_dir, "list_attr_celeba.txt"))
            for line in fp:
                sline = line.replace("\n", "").split(" ")
                sline = [val for val in sline if val != " " and val != ""]
                label = [0 if x=="-1" else 1 for x in sline[1:]]
                self.celeb_labels[sline[0]] = np.array(label, dtype=np.float64)
            fp.close()
            self.images = {}
            print(len(self.paths))
            for i, path in enumerate(self.paths):
                print(i)
                img = Image.open(path).convert('RGB')
                self.images[path] = tfs.functional.resize(img, (self.image_size, self.image_size))
        if os.path.exists(os.path.join(data_dir, "identity_CelebA.txt")) and celeb_id:
            all_ids = {}
            fp = open(os.path.join(data_dir, "identity_CelebA.txt"))
            for line in fp:
                sline = line.replace("\n","").split(" ")
                all_ids[sline[0]] = int(sline[1])
            fp.close()
            self.celeb_ids = {}
            self.images = {}
            for i, path in enumerate(self.paths):
                print(i)
                fname = path.split("/")[-1]
                self.celeb_ids[fname] = all_ids[fname]
                img = Image.open(path).convert('RGB')
                self.images[path] = tfs.functional.resize(img, (self.image_size, self.image_size))
            temp = {}
            unique = list(set(self.celeb_ids.values()))
            for key,val in self.celeb_ids.items():
                temp[key] = unique.index(val)
            self.celeb_ids = temp
        self.transforms = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )


    def transform(self, img, hflip=False):
        '''
        if self.crop is not None:
            if isinstance(self.crop, int):
                img = tfs.CenterCrop(self.crop)(img)
            else:
                assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *self.crop)
        img = tfs.functional.resize(img, (self.image_size, self.image_size))
        '''
        if hflip:
            img = tfs.functional.hflip(img)
        return self.transforms(img)
        #return tfs.functional.to_tensor(img)

    def onehot(self, idx):
        if idx == 0:
            return np.array([1,0,0,0,0], dtype=np.float64)
        elif idx == 1:
            return np.array([0,1,0,0,0], dtype=np.float64)
        elif idx == 2:
            return np.array([0,0,1,0,0], dtype=np.float64)
        elif idx == 3:
            return np.array([0,0,0,1,0], dtype=np.float64)
        elif idx == 4:
            return np.array([0,0,0,0,1], dtype=np.float64)
        else:
            raise("Error: index out of expected range 0-4!")

    def __getitem__(self, index):
        img = None
        fpath = self.paths[index % self.size]
        if self.images:
            img = self.images[fpath]
        else:
            img = Image.open(fpath).convert('RGB')
        hflip = not self.is_validation and np.random.rand()>0.5
        #print(fpath)
        #print(labels)
        if self.celeb_labels:
            label = self.celeb_labels[fpath.split("/")[-1]]
            return self.transform(img, hflip=hflip), label
        elif self.celeb_ids:
            label = self.celeb_ids[fpath.split("/")[-1]]
            return self.transform(img, hflip=hflip), label
        else:
            labels = fpath.split("/")[-1][:-4].split("_")
            gender = np.float64(int(labels[1]))
            race = np.float64(int(labels[2]))
            #race = self.onehot(int(labels[2]))
            return self.transform(img, hflip=hflip), gender, race

    def __len__(self):
        return self.size

    def name(self):
        return 'ImageDataset'

def get_image_loader(data_dir, is_validation=False,
    batch_size=256, num_workers=4, image_size=256, crop=None, fraction=1):

    dataset = ImageDataset(data_dir, image_size=image_size, crop=crop, is_validation=is_validation, fraction=fraction)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader

def get_contrastive_loader(data_dir, is_validation=False,
    batch_size=256, num_workers=4, image_size=256, crop=None):

    dataset = ContrastiveImageDataset(data_dir, image_size=image_size, crop=crop, is_validation=is_validation)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


## paired AB image dataset ##
def make_paied_dataset(dir, AB_dnames=None, AB_fnames=None):
    A_dname, B_dname = AB_dnames or ('A', 'B')
    dir_A = os.path.join(dir, A_dname)
    dir_B = os.path.join(dir, B_dname)
    assert os.path.isdir(dir_A), '%s is not a valid directory' % dir_A
    assert os.path.isdir(dir_B), '%s is not a valid directory' % dir_B

    images = []
    for root_A, _, fnames_A in sorted(os.walk(dir_A)):
        for fname_A in sorted(fnames_A):
            if is_image_file(fname_A):
                path_A = os.path.join(root_A, fname_A)
                root_B = root_A.replace(dir_A, dir_B, 1)
                if AB_fnames is not None:
                    fname_B = fname_A.replace(*AB_fnames)
                else:
                    fname_B = fname_A
                path_B = os.path.join(root_B, fname_B)
                if os.path.isfile(path_B):
                    images.append((path_A, path_B))
    return images


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size=256, crop=None, is_validation=False, AB_dnames=None, AB_fnames=None):
        super(PairedDataset, self).__init__()
        self.root = data_dir
        self.paths = make_paied_dataset(data_dir, AB_dnames=AB_dnames, AB_fnames=AB_fnames)
        self.size = len(self.paths)
        self.image_size = image_size
        self.crop = crop
        self.is_validation = is_validation

    def transform(self, img, hflip=False):
        if self.crop is not None:
            if isinstance(self.crop, int):
                img = tfs.CenterCrop(self.crop)(img)
            else:
                assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *self.crop)
        img = tfs.functional.resize(img, (self.image_size, self.image_size))
        if hflip:
            img = tfs.functional.hflip(img)
        return tfs.functional.to_tensor(img)

    def __getitem__(self, index):
        path_A, path_B = self.paths[index % self.size]
        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        hflip = not self.is_validation and np.random.rand()>0.5
        return self.transform(img_A, hflip=hflip), self.transform(img_B, hflip=hflip)

    def __len__(self):
        return self.size

    def name(self):
        return 'PairedDataset'


def get_paired_image_loader(data_dir, is_validation=False,
    batch_size=256, num_workers=4, image_size=256, crop=None, AB_dnames=None, AB_fnames=None):

    dataset = PairedDataset(data_dir, image_size=image_size, crop=crop, \
        is_validation=is_validation, AB_dnames=AB_dnames, AB_fnames=AB_fnames)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader
