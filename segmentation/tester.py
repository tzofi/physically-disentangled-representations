
import os
import time
import torch
import datetime
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms

import cv2
import PIL
from unet_encoder2 import unet
from utils import *
from PIL import Image

from iou import IoU

CLASSES = [
  'background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow',
  'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 
  'ear_r', 'neck_l', 'neck', 'cloth'
]

class AverageMeter(object):
  def __init__(self):
    self.val = None
    self.sum = None
    self.cnt = None
    self.avg = None
    self.ema = None
    self.initialized = False

  def update(self, val, n=1):
    if not self.initialized:
      self.initialize(val, n)
    else:
      self.add(val, n)

  def initialize(self, val, n):
    self.val = val
    self.sum = val * n
    self.cnt = n
    self.avg = val
    self.ema = val
    self.initialized = True

  def add(self, val, n):
    self.val = val
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
    self.ema = self.ema * 0.99 + self.val * 0.01

EPS = 1e-10
def _fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist


def overall_pixel_accuracy(hist):
    """Computes the total pixel accuracy.
    The overall pixel accuracy provides an intuitive
    approximation for the qualitative perception of the
    label when it is viewed in its overall shape but not
    its details.
    Args:
        hist: confusion matrix.
    Returns:
        overall_acc: the overall pixel accuracy.
    """
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc

def inter_and_union(pred, mask, num_class):
  pred = np.asarray(pred, dtype=np.uint8).copy()
  mask = np.asarray(mask, dtype=np.uint8).copy()

  # 255 -> 0
  pred += 1
  mask += 1
  pred = pred * (mask > 0)

  inter = pred * (pred == mask)
  (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class))
  (area_pred, _) = np.histogram(pred, bins=num_class, range=(1, num_class))
  (area_mask, _) = np.histogram(mask, bins=num_class, range=(1, num_class))
  area_union = area_pred + area_mask - area_inter

  return (area_inter, area_union)

def transformer(resize, totensor, normalize, centercrop, imsize):
    options = []
    if centercrop:
        options.append(transforms.CenterCrop(160))
    if resize:
        options.append(transforms.Resize((imsize,imsize), interpolation=PIL.Image.NEAREST))
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(options)
    
    return transform

def transform_label(resize, totensor, normalize, centercrop):
    options = []
    #if centercrop:
    #    options.append(transforms.CenterCrop(160))
    #if resize:
    #    options.append(transforms.Resize((self.imsize,self.imsize)))
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0, 0, 0), (0, 0, 0)))
    transform = transforms.Compose(options)
    return transform

def make_dataset(dir):
    images = []
    labels = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for f in os.listdir(dir):
        images.append(os.path.join(dir, f))
        labels.append(os.path.join("./data/CelebAMask-HQ/CelebAMaskHQ-mask", f[:-4] + ".png"))
   
    return images, labels

class Tester(object):
    def __init__(self, config):
        # exact model and loss
        self.model = config.model

        # Model hyper-parameters
        self.imsize = config.imsize
        self.parallel = config.parallel

        self.total_step = config.total_step
        self.batch_size = 128 #config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.img_path = config.img_path
        self.label_path = config.label_path 
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)
        self.test_label_path = config.test_label_path
        self.test_color_label_path = config.test_color_label_path
        self.test_image_path = config.test_image_path

        # Test size and model
        self.test_size = config.test_size
        self.model_name = config.model_name

        self.build_model()

    def test(self):
        transform = transformer(True, True, True, False, self.imsize) 
        label_transformer = transform_label(True, True, False, False)
        test_paths, label_paths = make_dataset(self.test_image_path)
        make_folder(self.test_label_path, '')
        make_folder(self.test_color_label_path, '') 
        self.G.load_state_dict(torch.load(os.path.join(self.model_save_path, self.model_name)))
        self.G.eval() 
        batch_num = int(self.test_size / self.batch_size)

        inter_meter = AverageMeter()
        union_meter = AverageMeter()
        iou_meter = IoU(19)
        accs = []
        for i in range(batch_num):
            print (i)
            imgs = []
            lbls = []
            paths = []
            for j in range(self.batch_size):
                path = test_paths[i * self.batch_size + j]
                paths.append(path.split("/")[-1][:-4])
                lpath = label_paths[i * self.batch_size + j]
                img = transform(Image.open(path))
                lbl = label_transformer(Image.open(lpath))[0]
                imgs.append(img)
                lbls.append(lbl)
            imgs = torch.stack(imgs) 
            imgs = imgs.cuda()
            labels_predict = self.G(imgs)
            labels_predict_plain = generate_label_plain(labels_predict, 64)
            labels_predict_color = generate_label(labels_predict, 64)

            lbls = np.stack(lbls)

            inter, union = inter_and_union(labels_predict_plain, lbls, 19)
            inter_meter.update(inter)
            union_meter.update(union)

            iou_meter.add(torch.tensor(labels_predict_plain), torch.tensor(lbls))

            hist = _fast_hist(torch.tensor(lbls).long(),torch.tensor(labels_predict_plain).long(),19)
            acc = overall_pixel_accuracy(hist)
            accs.append(acc)

            for k in range(self.batch_size):
                #cv2.imwrite(os.path.join(self.test_label_path, str(i * self.batch_size + k) +'.png'), labels_predict_plain[k])
                #save_image(labels_predict_color[k], os.path.join(self.test_color_label_path, str(i * self.batch_size + k) +'.png'))
                cv2.imwrite(os.path.join(self.test_label_path, paths[k] +'.png'), labels_predict_plain[k])
                save_image(labels_predict_color[k], os.path.join(self.test_color_label_path, paths[k] +'.png'))
        iou = inter_meter.sum / (union_meter.sum + 1e-10)
        ious = []
        for i, val in enumerate(iou):
          print('IoU {}: {}'.format(CLASSES[i], val * 100))
          ious.append(val)

        ious = np.array(ious)
        ious = ious[~np.isnan(ious)]
        ind = np.argpartition(ious, -5)[-5:]
        top5 = ious[ind]
        iou, miou = iou_meter.value()
        print([iou, miou])
        iou = iou[~np.isnan(iou)]
        ind = np.argpartition(iou, -5)[-5:]
        top5_1 = ious[ind]
        print('Mean IoU: {}, Top 5 Mean IoU: {}, Other calcTop 5 Mean IoU: {}, Pixel Acc: {}'.format(iou.mean() * 100, np.mean(top5), np.mean(top5_1), np.mean(np.array(accs))))

    def build_model(self):
        self.G = unet().cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)

        # print networks
        print(self.G)
