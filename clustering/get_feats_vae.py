import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image

from torch.utils.data import DataLoader

sys.path.insert(1, "../pdr")
import pdr
from pdr.dataloaders import LabeledImageDataset

sys.path.insert(1, "../vae")
from vae import VQ_CVAE

args = parser.parse_args()
parser = argparse.ArgumentParser(description='Training configurations.')
parser.add_argument('--config', default=None, type=str, help='Specify a config file path')
parser.add_argument('--gpu', default=None, type=int, help='Specify a GPU device')
parser.add_argument('--num_workers', default=4, type=int, help='Specify the number of worker threads for data loaders')
parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
parser.add_argument('--data', default='data')
parser.add_argument('--utk_data', default='UTKFace_data')
parser.add_argument('--checkpoint_path', default='checkpoint_dir')
parser.add_argument('--batch_size', default=1)
parser.add_argument('--epochs', default=50, metavar='N')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument("--decay-lr", default=1e-6, action="store", type=float)
parser.add_argument('--output_dir', default="./classifier")
parser.add_argument('--mlp_only', default=False)
parser.add_argument('--test_only', default=False)
parser.add_argument('--save_feats', default=False)
parser.add_argument('--cluster_acc', default=True)
parser.add_argument('--ids', default=False)
parser.add_argument('--utk', default=False)
parser.add_argument('--test_checkpoint_path1', default='vae.ckpt')
parser.add_argument('--test_checkpoint_path2', default='classifier.ckpt')
args = parser.parse_args()
cfg = setup_runtime(args)
device = torch.device("cuda")
print(args)

# Choose Buffy or BBT
#data =  "./multimedia/bbt_tracks"
data =  "./multimedia/buffy_tracks"
weights = args.test_checkpoint_path1
device = torch.device("cuda")
model = VQ_CVAE(416, k=512, num_channels=3)
model = model.to(device)
ckpt = torch.load(args.test_checkpoint_path1, map_location=device)
state_dict = ckpt['model']
model.load_state_dict(state_dict)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) \
            if os.path.isdir(os.path.join(a_dir, name))]

# Choose Buffy or BBT
#ids = ['howard','leonard', 'penny', 'raj', 'sheldon']
ids = ['xander','buffy','dawn','anya','willow','giles']
labels = {}

# Choose Buffy or BBT
#fp = open("./multimedia/bbt_s01e01.ids", "r")
fp = open("./multimedia/buffy_s05e02.ids", "r")
for i, line in enumerate(fp):
    if i == 0: continue
    line = line.replace("\n", "").split(" ")
    if line[1] not in ids:
        continue 
    labels[line[0]] = ids.index(line[1])

SINGLE_IMAGE = True

total = 0
feats_to_save = []
labels_to_save = []
tracks = get_immediate_subdirectories(data)
for track in tracks:
    track_id = track.split("_")[1]
    if track_id not in labels.keys():
        print("Skipping {}".format(track_id))
        continue
    path = os.path.join(data, track)
    avg_feat = None
    count = 0
    direct = list(sorted(list(os.listdir(path))))[:-1]
    for fname in direct:
        print(total)
        fpath = os.path.join(path, fname)
        image = np.expand_dims(np.rollaxis(np.array(Image.open(fpath)),2,0), 0)
        image = torch.tensor(image, dtype=torch.float).to(device)
        batch = 1

        model.eval()
        _, _, feats, _ = model.forward(image)
        if SINGLE_IMAGE:
            feats_to_save.append(feats.flatten().view(batch, -1).detach().cpu().numpy())
            labels_to_save.append(labels[track_id])
        else:
            if avg_feat is not None:
                avg_feat = avg_feat + feats.flatten().view(batch, -1).detach().cpu().numpy()
            else:
                avg_feat = feats.flatten().view(batch, -1).detach().cpu().numpy()
        count += 1
        total += 1
    if not SINGLE_IMAGE:
        feats_to_save.append(avg_feat/count)
        labels_to_save.append(labels[track_id])

np.savez('vae_buffy_single_image_feats.npz', feats=np.array(feats_to_save), labels=np.array(labels_to_save))
