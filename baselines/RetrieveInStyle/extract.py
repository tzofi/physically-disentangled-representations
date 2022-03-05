import torch
from torch import nn
import numpy as np
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import *
from spherical_kmeans import MiniBatchSphericalKMeans as sKmeans
from tqdm import tqdm as tqdm
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # get rid of interpolation warning
from util import *
from e4e.models.psp import pSp
from argparse import Namespace
#from google.colab import files
from util import align_face, remove_2048
import os
import sys
from e4e_projection import projection, extract_projection
from PIL import Image

stop_idx = 11
labels2idx = {
    'nose': 0,
    'eyes': 1,
    'mouth':2,
    'hair': 3,
    'background': 4,
    'cheeks': 5,
    'neck': 6,
    'clothes': 7,
}
# Assign to each feature the cluster index from segmentation
labels_map = {
    0: torch.tensor([7]),
    1: torch.tensor([1,6]),
    2: torch.tensor([4]),
    3: torch.tensor([0,3,5,8,10,15,16]),
    4: torch.tensor([11,13,14]),
    5: torch.tensor([9]),
    6: torch.tensor([17]),
    7: torch.tensor([2,12]),
}

idx2labels = dict((v,k) for k,v in labels2idx.items())
n_class = len(labels2idx)

@torch.no_grad()
def compute_M(w, device='cuda'):
    M = []

    # get segmentation
    _, outputs = generator(w, is_cluster=1)
    cluster_layer = outputs[stop_idx][0]
    activation = flatten_act(cluster_layer)
    seg_mask = clusterer.predict(activation)
    b,c,h,w = cluster_layer.size()

    # create masks for each feature
    all_seg_mask = []
    seg_mask = torch.from_numpy(seg_mask).view(b,1,h,w,1).to(device)

    for key in range(n_class):
        # combine masks for all indices for a particular segmentation class
        indices = labels_map[key].view(1,1,1,1,-1)
        key_mask = (seg_mask == indices.to(device)).any(-1) #[b,1,h,w]
        all_seg_mask.append(key_mask)

    all_seg_mask = torch.stack(all_seg_mask, 1)

    # go through each activation layer and compute M
    for layer_idx in range(len(outputs)):
        layer = outputs[layer_idx][1].to(device)
        b,c,h,w = layer.size()
        layer = F.instance_norm(layer)
        layer = layer.pow(2)

        # resize the segmentation masks to current activations' resolution
        layer_seg_mask = F.interpolate(all_seg_mask.flatten(0,1).float(), align_corners=False,
                                     size=(h,w), mode='bilinear').view(b,-1,1,h,w)

        masked_layer = layer.unsqueeze(1) * layer_seg_mask # [b,k,c,h,w]
        masked_layer = (masked_layer.sum([3,4])/ (h*w))#[b,k,c]

        M.append(masked_layer.to(device))

    M = torch.cat(M, -1) #[b, k, c]

    # softmax to assign each channel to a particular segmentation class
    M = F.softmax(M/.1, 1)
    # simple thresholding
    M = (M>.8).float()

    # zero out torgb transfers, from https://arxiv.org/abs/2011.12799
    for i in range(n_class):
        part_M = style2list(M[:, i])
        for j in range(len(part_M)):
            if j in rgb_layer_idx:
                part_M[j].zero_()
        part_M = list2style(part_M)
        M[:, i] = part_M

    return M

# Load pretrained generator
device = 'cuda' # if GPU memory is low, use cpu instead

generator = Generator(1024, 512, 8, channel_multiplier=2).to(device).eval()

# load model file from current directory
ensure_checkpoint_exists('stylegan2-ffhq-config-f.pt')
ckpt = torch.load('stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
generator.load_state_dict(ckpt["g_ema"], strict=False)
model_path = 'e4e_ffhq_encode.pt'
ensure_checkpoint_exists(model_path)
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
opts['checkpoint_path'] = model_path
opts= Namespace(**opts)
net = pSp(opts, device).eval().to(device)

direct = list(sorted(list(os.listdir(sys.argv[1]))))
labels_to_save = []
feats_to_save = []
count = 0
clusterer = pickle.load(open("catalog.pkl", "rb"))

truncation = 0.5
stop_idx = 11 # choose 32x32 layer to do kmeans clustering
n_clusters = 18
for fname in direct:
    print(count)
    label = int(float(fname.split("_")[-2]))
    fpath = os.path.join(sys.argv[1], fname)
    img = Image.open(fpath)
    latent = extract_projection(img, net, device)
    source = latent[0]

    """
    # Uncomment this for RIS features, else StyleGAN2 features
    sources = []
    if source.size(0) != 1:
        source = source.unsqueeze(0)

    if source.ndim == 3:
        source = generator.get_latent(source, truncation=1, is_latent=True)
        source = list2style(source)
        
    sources.append(source)
    sources = torch.cat(sources, 0)
    if type(sources) is not list:
        sources = style2list(sources)
    print(len(sources))
    print(sources[0].shape)

    query_M = remove_2048(compute_M(sources, device=device), labels2idx).to(device)
    latent = latent[:,0,:]
    print(latent.shape)
    source_im, _ = generator(latent)
    print(im_source.shape)
    exit()

    sample, outputs = generator(latent, is_cluster=1) # [b, c, h, w]
    act = flatten_act(outputs[stop_idx][0])
    print(act.shape)
    b,c,h,w = outputs[stop_idx][0].size()

    alpha = 0.5
    seg_mask = clusterer.predict(act)
    print(seg_mask.shape)
    exit()

    #image = np.expand_dims(np.rollaxis(np.array(Image.open(fpath)),2,0), 0)
    #image = torch.tensor(image).to(device)
    #batch = 1

    #model.set_eval()
    #depth, albedo, light, view = model.get_features(image)
    #feats = torch.cat([depth,albedo],1).to(device).view(batch, -1) #.unsqueeze(0)
    #feats = feats.detach().cpu().numpy()
    """
    labels_to_save.append(label)
    feats_to_save.append(source.flatten().detach().cpu().numpy())
    count += 1

np.savez('stylegan_rafdb_64x64_feats.npz', feats=np.array(feats_to_save), labels=np.array(labels_to_save))
