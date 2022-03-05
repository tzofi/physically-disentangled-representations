import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pdb
import collections
from scipy.cluster import hierarchy as sphac
from scipy.spatial import distance as spdist
import warnings
from sklearn import cluster
from sklearn.metrics import recall_score, precision_score, f1_score, precision_recall_fscore_support
from scipy.stats import mode

from torch.utils.data import DataLoader

sys.path.insert(1, "../pdr")
import pdr
from utils import setup_runtime 
from pdr.dataloaders import LabeledImageDataset
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import accuracy_score


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
parser.add_argument('--test_checkpoint_path1', default='pre_trained_weights.ckpt')
parser.add_argument('--test_checkpoint_path2', default='classifier.ckpt')
args = parser.parse_args()
device = torch.device("cuda")
print(args)


def weighted_purity(Y, C):
    """Computes weighted purity of HAC at one particular clustering "C".
    Y, C: np.array([...]) containing unique cluster indices (need not be same!)
    Note: purity --> 1 as the number of clusters increase, so don't look at this number alone!
    """

    purity = 0.
    uniq_clid, clustering_skew = np.unique(C, return_counts=True)
    num_samples = np.zeros(uniq_clid.shape)
    # loop over all predicted clusters in C, and measure each one's cardinality and purity
    for k in uniq_clid:
        # gt labels for samples in this cluster
        k_gt = Y[np.where(C == k)[0]]
        values, counts = np.unique(k_gt, return_counts=True)
        # technically purity = max(counts) / sum(counts), but in WCP, the sum(counts) multiplies to "weight" the clusters
        purity += max(counts)

    purity /= Y.shape[0]
    return purity, clustering_skew


def WCP(Y_pred, Y):
    gt_num_clusters = len(set(Y_pred.flatten()))

    predicted_label = np.zeros((len(Y_pred), 1)).astype('int')

    wcp = 0
    for i in range(gt_num_clusters):
        idx = np.where(Y_pred == i)
        actual_cluster_i = Y[idx]

        id = mode(actual_cluster_i)[0][0]
        predicted_label[idx,:]= id
        wcp = wcp + len(np.where(actual_cluster_i==id)[0])

    wcp = wcp/len(Y_pred)

    return wcp, predicted_label


def cluster_acc(Y_pred, Y):
    from sklearn.utils.linear_assignment_ import linear_assignment
    #from scipy.optimize import linear_sum_assignment as linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size,ind


def NMI(Y, C):
    """Normalized Mutual Information: Clustering performance between ground-truth Y and prediction C
    Based on https://course.ccs.neu.edu/cs6140sp15/7_locality_cluster/Assignment-6/NMI.pdf

    Result matches examples on pdf
    Example:
    Y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    C = np.array([1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2])
    NMI(Y, C) = 0.1089

    C = np.array([1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
    NMI(Y, C) = 0.2533
    """

    def entropy(labels):
        # H(Y) and H(C)
        H = 0.
        for k in np.unique(labels):
            p = (labels == k).sum() / labels.size
            H -= p * np.log2(p)
        return H

    def h_y_given_c(labels, pred):
        # H(Y | C)
        H = 0.
        for c in np.unique(pred):
            p_c = (pred == c).sum() / pred.size
            labels_c = labels[pred == c]
            for k in np.unique(labels_c):
                p = (labels_c == k).sum() / labels_c.size
                H -= p_c * p * np.log2(p)
        return H

    h_Y = entropy(Y)
    h_C = entropy(C)
    h_Y_C = h_y_given_c(Y, C)
    # I(Y; C) = H(Y) - H(Y|C)
    mi = h_Y - h_Y_C
    # NMI = 2 * MI / (H(Y) + H(C))
    nmi = 2 * mi / (h_Y + h_C)
    return nmi

def hac_cluster(data,req_n_clusters):
    algorithm = cluster.AgglomerativeClustering(n_clusters=req_n_clusters, linkage='ward')
    warnings.filterwarnings("ignore")
    algorithm.fit(data)

    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_
    else:
        y_pred = algorithm.predict(data)
    return y_pred


class MLP(nn.Module):
    def __init__(self, input_dim=512, num_classes=40):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.fc = nn.Linear(self.input_dim, self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x) 
        return x

def get_accuracy(y_true, y_prob):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)

# train validate
def train_validate(model, loader, optimizer, is_train, epoch):

    criterion = torch.nn.BCELoss()
    model, classifier = model
    model.set_eval()

    if is_train:
        classifier.train()
        classifier.zero_grad()
    else:
        classifier.eval()

    desc = 'Train' if is_train else 'Validation'

    total_loss = 0.0
    total_accuracy = 0.0
    per_class_acc = {}
    lbl_counts = []

    count_0 = 0
    for i, (image, labels_race) in enumerate(loader):
        batch = image.shape[0]
        image = image.to(device, non_blocking=True)
        labels = labels_race.to(device).float()

        depth, albedo, light, view = model.get_features(image)
        # Uncomment below line and comment line after to train with all features, not just albedo and geometry
        #feats = torch.cat([depth,albedo,light,view],1).to(device).view(batch, -1) 
        feats = torch.cat([depth,albedo],1).to(device).view(batch, -1)
        if args.save_feats or args.cluster_acc:
            feats_to_save.append(feats.detach().cpu().numpy())
            labels_to_save.append(labels.detach().cpu().numpy())
            if lbl_counts == []:
                lbl_counts = labels.detach().cpu().numpy()
            else:
                lbl_counts += labels.detach().cpu().numpy()
            labels = torch.zeros((batch,40)).to(device)
            continue
        outputs = classifier(feats)
        loss = criterion(outputs, labels)
        accuracy = get_accuracy(labels.flatten(), outputs.flatten())
        if not is_train:
            for i in range(outputs.shape[1]):
                if i == 0: count_0 += 1
                labelS = labels[:,i]
                outputS = outputs[:,i]
                accuracyS = get_accuracy(labelS.flatten(), outputS.flatten())
                if i in per_class_acc:
                    per_class_acc[i] += accuracyS 
                else:
                    per_class_acc[i] = accuracyS

        if is_train:
            loss.backward()
            optimizer.step()
            classifier.zero_grad()
    if not is_train and not args.save_feats and not args.cluster_acc:
        for i in range(labels.shape[1]):
            per_class_acc[i] = per_class_acc[i]/count_0

        total_loss += loss.item()
        total_accuracy += accuracy

        print('{} Epoch: {}, Step: [{}/{}], Average Accuracy: {:.4f}, Accuracy: {:.4f}, Average Loss: {:.4f}, Loss: {:.4f}'.format(desc, epoch, i+1, len(loader), total_accuracy/(i+1), accuracy, total_loss/(i+1), loss.item()))

    return total_loss / (len(loader)), total_accuracy/ (len(loader)), per_class_acc

def execute_graph(model, loaders, optimizer, scheduler, epoch):
    t_loss, t_acc, _ = train_validate(model, loaders[0], optimizer, True, epoch)
    v_loss = 0
    v_acc = 0
    if epoch == args.epochs: 
        v_loss, v_acc, per_class_acc = train_validate(model, loaders[1], optimizer, False, epoch)

    print("Test loss: {}, Test Acc: {}".format(v_loss, v_acc))
    print("Per class acc:\n{}".format(per_class_acc))

    return v_loss


cfg = setup_runtime(args)

test_dataset = None
if args.ids:
    test_dataset    = LabeledImageDataset(os.path.join(args.data, "test_id"), celeb=False, celeb_id=True, limit=6000)
elif args.utk:
    test_dataset    = LabeledImageDataset(os.path.join(args.utk_data, "test"), celeb=False, celeb_id=False, limit=6000)
else:
    test_dataset    = LabeledImageDataset(os.path.join(args.data, "test"), celeb=True, limit=6000)
test_loader     = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

if args.test_only:
    model = pdr.model.PDR(cfg)
    ckpt = torch.load(args.test_checkpoint_path1, map_location=device)
    model.load_model_state(ckpt)
    pdr.model.PDR(cfg).to_device(device)
    classifier = MLP().to(device)
    ckpt = torch.load(args.test_checkpoint_path2, map_location=device)
    state_dict = ckpt['model']
    classifier.load_state_dict(state_dict)
    model = [model, classifier]
    _, _, per_cls = train_validate(model, test_loader, None, False, 0)
    print(per_cls)
    exit()

feats_to_save = []
labels_to_save = []
if args.save_feats:
    model = pdr.model.PDR(cfg)
    ckpt = torch.load(args.test_checkpoint_path1, map_location=device)
    model.load_model_state(ckpt)
    pdr.model.PDR(cfg).to_device(device)
    classifier = MLP().to(device)
    ckpt = torch.load(args.test_checkpoint_path2, map_location=device)
    state_dict = ckpt['model']
    classifier.load_state_dict(state_dict)
    model = [model, classifier]
    _, _, per_cls = train_validate(model, test_loader, None, False, 0)
    print(per_cls)
    np.savez('inverse_rendered_feats.npz', feats=np.array(feats_to_save), labels=np.array(labels_to_save))
    exit()

feats_to_save = []
labels_to_save = []
if args.cluster_acc:
    model = pdr.model.PDR(cfg)
    ckpt = torch.load(args.test_checkpoint_path1, map_location=device)
    model.load_model_state(ckpt)
    pdr.model.PDR(cfg).to_device(device)
    classifier = MLP().to(device)
    ckpt = torch.load(args.test_checkpoint_path2, map_location=device)
    state_dict = ckpt['model']
    classifier.load_state_dict(state_dict)
    model = [model, classifier]
    _, _, per_cls = train_validate(model, test_loader, None, False, 0)

    d = np.array(feats_to_save)
    d = d[:,:,0:256]
    gt = np.array(labels_to_save)
    accs = []
    nums = []
    for label in range(gt.shape[-1]):
        gt_i = gt[:,:,label]
        y_pred = hac_cluster(np.squeeze(d, axis=1), np.unique(gt_i).shape[0])
        gt_i = gt_i.flatten()

        nmi = NMI(gt_i, y_pred)
        wcp, predicted = WCP(y_pred,gt_i)
        prec = precision_score(gt_i, predicted, average='weighted')
        rec = recall_score(gt_i, predicted, average='weighted')
        f1 = f1_score(gt_i, predicted, average='weighted')
        if accs == []:
            accs = np.array([nmi, prec, rec, f1])
        else:
            accs += np.array([nmi, prec, rec, f1])

        print('NMI: {}, WCP: {}, Precision: {}, Recall: {}, F1: {}, No. of Clusters: {}'.format(nmi,wcp,prec,rec,f1,np.unique(gt_i).shape[0]))

    print(accs/gt.shape[-1])

    exit()

train_dataset     = LabeledImageDataset(os.path.join(args.data, "train"), celeb=True, limit=6000)
train_loader      = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)


''' Verify weights directory exists, if not create it '''
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

# BELOW CODE WILL TRAIN CLASSIFIERS FOR EACH WEIGHTS IN THE CHECKPOINT_DIR 

skipped = 4
direct = list(reversed(sorted(list(os.listdir(args.checkpoint_path)))))
for ckpt_file in direct:
    if ckpt_file[-4:] != ".pth": continue
    else:
        if skipped == 4:
            skipped = 0 
            print("Processing checkpoint: {}".format(ckpt_file))
        else:
            skipped += 1
            continue

    ''' Verify weights directory exists, if not create it '''
    if not os.path.isdir(os.path.join(args.output_dir, ckpt_file[:-4])):
        os.makedirs(os.path.join(args.output_dir, ckpt_file[:-4]))

    #################
    # Create Models #
    #################
    model = pdr.model.PDR(cfg)
    ckpt = torch.load(os.path.join(args.checkpoint_path, ckpt_file), map_location=device)
    recon_loss = ckpt['reconstruction_loss']
    print("Reconstruction loss for {} is {}".format(ckpt_file, recon_loss))
    model.load_model_state(ckpt)
    pdr.model.PDR(cfg).to_device(device)
    classifier = MLP().to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=2e-4)

    # Main training loop
    for epoch in range(args.epochs):
        v_loss = execute_graph([model, classifier], [train_loader, test_loader], optimizer, None, epoch+1)

        print('Writing model checkpoint')
        state = {
            'epoch': epoch,
            'model': classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': v_loss
        }
        torch.save(state, os.path.join(args.output_dir, ckpt_file[:-4]) + "/best.ckpt")
