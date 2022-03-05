"""
This file allows testing a classifier , and mainly contains the code for integrated gradients and GRAD-Cam.
Using PDR, you can attribute not just pixels, but physically meaningful features to the classification made by your model.
"""

import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2

from torch.utils.data import DataLoader

sys.path.insert(1, "../pdr")
import pdr

from pdr.dataloaders import LabeledImageDataset
from torch.optim.lr_scheduler import ExponentialLR
from utils import setup_runtime 
from scipy.stats import pearsonr, spearmanr
from sklearn import svm
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

celeba_attributes = ["5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald","Bangs","Big_Lips","Big_Nose","Black_Hair","Blond_Hair","Blurry","Brown_Hair","Bushy_Eyebrows","Chubby","Double_Chin","Eyeglasses","Goatee","Gray_Hair","Heavy_Makeup","High_Cheekbones","Male","Mouth_Slightly_Open","Mustache","Narrow_Eyes","No_Beard","Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline","Rosy_Cheeks","Sideburns","Smiling","Straight_Hair","Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick","Wearing_Necklace","Wearing_Necktie","Young"]

class MLP(nn.Module):
    def __init__(self, input_dim=512, num_classes=1):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.fc = nn.Linear(self.input_dim, self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x) 
        return x

class Model(nn.Module):
    def __init__(self, netD, netA, netL, netV, input_dim=512, num_classes=1):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.netD = netD
        self.netA = netA
        self.netL = netL
        self.netV = netV
        self.fc = nn.Linear(self.input_dim, self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch = x.shape[0]
        _, d = self.netD(x)
        _, a = self.netA(x)
        _, l = self.netL(x)
        _, v = self.netV(x)
        x = torch.cat([d,a,l,v],1).to(device).view(batch, -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class Combined(nn.Module):
    def __init__(self, feat, classifier):
        super(Combined, self).__init__()

        self.netD = feat.netD
        self.netA = feat.netA
        self.netL = feat.netL
        self.netV = feat.netV
        self.fc = classifier.fc
        self.sigmoid = classifier.sigmoid

    def forward(self, x):
        batch = x.shape[0]
        _, d = self.netD(x)
        _, a = self.netA(x)
        _, l = self.netL(x)
        _, v = self.netV(x)
        x = torch.cat([d,a],1).to(device).view(batch, -1)
        x = self.fc(x)
        outputs = self.sigmoid(x)
        return outputs


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

    count_0 = 0
    for i, (image, labels) in enumerate(loader):
        rgb_image = np.array(image)
        batch = image.shape[0]
        image = image.to(device, non_blocking=True)
        labels = labels.to(device).float()

        depth, albedo, light, view = model.get_features(image)
        feats = torch.cat([depth,albedo],1).to(device).view(batch, -1) #.unsqueeze(0)
        if args.save_feats:
            feats_to_save.append(feats.detach().cpu().numpy())
            labels_to_save.append(labels.detach().cpu().numpy())
            labels = torch.zeros((batch,40)).to(device)
        outputs = classifier(feats)

        from captum.attr import IntegratedGradients


        # Index for Open Mouth
        x = 21

        from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image

        combined = Combined(model, classifier)
        target_layers = [combined.netD.network_down[-4]]
        cam = GradCAM(model=combined, target_layers=target_layers, use_cuda=True)
        target_category=[ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=image, targets=target_category)
        if np.sum(grayscale_cam.flatten()) == 0:
            continue

        grayscale_cam = grayscale_cam[0, :]
        rgb_image = rgb_image[0]
        rgb_image = np.rollaxis(rgb_image, 0, 3) 
        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        cv2.imwrite("{}_gradcam_mouth_depth.png".format(str(i).zfill(3)), visualization)

        target_layers = [combined.netA.network_down[-4]]
        cam = GradCAM(model=combined, target_layers=target_layers, use_cuda=True)
        target_category=[ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=image, targets=target_category)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        cv2.imwrite("{}_gradcam_mouth_albedo.png".format(str(i).zfill(3)), visualization)

        target_layers = [combined.netD.network_down[-4], combined.netA.network_down[-4]]
        cam = GradCAM(model=combined, target_layers=target_layers, use_cuda=True)
        target_category=[ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=image, targets=target_category)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        cv2.imwrite("{}_gradcam_mouth_total.png".format(str(i).zfill(3)), visualization)
        continue
        exit()

        # For computing integrated gradients, use the code below:

        ig = IntegratedGradients(classifier)
        print(outputs.shape)
        for i in range(0, 40):
            attributes, delta = ig.attribute(feats, feats * 0, target=i, return_convergence_delta=True)
            attributes = attributes.detach().cpu().numpy()
            depth = np.sum(np.abs(attributes[:,0:256]))
            albedo = np.sum(np.abs(attributes[:,256:512]))
            light = np.sum(np.abs(attributes[:,512:768]))
            view = np.sum(np.abs(attributes[:,768:1024]))
            m = np.argmax(np.array([depth,albedo,light,view]))
            #depth = np.sum(np.abs(attributes[:,0:256]))
            #albedo = np.sum(np.abs(attributes[:,256:512]))
            total = depth + albedo + light + view
            depth = depth/total
            albedo = albedo/total
            light = light/total
            view = view/total
            print("{}, Depth: {}, Albedo: {}, Light: {}, View: {}".format(celeba_attributes[i], depth, albedo, light, view))

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
    if not is_train:
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
    gt = np.array(labels_to_save)
    accs = []
    nums = []
    for label in range(gt.shape[-1]):
        gt_i = gt[:,:,label]
        y_pred = hac_cluster(np.squeeze(d, axis=1), np.unique(gt_i).shape[0])
        gt_i = gt_i.flatten()

        print("Results for {}, {}:".format(fname, d.shape))
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
