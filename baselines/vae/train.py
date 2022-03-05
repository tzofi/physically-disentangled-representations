import sys
import os
import time
import argparse
import torch
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import yaml

from torch.utils.data import DataLoader
from dataset import ImageDataset
from vae import VQ_CVAE

device = torch.device("cuda")

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
params = config['exp_params']
data_params = config['data_params']
log_params = config['logging_params']


# train validate
def train_validate(model, loader, optimizer, is_train, epoch):

    if is_train:
        model.train()
        model.zero_grad()
    else:
        model.eval()

    desc = 'Train' if is_train else 'Validation'

    total_loss = 0.0

    for i, image in enumerate(loader):

        image = image.to(device, non_blocking=True)

        outputs = model(image)
        loss = model.loss_function(image, *outputs)

        if is_train:
            loss.backward()
            optimizer.step()
            model.zero_grad()

        total_loss += loss.item()

        print('{} Epoch: {}, Step: [{}/{}], Average Loss: {:.4f}, Loss: {:.4f}'.format(desc, epoch, i+1, len(loader), total_loss/(i+1), loss.item()))

    return total_loss / (len(loader))

def sample_images(model, loader, epoch, name):
    test_input = next(iter(loader))
    test_input = test_input.to(device)

    recons = model(test_input)[0]
    vutils.save_image(recons.data,
                      os.path.join(log_params['save_dir'],
                                   log_params['name'],
                                   f"recons_{name}_Epoch_{epoch}.png"),
                      normalize=True,
                      nrow=12)

def execute_graph(model, loaders, optimizer, scheduler, epoch):
    t_loss = train_validate(model, loaders[0], optimizer, True, epoch)
    v_loss = train_validate(model, loaders[1], optimizer, False, epoch)

    print("Validation loss: {}".format(v_loss))

    scheduler.step(v_loss)

    return v_loss

model = VQ_CVAE(416, k=512, num_channels=3)
model = model.to(device)
pcount = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Param count: {}".format(pcount))

optimizer = optim.Adam(model.parameters(), lr=2e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.5,)

train_dataset   = ImageDataset(os.path.join(data_params['data_path']), "train", data_params['patch_size'], fraction=1)
val_dataset     = ImageDataset(os.path.join(data_params['data_path']), "val", data_params['patch_size'])
#test_dataset    = ImageDataset(os.path.join(data_params['data_path']), "test", data_params['patch_size'])

train_loader    = DataLoader(train_dataset, batch_size=data_params['train_batch_size'], shuffle=True, pin_memory=True)
val_loader      = DataLoader(val_dataset, batch_size=data_params['val_batch_size'], shuffle=False, pin_memory=True)
#test_loader     = DataLoader(test_dataset, batch_size=data_params['val_batch_size'], shuffle=False)
loaders = [train_loader, val_loader]

''' Verify weights directory exists, if not create it '''
if not os.path.isdir(os.path.join(log_params['save_dir'],log_params['name'])):
    os.makedirs(os.path.join(log_params['save_dir'],log_params['name']))

# Main training loop
best_loss = np.inf
epochs_since_best = 0

for epoch in range(config['trainer_params']['max_epochs']):
    v_loss = execute_graph(model, loaders, optimizer, scheduler, epoch)
    sample_images(model, val_loader, epoch, log_params['name'])

    if v_loss < best_loss:
        best_loss = v_loss
        print('Writing model checkpoint')
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_loss': v_loss
        }
        torch.save(state, log_params['save_dir'] + "/" + log_params['name'] + "/checkpoint_" + str(epoch).zfill(3) + ".ckpt")
        epochs_since_best = 0
    else:
        epochs_since_best += 1
        print("Epochs since best: {}".format(epochs_since_best))

    if epochs_since_best == config['trainer_params']['patience']:
        print("Early stopping reached.")
        break
