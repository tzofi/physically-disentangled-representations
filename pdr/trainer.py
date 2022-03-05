import copy
import os
import glob
from datetime import datetime
import numpy as np
import torch
from . import meters
from . import utils
from .dataloaders import get_data_loaders
from .contrastive_loss import SupConLoss

class Trainer():
    def __init__(self, cfgs, model):
        self.device = cfgs.get('device', 'cuda')
        self.num_epochs = cfgs.get('num_epochs', 30)
        self.batch_size = 1 #cfgs.get('batch_size', 64)
        self.checkpoint_dir = cfgs.get('checkpoint_dir', 'results')
        self.save_checkpoint_freq = cfgs.get('save_checkpoint_freq', 1)
        self.keep_num_checkpoint = cfgs.get('keep_num_checkpoint', 2)  # -1 for keeping all checkpoints
        self.resume = cfgs.get('resume', True)
        self.use_logger = cfgs.get('use_logger', True)
        self.log_freq = cfgs.get('log_freq', 1000)
        self.archive_code = cfgs.get('archive_code', True)
        self.checkpoint_name = cfgs.get('checkpoint_name', None)
        self.test_result_dir = cfgs.get('test_result_dir', None)
        self.cfgs = cfgs
        self.contrastive = cfgs.get('contrastive', False)
        self.cyclic = cfgs.get('cyclic', False)

        self.metrics_trace = meters.MetricsTrace()
        self.make_metrics = lambda m=None: meters.StandardMetrics(m)
        self.model = model(cfgs)
        self.model.trainer = self
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(cfgs)

    def load_checkpoint(self, optim=True):
        """Search the specified/latest checkpoint in checkpoint_dir and load the model and optimizer."""
        if self.checkpoint_name is not None:
            checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)
        else:
            checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, '*.pth')))
            if len(checkpoints) == 0:
                return 0
            checkpoint_path = checkpoints[-1]
            self.checkpoint_name = os.path.basename(checkpoint_path)
        print(f"Loading checkpoint from {checkpoint_path}")
        cp = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_model_state(cp)
        if optim:
            self.model.load_optimizer_state(cp)
        self.metrics_trace = cp['metrics_trace']
        epoch = cp['epoch']
        return epoch

    def save_checkpoint(self, epoch, recon_loss, optim=True):
        """Save model, optimizer, and metrics state to a checkpoint in checkpoint_dir for the specified epoch."""
        utils.xmkdir(self.checkpoint_dir)
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint{epoch:03}.pth')
        state_dict = self.model.get_model_state()
        if optim:
            optimizer_state = self.model.get_optimizer_state()
            state_dict = {**state_dict, **optimizer_state}
        state_dict['metrics_trace'] = self.metrics_trace
        state_dict['epoch'] = epoch
        state_dict['reconstruction_loss'] = recon_loss
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(state_dict, checkpoint_path)
        if self.keep_num_checkpoint > 0:
            utils.clean_checkpoint(self.checkpoint_dir, keep_num=self.keep_num_checkpoint)

    def save_clean_checkpoint(self, path):
        """Save model state only to specified path."""
        torch.save(self.model.get_model_state(), path)

    def test(self):
        """Perform testing."""
        self.model.to_device(self.device)
        self.current_epoch = self.load_checkpoint(optim=False)
        if self.test_result_dir is None:
            self.test_result_dir = os.path.join(self.checkpoint_dir, f'test_results_{self.checkpoint_name}'.replace('.pth',''))
        print(f"Saving testing results to {self.test_result_dir}")

        with torch.no_grad():
            m, _ = self.run_epoch(self.test_loader, epoch=self.current_epoch, is_test=True)

        score_path = os.path.join(self.test_result_dir, 'eval_scores.txt')
        self.model.save_scores(score_path)

    def train(self):
        """Perform training."""
        ## archive code and configs
        if self.archive_code:
            utils.archive_code(os.path.join(self.checkpoint_dir, 'archived_code.zip'), filetypes=['.py', '.yml'])
        utils.dump_yaml(os.path.join(self.checkpoint_dir, 'configs.yml'), self.cfgs)

        ## initialize
        start_epoch = 0
        self.metrics_trace.reset()
        self.train_iter_per_epoch = len(self.train_loader)
        self.model.to_device(self.device)
        self.model.init_optimizers()
        torch.random.seed()

        ## resume from checkpoint
        if self.resume:
            start_epoch = self.load_checkpoint(optim=True)

        ## initialize tensorboardX logger
        if self.use_logger:
            from tensorboardX import SummaryWriter
            self.logger = SummaryWriter(os.path.join(self.checkpoint_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")))

            ## cache one batch for visualization
            self.viz_input = self.val_loader.__iter__().__next__()

        ## run epochs
        print(f"{self.model.model_name}: optimizing to {self.num_epochs} epochs")
        best_loss = np.inf
        epochs_since_best = 0
        cont_loss_weight = 0.01
        for epoch in range(start_epoch, self.num_epochs):
            if epoch % 10 == 0:
                # Optional update on weighting term during training. Currently weight stays the same.
                cont_loss_weight *= 1
            self.current_epoch = epoch
            metrics, _, _ = self.run_epoch(self.train_loader, epoch, cont_loss_weight=cont_loss_weight)
            self.metrics_trace.append("train", metrics)

            with torch.no_grad():
                metrics, total_loss, recon_loss = self.run_epoch(self.val_loader, epoch, is_validation=True, cont_loss_weight=cont_loss_weight)
                self.metrics_trace.append("val", metrics)

                # early stopping
                if total_loss <= best_loss:
                    best_loss = total_loss
                    epochs_since_best = 0
                else:
                    epochs_since_best += 1
                    print("Epochs since best loss: {}".format(epochs_since_best))

                if epochs_since_best == 20:
                    print("Early stopping point reached (patience=10). Stopping training.")
                    break

            #if (epoch+1) % self.save_checkpoint_freq == 0:
            if epochs_since_best == 0:
                self.save_checkpoint(epoch+1, recon_loss, optim=True)
            self.metrics_trace.plot(pdf_path=os.path.join(self.checkpoint_dir, 'metrics.pdf'))
            self.metrics_trace.save(os.path.join(self.checkpoint_dir, 'metrics.json'))

        print(f"Training completed after {epoch+1} epochs.")

    def run_epoch(self, loader, epoch=0, is_validation=False, is_test=False, cont_loss_weight=0.01):
        """Run one epoch."""
        is_train = not is_validation and not is_test
        metrics = self.make_metrics()
        total_loss = 0
        recon_loss = 0

        if is_train:
            print(f"Starting training epoch {epoch}")
            self.model.set_train()
        else:
            print(f"Starting validation epoch {epoch}")
            self.model.set_eval()

        if not self.contrastive and not self.cyclic:

            for iter, input in enumerate(loader):
                m, _ = self.model.forward(input, cyclic=False)
                if is_train:
                    self.model.backward()
                elif is_test:
                    self.model.save_results(self.test_result_dir, img_name)

                metrics.update(m, self.batch_size)
                total_loss += self.model.loss_total
                print("Epoch: {}, Iter: {}, Unsup Loss: {}, Avg Loss: {}".format(epoch, iter, self.model.loss_total, total_loss/(iter+1)))

            recon_loss = total_loss

        elif self.cyclic:
            # This is the training for the proposed LOOCC method

            print("Contrastive loss weight factor is: {}".format(cont_loss_weight))
            cont_loss = SupConLoss(temperature=0.5) 
            for iter, input in enumerate(loader):
                # Feats 1 are from the original image, Feats 2 are from a light or view "augmentation" of the original image
                # Light or cam indicates which augmentation was done for this batch
                m, feats1, feats2, light_or_cam = self.model.forward(input, cyclic=True)
                unsup_loss = self.model.loss_total.item()
                recon_loss += unsup_loss

                if light_or_cam == 0:
                    # Light is 3rd feature, so we don't include it in features that should be pushed together
                    feats1 = torch.cat([feats1[0].squeeze(), feats1[1].squeeze(), feats1[3].squeeze()], dim=-1)
                    feats2 = torch.cat([feats2[0].squeeze(), feats2[1].squeeze(), feats2[3].squeeze()], dim=-1)
                else:
                    # Cam is 4th feature, so we don't include it in features that should be pushed together
                    feats1 = torch.cat([feats1[0].squeeze(), feats1[1].squeeze(), feats1[2].squeeze()], dim=-1)
                    feats2 = torch.cat([feats2[0].squeeze(), feats2[1].squeeze(), feats2[2].squeeze()], dim=-1)

                closs = cont_loss(torch.cat([feats1.unsqueeze(1), feats2.unsqueeze(1)], dim=1)) 
                self.model.loss_total += cont_loss_weight * closs
                total_loss += self.model.loss_total

                if is_train:
                    self.model.backward()
                elif is_test:
                    self.model.save_results(self.test_result_dir)

                metrics.update(m, self.batch_size)
                print("Epoch: {}, Iter: {}, Unsup Loss: {}, Contrastive Loss: {}, Total Loss: {}, Avg Loss: {}".format(epoch, iter, unsup_loss, closs*cont_loss_weight, self.model.loss_total, total_loss/(iter+1)))

        else:
            # This is for contrastive learning without any cyclic encoding, i.e. normal data augmentation is used to create views
            
            cont_loss = SupConLoss(temperature=0.5) 

            for iter, (img1, img2) in enumerate(loader):
                img2 = img2.to(self.device)
                m, feats1 = self.model.forward(img1)
                feats2 = self.model.get_features(img2)

                feats1 = torch.stack((feats1[0], feats1[1], feats1[2], feats1[3])).view(self.batch_size, -1)
                feats2 = torch.stack((feats2[0], feats2[1], feats2[2], feats2[3])).view(self.batch_size, -1)

                #print([torch.any(torch.isnan(feats1)), torch.any(torch.isnan(feats2))])

                closs = cont_loss(feats1, feats2)

                self.model.loss_total += 2 * closs
                #m['loss'] = self.model.loss_total
                total_loss += self.model.loss_total
                
                if is_train:
                    self.model.backward()
                elif is_test:
                    self.model.save_results(self.test_result_dir)

                metrics.update(m, self.batch_size)
                #print(f"{'T' if is_train else 'V'}{epoch:02}/{iter:05}/{metrics}")
                print("Epoch: {}, Iter: {}, Contrastive Loss: {}, Total Loss: {}".format(epoch, iter, closs, self.model.loss_total))

                if self.use_logger and is_train:
                    total_iter = iter + epoch*self.train_iter_per_epoch
                    if total_iter % self.log_freq == 0:
                        self.model.forward(self.viz_input[0])
                        self.model.visualize(self.logger, total_iter=total_iter, max_bs=25)

        return metrics, total_loss/len(loader), recon_loss/len(loader)
