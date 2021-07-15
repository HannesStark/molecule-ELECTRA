import os
from itertools import chain
from typing import Dict, Callable

import torch
from torch.utils.data import DataLoader

from commons.utils import move_to_device
from trainer.trainer import Trainer


class PreTrainer(Trainer):
    def __init__(self,**kwargs):
        super(PreTrainer, self).__init__(**kwargs)

    def forward_pass(self, batch):
        graph, mask = tuple(batch)
        generator_loss, predictions = self.model(graph, mask)  # foward the rest of the batch to the model
        discriminator_loss = self.loss_func(predictions, mask.float())
        return generator_loss, discriminator_loss, predictions, mask

    def process_batch(self, batch, optim):
        generator_loss, discriminator_loss, predictions, target_mask = self.forward_pass(batch)
        if optim != None:  # run backpropagation if an optimizer is provided
            generator_loss.backward()
            discriminator_loss.backward()
            self.optim.step()
            self.after_optim_step()  # overwrite this function to do stuff before zeroing out grads
            self.optim.zero_grad()
            self.optim_steps += 1
        return generator_loss, discriminator_loss, predictions.detach(), target_mask.detach()

    def predict(self, data_loader: DataLoader, epoch: int = 0, optim: torch.optim.Optimizer = None,return_predictions: bool = False):
        total_metrics = {k: 0 for k in self.evaluate_metrics(torch.ones((2, 2), device=self.device),
                                                             torch.ones((2, 2), device=self.device), val=True).keys()}
        total_metrics['generator_loss'] = 0
        total_metrics[type(self.loss_func).__name__] = 0
        epoch_targets = torch.tensor([]).to(self.device)
        epoch_predictions = torch.tensor([]).to(self.device)
        epoch_loss = 0
        for i, batch in enumerate(data_loader):
            batch = move_to_device(list(batch), self.device)
            generator_loss, discriminator_loss, predictions, targets = self.process_batch(batch, optim)
            with torch.no_grad():
                if self.optim_steps % self.args.log_iterations == 0 and optim != None:
                    metrics = self.evaluate_metrics(predictions, targets)
                    metrics['generator_loss'] = generator_loss.item()
                    metrics[type(self.loss_func).__name__] = discriminator_loss.item()
                    self.run_tensorboard_functions(predictions, targets, step=self.optim_steps, data_split='train')
                    self.tensorboard_log(metrics, data_split='train', step=self.optim_steps, epoch=epoch)
                    print('[Epoch %d; Iter %5d/%5d] %s: discriminator_loss: %.7f' % (
                    epoch, i + 1, len(data_loader), 'train', discriminator_loss.item()))
                if optim == None and self.val_per_batch:  # during validation or testing when we want to average metrics over all the data in that dataloader
                    metrics_results = self.evaluate_metrics(predictions, targets, val=True)
                    metrics_results['generator_loss'] = generator_loss.item()
                    metrics_results[type(self.loss_func).__name__] = discriminator_loss.item()
                    self.run_tensorboard_functions(predictions, targets, step=self.optim_steps, data_split='val')
                    for key, value in metrics_results.items():
                        total_metrics[key] += value
                if optim == None and not self.val_per_batch:
                    epoch_loss += discriminator_loss.item()
                    epoch_targets = torch.cat((targets, epoch_targets), 0)
                    epoch_predictions = torch.cat((predictions, epoch_predictions), 0)

        if optim == None:
            if self.val_per_batch:
                total_metrics = {k: v / len(data_loader) for k, v in total_metrics.items()}
            else:
                total_metrics = self.evaluate_metrics(epoch_predictions, epoch_targets, val=True)
                total_metrics[type(self.loss_func).__name__] = epoch_loss / len(data_loader)
            return total_metrics

    def evaluate_metrics(self, pred, target, batch=None, val=False) -> Dict[str, float]:
        metric_results = {}
        for key, metric in self.metrics.items():
            if not hasattr(metric, 'val_only') or val:
                metric_results[key] = metric(pred, target).item()
        return metric_results

