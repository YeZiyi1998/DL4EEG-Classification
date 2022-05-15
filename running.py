import torch
import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, ndcg_score, roc_auc_score, precision_score, f1_score
import torch.nn as nn
import math

class BaseRunner(object):
    def __init__(self, model, train_loader, valid_loader, device, loss_module, optimizer, print_interval=30, batch_size= 8, l2_reg=0.0):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.loss_module = loss_module
        self.print_interval = print_interval
        self.batch_size = batch_size
        self.optimizer = optimizer 
        self.cos_fuc = nn.CosineSimilarity()
        self.l2_reg = l2_reg
        
    def train_epoch(self, epoch_num = None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num = None, keep_all = True):
        raise NotImplementedError('Please override in child class')

def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""
    total_loss = 0
    for name, param in model.named_parameters():
        total_loss += torch.sum(torch.square(param))
    return total_loss

class UnsupervisedRunner(BaseRunner):
    def __init__(self, *args, **kwargs):
        super(UnsupervisedRunner, self).__init__(*args, **kwargs)

    def train_epoch(self, epoch_num=None):
        self.model = self.model.train()
        epoch_loss = 0  # total loss of epoch
        rl = 0
        total_samples = 0  # total samples in epoch
        epoch_metrics = {}
        for i, batch in enumerate(self.train_loader):
            X1, X2, Y1, Y2, X1_mask, X2_mask, padding_masks = batch

            predictions1, predictions2 = self.model(X1, X2, padding_masks)  # (batch_size, padded_length, feat_dim)

            l1 = torch.sum(self.loss_module(predictions1, Y1, X1_mask)) + torch.sum(self.loss_module(predictions2, Y2, X2_mask))  # (num_active,) 
            l2 = self.l2_reg * l2_reg_loss(self.model)
            loss = l1 + l2

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            with torch.no_grad():
                total_samples += self.batch_size
                epoch_loss += l1.item()  # add total loss of batch
                rl += l2.item()
                rl += 0

        epoch_metrics['epoch'] = epoch_num
        epoch_metrics['loss'] = epoch_loss / total_samples
        epoch_metrics['rl'] = rl / total_samples

        return epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        total_loss = 0
        total_predictions = [[],[]]
        total_Y = [[],[]]
        total_samples = 0
        for i, batch in enumerate(self.valid_loader):
            X1, X2, Y1, Y2, X1_mask, X2_mask, padding_masks = batch
            predictions1, predictions2 = self.model(X1, X2, padding_masks)  # (batch_size, padded_length, feat_dim)
            total_predictions[0].append(predictions1.cpu().detach().numpy().tolist())
            total_predictions[1].append(predictions2.cpu().detach().numpy().tolist())
            current_loss = torch.sum(self.loss_module(predictions1, Y1, X1_mask)) + torch.sum(self.loss_module(predictions2, Y2, X2_mask))
            total_loss += current_loss.cpu().detach().numpy()
            total_samples += self.batch_size
            total_Y[0].append(Y1.cpu().detach().numpy().tolist())
            total_Y[1].append(Y2.cpu().detach().numpy().tolist())
        auc = total_loss / total_samples
        return auc, 0, total_predictions, total_Y, auc

class SupervisedRunner(BaseRunner):
    def __init__(self, *args, **kwargs):
        super(SupervisedRunner, self).__init__(*args, **kwargs)

    def train_epoch(self, epoch_num = None):
        self.model = self.model.train()
        epoch_loss = {'l1':0, 'l2':0}  # total loss of epoch
        total_samples = 0  # total samples in epoch
        epoch_metrics = {}

        for i, batch in enumerate(self.train_loader):
            X1, X2, Y, padding_masks = batch
            predictions = self.model(X1, X2, padding_masks=padding_masks)
            l1 = self.loss_module(predictions, Y)
            l2 = self.l2_reg * l2_reg_loss(self.model)
            
            loss = l1 + l2

            self.optimizer.zero_grad()
            l1.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            with torch.no_grad():
                total_samples += self.batch_size
                epoch_loss['l1'] += l1.item()
                epoch_loss['l2'] += l2.item()  
        
        epoch_metrics['epoch'] = epoch_num
        epoch_metrics['loss'] = epoch_loss['l1'] / total_samples
        epoch_metrics['rl'] = epoch_loss['l2'] / total_samples
        
        return epoch_metrics

    def evaluate(self, ):
        self.model = self.model.eval()
        total_Y = []
        total_predictions = []
        epoch_loss = {'loss': 0}
        total_samples = 0
        for i, batch in enumerate(self.valid_loader):
            X1, X2, Y, padding_masks = batch
            predictions = self.model(X1, X2, padding_masks)
            l1 = self.loss_module(predictions, Y)
            with torch.no_grad():
                total_samples += self.batch_size
                epoch_loss['loss'] += l1.item()  
            predictions = predictions.cpu().detach().numpy()
            total_Y += Y.cpu().detach().numpy().tolist()
            try:
                total_predictions += [math.exp(item[1])/(math.exp(item[0])+math.exp(item[1])) for item in predictions]
            except:
                total_predictions += [math.exp(item[1]-item[0])/(1+math.exp(item[1]-item[0])) if item[1]-item[0] < 700 else 1 for item in predictions]
        acc = accuracy_score(total_Y, [1 if item > 0.5 else 0 for item in total_predictions])
        if np.std(total_Y) > 0:
            auc = roc_auc_score(total_Y, total_predictions)
        else:
            auc = 1
        return auc, acc, total_predictions, total_Y, epoch_loss['loss']/total_samples
