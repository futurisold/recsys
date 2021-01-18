from torch import save, load, device, sigmoid
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from tqdm import tqdm
from es import EarlyStopping
from data import uiDataset
from utils import auc_score
from mf import MF
import torch
import pickle


class Experiment:
    def __init__(self, config):
        self.cuda = config['cuda']
        # datasets
        self.train_dataset = uiDataset(*pickle.load(open('data/train_vectors', 'rb')))
        self.valid_dataset = uiDataset(*pickle.load(open('data/valid_vectors', 'rb')))
        # dataloaders
        self.train_loader = DataLoader(self.train_dataset, config['bs'], shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, config['bs'], shuffle=False)
        # setup
        self.model = MF(config).cuda() if self.cuda else MF(config).cpu()
        self.opt = Adam(self.model.parameters(), lr=config['lr'], weight_decay=config['wd'])
        self.crit = BCEWithLogitsLoss(reduction='sum')
        self.epochs = config['epochs']
        self.writer = SummaryWriter(comment=config['comment'], flush_secs=60)
        self.es = EarlyStopping(mode='max', patience=6, min_delta=5, percentage=True) if config['es'] else None
        self.path = 'checkpoints/model' + config['comment'] + '.pt'

    def train_batch(self, batch):
        users, items, targets = self.tensors_to_cuda(batch) if self.cuda else self.tensors_to_cpu(batch)
        self.opt.zero_grad()
        preds = self.model(users, items)
        loss = self.crit(preds.view(-1), targets)
        loss.backward()
        self.opt.step()
        return loss.item()

    def train_epoch(self, loader, idx):
        self.model.train()
        train_loss = 0
        for batch in loader:
            batch_loss = self.train_batch(batch)
            train_loss += batch_loss
        return train_loss / len(loader.dataset)

    def evaluate_epoch(self, loader, idx):
        self.model.eval()
        valid_loss = 0
        scores = defaultdict(list)
        with torch.no_grad():
            for batch in loader:
                users, items, targets = self.tensors_to_cuda(batch) if self.cuda else self.tensors_to_cpu(batch)
                preds = self.model(users, items)
                batch_loss = self.crit(preds.view(-1), targets)
                valid_loss += batch_loss.item()
                scores['targets'].extend(targets.data.cpu().flatten().numpy())
                scores['preds'].extend(sigmoid(preds).data.cpu().flatten().numpy())
        return valid_loss / len(loader.dataset), auc_score(scores)

    @property
    def fit(self):
        for epoch_idx in (tbar := tqdm(range(self.epochs))):
            train_loss = self.train_epoch(self.train_loader, epoch_idx)
            valid_loss, auc = self.evaluate_epoch(self.valid_loader, epoch_idx)
            self.writer.add_scalar('Loss/Train', train_loss, epoch_idx)
            self.writer.add_scalar('Loss/Validation', valid_loss, epoch_idx)
            self.writer.add_scalar('AUC/Validation', auc, epoch_idx)
            tbar.set_postfix(trLoss=train_loss, valLoss=valid_loss, AUC=auc)
            if self.es is not None:
                if self.es.step(auc):
                    self.checkpoint()
                    self.writer.flush()
                    print(f'Best: {auc}')
                    return
        self.checkpoint()
        self.writer.flush()

    def tensors_to_cuda(self, batch):
        u, i, s = batch
        return u.cuda(), i.cuda(), s.cuda()

    def tensors_to_cpu(self, batch):
        u, i, s = batch
        return u.cpu(), i.cpu(), s.cpu()

    def checkpoint(self, action='save', gpu=True):
        if action == 'save':
            save(self.model.state_dict(), self.path)
        elif action == 'load':
            if gpu:
                state_dict = load(self.path)
                self.model.load_state_dict(state_dict)
            else:
                state_dict = load(self.path, map_location=device('cpu'))
                self.model.load_state_dict(state_dict)
        else:
            raise NotImplementedError
