from gmf import GMF
from torch import sigmoid, save, load, device
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from es import EarlyStopping
import torch


class Experiment:
    def __init__(self, config):
        self.cuda = config['cuda']
        self.model = GMF(config).cuda() if self.cuda else GMF(config).cpu()
        self.opt = Adam(self.model.parameters(), lr=config['lr'], weight_decay=config['wd'])
        self.crit = BCEWithLogitsLoss()
        self.train_loader = DataLoader(config['train_dataset'], config['bs'], shuffle=True)
        self.test_loader = DataLoader(config['test_dataset'], config['bs'], shuffle=False)
        self.test_dataframe = config['test_dataframe']
        self.epochs = config['epochs']
        self.writer = SummaryWriter(comment=config['comment'], flush_secs=60)
        self.path = 'checkpoints/model' + config['comment'] + '.pt'
        self.es = EarlyStopping(mode='max', patience=5, min_delta=1, percentage=True)

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
        return train_loss

    def evaluate_epoch(self, loader, idx):
        self.model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in loader:
                users, items, _ = self.tensors_to_cuda(batch) if self.cuda else self.tensors_to_cpu(batch)
                preds = self.model(users, items)
                val_preds.extend(sigmoid(preds.view(-1)).tolist())
        self.test_dataframe['preds'] = val_preds
        hr = self.hit_ratio
        return hr

    @property
    def fit(self):
        for epoch_idx in (tbar := tqdm(range(self.epochs))):
            train_loss = self.train_epoch(self.train_loader, epoch_idx)
            hr = self.evaluate_epoch(self.test_loader, epoch_idx)
            self.writer.add_scalar('Loss/Train', train_loss, epoch_idx)
            self.writer.add_scalar('HR@K/Validation', hr, epoch_idx)
            tbar.set_postfix(Loss=train_loss, hr=hr)
            if self.es.step(hr):
                self.checkpoint()
                self.writer.flush()
                print(f'Best: {hr * 100:.2f}%')
                break

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

    @property
    def hit_ratio(self):
        dataframe = self.test_dataframe.copy()
        dataframe['ranking'] = dataframe.groupby('userId').preds.rank(method='first', ascending=False)
        top_k = dataframe[dataframe.ranking <= 10]
        hr = len(top_k[top_k.target == 1]) / dataframe.userId.nunique()
        return hr

    @property
    def overfit_batch(self):
        for epoch_idx in (tbar := tqdm(range(self.epochs))):
            batch = next(iter(self.train_loader))
            batch_loss = self.train_batch(batch)
            self.writer.add_scalar('Loss/Batch', batch_loss, epoch_idx)
            tbar.set_postfix(Loss=batch_loss)
