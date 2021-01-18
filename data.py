from tqdm import tqdm
from collections import defaultdict
from random import sample, seed
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import pickle


seed(0)
PATH = 'data/'


class DataPrep:
    def __init__(self, path, save=True):
        self.sample_data = self.wrangling(path)
        # get the user pool & item pool
        self.item_pool = set(self.sample_data.itemId)
        self.user_pool = set(self.sample_data.userId)
        # mappings
        self.item_to_int, self.item_from_int = self.mapping(self.sample_data.id_anunt.unique())
        self.user_to_int, self.user_from_int = self.mapping(self.sample_data.imovzt.unique())
        # train/validation
        self.train, self.test = self.leave_one_out_cross_validation(self.sample_data)
        # get interactions
        self.train_inter, self.valid_inter, self.test_inter, self.hr_inter = self.get_train_valid_test_data(self.train, self.test)
        # get pairs
        self.train_vectors = self.get_pairs(self.train_inter)
        self.valid_vectors = self.get_pairs(self.valid_inter)
        self.test_vectors = self.get_pairs(self.test_inter)
        self.hr_vectors = self.get_pairs(self.hr_inter)
        if save:
            pickle.dump({'item_to_int': self.item_to_int, 'item_from_int': self.item_from_int}, open('data/item_mappings', 'wb'))
            pickle.dump({'user_to_int': self.user_to_int, 'user_from_int': self.user_from_int}, open('data/user_mappings', 'wb'))
            pickle.dump(self.sample_data, open('data/dataset', 'wb'))
            pickle.dump(self.train_vectors, open('data/train_vectors', 'wb'))
            pickle.dump(self.valid_vectors, open('data/valid_vectors', 'wb'))
            pickle.dump(self.test_vectors, open('data/test_vectors', 'wb'))
            pickle.dump(self.hr_vectors, open('data/hr_vectors', 'wb'))

    def wrangling(self, fname):
        data = pd.read_csv(fname, usecols=['imovzt', 'id_anunt'])
        data['imovzt'] = pd.to_numeric(data.imovzt, errors='coerce')
        data.dropna(inplace=True)
        user_pool = self.have_enough_items(data)
        data = data.set_index('imovzt').loc[user_pool].reset_index()
        item_to_int, _ = self.mapping(data.id_anunt.unique())
        user_to_int, _ = self.mapping(data.imovzt.unique())
        data['itemId'] = data.id_anunt.apply(lambda x: item_to_int[x])
        data['userId'] = data.imovzt.apply(lambda x: user_to_int[x])
        data.drop_duplicates(inplace=True)
        return data

    def have_enough_items(self, data, bound=2):
        data = data.groupby('imovzt').agg({'id_anunt': set})
        data = data[data['id_anunt'].apply(len) >= bound]
        return data.index.tolist()

    def mapping(self, vec):
        to_int = {vec[i]: i for i in range(len(vec))}
        from_int = {v: k for k, v in to_int.items()}
        return to_int, from_int

    def leave_one_out_cross_validation(self, data):
        data['ranking'] = data.groupby('userId').itemId.rank(method='first')
        train = data[data.ranking > 1]
        test = data[data.ranking == 1]
        return train[['userId', 'itemId']], test[['userId', 'itemId']]

    def get_train_valid_test_data(self, train, test, n=4, sz=.1, hr=True):
        train.set_index('userId', inplace=True)
        test.set_index('userId', inplace=True)
        # pools
        user_pool = self.user_pool.copy()
        user_test_pool = set(sample(user_pool, int(sz * len(user_pool))))
        user_valid_pool = user_pool - user_test_pool
        item_pool = self.item_pool.copy()
        # interactions
        train_interactions = defaultdict(lambda: defaultdict(set))
        valid_interactions = defaultdict(lambda: defaultdict(set))
        test_interactions = defaultdict(lambda: defaultdict(set))
        hr_interactions = defaultdict(lambda: defaultdict(set))
        for user in tqdm(user_pool):
            _ = train.loc[user]
            positives = set(_.itemId) if not isinstance(_.itemId, np.int64) else set([_.itemId])
            available = item_pool-positives
            negatives = sample(available, len(positives) * n)
            train_interactions[user]['positives'].update(positives)
            train_interactions[user]['negatives'].update(negatives)
        for user in tqdm(user_valid_pool):
            _ = test.loc[user]
            positives = set([_.itemId])
            available = item_pool - positives - train_interactions[user]['positives'] - train_interactions[user]['negatives']
            negatives = sample(available, len(positives) * n)
            valid_interactions[user]['positives'].update(positives)
            valid_interactions[user]['negatives'].update(negatives)
        for user in tqdm(user_test_pool):
            _ = test.loc[user]
            positives = set([_.itemId])
            available = item_pool - positives - train_interactions[user]['positives'] - train_interactions[user]['negatives']
            negatives = sample(available, len(positives) * n)
            test_interactions[user]['positives'].update(positives)
            test_interactions[user]['negatives'].update(negatives)
        if not hr:
            return train_interactions, valid_interactions, test_interactions
        else:
            for user in tqdm(user_valid_pool):
                _ = test.loc[user]
                positives = set([_.itemId])
                negatives = sample(item_pool-positives, 99)
                hr_interactions[user]['positives'].update(positives)
                hr_interactions[user]['negatives'].update(negatives)
            return train_interactions, valid_interactions, test_interactions, hr_interactions

    def get_pairs(self, dic):
        interactions = []
        for user in tqdm(dic.keys()):
            for p in dic[user]['positives']:
                interactions.append((int(user), int(p), 1.))
            for n in dic[user]['negatives']:
                interactions.append((int(user), int(n), 0.))
        users, items, targets = zip(*interactions)
        return users, items, targets


class uiDataset(Dataset):
    def __init__(self, users, items, targets):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.targets = torch.FloatTensor(targets)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.targets[idx]

    def __len__(self):
        return self.users.size(0)


if __name__ == "__main__":
    data = DataPrep(PATH)
