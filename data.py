from tqdm import tqdm
from collections import defaultdict
from random import sample, seed
from torch.utils.data import Dataset
import pandas as pd
import torch


seed(0)
PATH = '/data'


class Wrangler:
    def __init__(self, fname):
        self.sample_data = self.wrangling(fname)
        # get the user pool & item pool
        self.item_pool = set(self.sample_data.itemId)
        self.user_pool = set(self.sample_data.userId)
        # save the mappings
        self.item_to_int, self.item_from_int = self.mapping(self.sample_data.id_anunt.unique())
        self.user_to_int, self.user_from_int = self.mapping(self.sample_data.imovzt.unique())
        # get interactions
        self.train, self.test = self.leave_one_out_cross_validation(self.sample_data)
        # get train & test data
        self.train_data = self.get_train_data(self.train.copy(), self.item_pool)
        self.test_data = self.get_test_data(self.test.copy(), self.item_pool)
        # convert to torch dataset
        self.train_dataset = self.to_dataset(self.train_data)
        self.test_dataset = self.to_dataset(self.test_data)
        # on disk test_dataframe
        self.test_dataframe = self.test_hit_ratio()

    def wrangling(self, fname):
        data = pd.read_csv(fname, usecols=['imovzt', 'id_anunt'])
        data['imovzt'] = pd.to_numeric(data.imovzt, errors='coerce')
        data.dropna(inplace=True)
        user_pool = self.have_enough_items(data)
        data = data.set_index('imovzt').loc[user_pool].reset_index()
        item_to_int, item_from_int = self.mapping(data.id_anunt.unique())
        user_to_int, user_from_int = self.mapping(data.imovzt.unique())
        data['itemId'] = data.id_anunt.apply(lambda x: item_to_int[x])
        data['userId'] = data.imovzt.apply(lambda x: user_to_int[x])
        data['target'] = 1
        return data.drop_duplicates()

    def have_enough_items(self, data, bound=3):
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
        return train[['userId', 'itemId', 'target']], test[['userId', 'itemId', 'target']]

    def get_train_data(self, data, item_pool, n=4):
        interactions = defaultdict(lambda: defaultdict(set))
        data.set_index('userId', inplace=True)
        user_pool = data.index.unique()
        for user in tqdm(user_pool):
            _ = data.loc[user]
            positives = set(_.itemId)
            negatives = sample(item_pool-positives, n*len(positives))
            interactions[user]['positives'].update(positives)
            interactions[user]['negatives'].update(negatives)
        return interactions

    def get_test_data(self, data, item_pool, n=99):
        interactions = defaultdict(lambda: defaultdict(set))
        data.set_index('userId', inplace=True)
        user_pool = data.index.unique()
        for user in tqdm(user_pool):
            _ = data.loc[user]
            positives = set([_.itemId])
            negatives = sample(item_pool-positives, n)
            interactions[user]['positives'].update(positives)
            interactions[user]['negatives'].update(negatives)
        return interactions

    def test_hit_ratio(self):
        users, items, targets = self.test_dataset[:]
        return pd.DataFrame({'userId': users, 'itemId': items, 'target': targets})

    def to_dataset(self, dic):
        interactions = []
        for user in tqdm(dic.keys()):
            for p in dic[user]['positives']:
                interactions.append((int(user), int(p), 1.))
            for n in dic[user]['negatives']:
                interactions.append((int(user), int(n), 0.))
        users, items, targets = zip(*interactions)
        return uiDataset(users, items, targets)


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
    data = Wrangler(PATH)
