from data import Wrangler
from engine import Experiment
import pickle


PATH = '/data'


def main(path):
    data = Wrangler(path)
    to_disk('mappings', {'users': (data.user_from_int, data.user_to_int),
                         'items': (data.item_from_int, data.item_to_int)})
    config = {
        'user_pool': len(data.user_pool),
        'item_pool': len(data.item_pool),
        'latent_dim': 64,
        'train_dataset': data.train_dataset,
        'test_dataset': data.test_dataset,
        'test_dataframe': data.test_dataframe,
        'lr': 1.65e-3,
        'wd': 1e-7,
        'bs': 256,
        'epochs': 100,
        'cuda': True,
        'comment': '_beta_latent-64'
    }
    engine = Experiment(config)
    engine.fit


def to_disk(path, obj):
    pickle.dump(obj, open(path, 'wb'))


if __name__ == "__main__":
    main(PATH)
