from engine import Experiment
import json


def main():
    config = {
        'user_pool': 96249,
        'item_pool': 25874,
        'latent_dim': 32,
        'lr': 3e-4,
        'wd': 1e-4,
        'bs': 256,
        'epochs': 50,
        'cuda': True,
        'es': False,
        'comment': '_lmf_50_4negs'
    }
    json.dump(config, open('data/config.json', 'w'))
    engine = Experiment(config)
    engine.fit


if __name__ == "__main__":
    main()
