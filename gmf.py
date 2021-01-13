from torch import nn
from torch import mul


class GMF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.user_embedding = nn.Embedding(config['user_pool'], config['latent_dim'])
        self.item_embedding = nn.Embedding(config['item_pool'], config['latent_dim'])
        self.out = nn.Linear(config['latent_dim'], 1)

    def forward(self, users, items):
        users, items = self.user_embedding(users), self.item_embedding(items)
        haddamard = mul(users, items)
        return self.out(haddamard)

    @property
    def summarize(self):
        print("Model summary")
        print("="*60)
        layers = list(self.children())
        total = 0
        for layer in layers:
            for params in layer.parameters():
                if params.requires_grad:
                    total += params.numel()
        print(self)
        print("="*60)
        print(f"Trainable parameters: {total}")
