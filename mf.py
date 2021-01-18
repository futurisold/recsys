from torch import nn


class MF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.user_embedding = nn.Embedding(config['user_pool'], config['latent_dim'])
        self.item_embedding = nn.Embedding(config['item_pool'], config['latent_dim'])
        self.user_biases = nn.Embedding(config['user_pool'], 1)
        self.item_biases = nn.Embedding(config['item_pool'], 1)

    def forward(self, users, items):
        users_emb, items_emb = self.user_embedding(users), self.item_embedding(items)
        biases_u, biases_i = self.user_biases(users).squeeze(), self.item_biases(items).squeeze()
        dots = (users_emb * items_emb).sum(1)
        res = dots + biases_u + biases_i
        return res

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
