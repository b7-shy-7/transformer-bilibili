import torch
import torch.nn as nn

import math
import numpy as np

def attention(Q, K, V, mask):
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))

    score /= 8 ** 0.5

    # print(score)
    score = score.masked_fill(mask, -float('inf'))
    # print(score)
    score = torch.softmax(score, dim = -1)

    score = torch.matmul(score, V)

    score = score.permute(0, 2, 1, 3).reshape(-1, 50, 32)

    return score

class MultiHead(nn.Module):
    def __init__(self):
        super(MultiHead, self).__init__()

        self.fc_Q = nn.Linear(32, 32)
        self.fc_K = nn.Linear(32, 32)
        self.fc_V = nn.Linear(32, 32)

        self.out_fc = nn.Linear(32, 32)

        self.norm = nn.LayerNorm(normalized_shape = 32, elementwise_affine = True)

        self.dropout = nn.Dropout(p = 0.1)

    def forward(self, Q, K, V, mask):
        b = Q.shape[0]

        clone_Q = Q.clone()

        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        K = self.fc_K(K)
        Q = self.fc_Q(Q)
        V = self.fc_V(V)

        Q = Q.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        K = K.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        V = V.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)

        score = attention(Q, K, V, mask)

        score = score + clone_Q

        return score
    
class PositionEmbedding(nn.Module):
    def __init__(self):
        super(PositionEmbedding, self).__init__()
        
        def get_pe(pos, i, d_model):
            pe = pos / 1e4 ** (i / d_model)

            if i%2 == 0:
                return math.sin(pe)
            return math.cos(pe)
        
        pe = torch.empty(50, 32)
        for i in range(50):
            for j in range(32):
                pe[i, j] = get_pe(i, j, 32)
        # 这两行是干嘛的，没看懂
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.embed = torch.nn.Embedding(39, 32)
        # 这个初始化是啥意思
        self.embed.weight.data.normal_(0, 0.1)

    def forward(self, x):
        embed = self.embed(x)

        embed = embed + self.pe
        return embed
    
class FullyConnectedOutput(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features = 32, out_features = 64),
            nn.ReLU(),
            nn.Linear(in_features = 64, out_features = 32),
            nn.Dropout(p = 0.1)
        )

        self.norm = nn.LayerNorm(normalized_shape = 32,
                                 elementwise_affine = True)
        
    def forward(self, x):
        clone_x = x.clone()

        x = self.norm(x)

        out = self.fc(x)

        out = out + clone_x
        return out

if __name__ == '__main__':
    # 测试multihead
    # Q = torch.tensor(np.random.random((10, 50, 32)), dtype = torch.float32)
    # K = torch.tensor(np.random.random((10, 50, 32)), dtype = torch.float32)
    # V = torch.tensor(np.random.random((10, 50, 32)), dtype = torch.float32)
    # mask = torch.tensor(np.random.randint(-1, 1, (50, 50)), dtype = bool)

    # print(mask)

    # multi = MultiHead()

    # print(multi(Q, K, V, mask))

    # 测试pe
    list_alpha = torch.LongTensor(np.random.random((10, 50)))
    # print(list_alpha)
    embed_alpha = PositionEmbedding()
    print(embed_alpha(list_alpha))