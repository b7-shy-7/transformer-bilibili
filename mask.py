import torch

from data import zidian_x, zidian_y

def mask_pad(data):
    mask = data == zidian_x['<PAD>']
    # input shape: [b, 50]
    mask = mask.reshape(-1, 1, 1, 50)

    mask = mask.expand(-1, 1, 50, 50)

    return mask

def mask_tril(data):
    tril = 1 - torch.tril(torch.ones(1, 50, 50, dtype = torch.long))
    tril = tril.cuda()
    mask = data == zidian_y['<PAD>']
    mask = mask.unsqueeze(1).long()

    mask = mask + tril

    mask = mask > 0
    # 这个==1是干嘛的
    mask = (mask == 1).unsqueeze(dim = 1)

    return mask


if __name__ == '__main__':
    tril = 1 - torch.tril(torch.ones(1, 5, 5, dtype = torch.long))
    t1 = torch.tensor([
        0 for i in range(3)
    ])
    t2 = torch.tensor([
        1 for i in range(2)
    ])
    t = torch.cat((t1, t2), dim = 0)
    out = (tril + t) > 0
    # print(out == (out == 1).unsqueeze(dim = 1))
    out = (out == 1).unsqueeze(dim = 1)
    # print(out)