import torch
import torch.nn as nn

from tqdm import tqdm

from data import zidian_xr, zidian_y, zidian_yr, loader
from mask import mask_pad, mask_tril
from model import Transformer

def predict(x):
    model.eval()
    mask_pad_x = mask_pad(x)

    target = [zidian_y['<SOS>']] + [zidian_y['<PAD>']] * 49
    target = torch.LongTensor(target).unsqueeze(0)

    x = model.embed_x(x)
    x = model.Encoder(x, mask_pad_x)

    for i in range(49):
        y = target.cuda()

        mask_tril_y = mask_tril(y)

        y = model.embed_y(y)

        y = model.Decoder(x, y, mask_pad_x, mask_tril_y)

        out = model.fc_out(y)

        out = out[:, i, :]

        out = out.argmax(dim = 1).detach()

        target[:, i + 1] = out

    return target

def main():
    loss_fn = nn.CrossEntropyLoss().cuda()
    optim = torch.optim.Adam(model.parameters(), lr = 2e-3)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size = 3, gamma = .5)

    for epoch in range(1):
        for x, y in tqdm(loader):
            x, y = x.cuda(), y.cuda()
            pred = model(x, y[:, :-1])

            pred = pred.reshape(-1, 39)

            y = y[:, 1:].reshape(-1)

            select = y != zidian_y['<PAD>']
            pred = pred[select]
            y = y[select]

            loss = loss_fn(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # if i % 200 == 0:
            #     pred = pred.argmax(1)
            #     correct = (pred == y).sum().item()
            #     accuracy = correct / len(pred)
            #     lr = optim.param_groups[0]['lr']
            #     print('epoch: {}, i: {}, lr: {}, loss: {}, accuracy: {}'.format(epoch, i, lr, loss.item(), accuracy))
        sched.step()
    
    for i, (x, y) in enumerate(loader):
        x, y = x.cuda(), y.cuda()
        break

    for i in range(8):
        print(i)
        print(''.join([zidian_xr[i] for i in x[i].tolist()]))
        print(''.join([zidian_yr[i] for i in y[i].tolist()]))
        print(''.join([zidian_yr[i] for i in predict(x[i].unsqueeze(0))[0].tolist()]))

if __name__ == '__main__':
    model = Transformer().cuda()
    main()