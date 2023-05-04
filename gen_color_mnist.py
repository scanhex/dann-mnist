# Modified version of https://github.com/salesforce/corr_based_prediction/blob/master/gen_color_mnist.py

import torchvision
import torch
import numpy as np
import torchvision.transforms as transforms
import tqdm
from torch.autograd import Variable
import argparse
import os

from typing import Tuple

data_path = './datasets/'

parser = argparse.ArgumentParser(description='Generate colored MNIST')

# Hyperparams
parser.add_argument('--cpr', nargs='+', type=float, default=[0.5, 0.5],
                    help='color choice is made corresponding to a class with these probability')
args = parser.parse_args()

trans = ([transforms.ToTensor()])
trans = transforms.Compose(trans)
fulltrainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=trans)
trainloader = torch.utils.data.DataLoader(fulltrainset, batch_size=2000, shuffle=False, num_workers=2, pin_memory=True)
test_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=trans)
testloader = torch.utils.data.DataLoader(test_set, batch_size=2000, shuffle=False, num_workers=2, pin_memory=True)
nb_classes = 10

COLOR_CODES = [
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.],
]
colors_used = 0


# generate color codes
def get_color_codes(cnt):
    global colors_used
    # use the same color for every class
    res = np.repeat(np.array(COLOR_CODES[colors_used:colors_used + cnt])[:, np.newaxis, :], nb_classes, axis=1)
    colors_used += cnt
    return np.array(res)
    """
    C = np.random.rand(cnt, nb_classes, 3)
    C = C / np.max(C, axis=2)[:, :, None]
    print(C.shape)
    return C
    """


def gen_fgbgcolor_data(loader: torch.utils.data.DataLoader, img_size: Tuple[int, int, int], cpr: Tuple[int], noise=10.):
    assert sum(cpr) == 1, '--cpr must be a non-negative list which sums to 1'
    Cfg = get_color_codes(len(cpr))
    # Cbg = get_color_codes(cpr)
    Cbg = np.ones((len(cpr), nb_classes, 3))
    tot_iters = len(loader)
    for i in tqdm.tqdm(range(tot_iters), total=tot_iters):
        x, targets = next(iter(loader))
        assert len(
            x.size()) == 4, 'Something is wrong, size of input x should be 4 dimensional (B x C x H x W; perhaps number of channels is degenrate? If so, it should be 1)'
        targets = targets.cpu().numpy()
        bs = targets.shape[0]

        x = (((x * 255) > 150) * 255).type('torch.FloatTensor')
        x_rgb = torch.ones(x.size(0), 3, x.size()[2], x.size()[3]).type('torch.FloatTensor')
        x_rgb = x_rgb * x
        x_rgb_fg = 1. * x_rgb

        color_choice = np.argmax(np.random.multinomial(1, cpr, targets.shape[0]), axis=1)
        c = Cfg[color_choice, targets]
        c = c.reshape(-1, 3, 1, 1)
        c = torch.from_numpy(c).type('torch.FloatTensor')
        x_rgb_fg[:, 0] = x_rgb_fg[:, 0] * c[:, 0]
        x_rgb_fg[:, 1] = x_rgb_fg[:, 1] * c[:, 1]
        x_rgb_fg[:, 2] = x_rgb_fg[:, 2] * c[:, 2]

        bg = (255 - x_rgb)
        # c = C[targets] if np.random.rand()>cpr else C[np.random.randint(C.shape[0], size=targets.shape[0])]
        color_choice = np.argmax(np.random.multinomial(1, cpr, targets.shape[0]), axis=1)
        c = Cbg[color_choice, targets]
        c = c.reshape(-1, 3, 1, 1)
        c = torch.from_numpy(c).type('torch.FloatTensor')
        bg[:, 0] = bg[:, 0] * c[:, 0]
        bg[:, 1] = bg[:, 1] * c[:, 1]
        bg[:, 2] = bg[:, 2] * c[:, 2]
        x_rgb = x_rgb_fg + bg
        x_rgb = x_rgb + torch.tensor(noise * np.random.randn(*x_rgb.size())).type('torch.FloatTensor')
        x_rgb = torch.clamp(x_rgb, 0., 255.)
        if i == 0:
            color_data_x = np.zeros((bs * tot_iters, *img_size))
            color_data_y = np.zeros((bs * tot_iters,))
        color_data_x[i * bs: (i + 1) * bs] = x_rgb / 255.
        color_data_y[i * bs: (i + 1) * bs] = targets
    return color_data_x, color_data_y


dir_name = data_path + 'cmnist/' + 'fgbg_cmnist_cpr' + '-'.join(str(p) for p in args.cpr) + '/'
print(dir_name)
if not os.path.exists(data_path + 'cmnist/'):
    os.mkdir(data_path + 'cmnist/')
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

color_data_x, color_data_y = gen_fgbgcolor_data(trainloader, img_size=(3, 28, 28), cpr=args.cpr, noise=10.)
np.save(dir_name + '/train_x.npy', color_data_x)
np.save(dir_name + '/train_y.npy', color_data_y)

color_data_x, color_data_y = gen_fgbgcolor_data(testloader, img_size=(3, 28, 28), cpr=(1,), noise=10.)
np.save(dir_name + 'test_x.npy', color_data_x)
np.save(dir_name + 'test_y.npy', color_data_y)
