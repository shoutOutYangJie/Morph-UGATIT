#-- coding:UTF-8 --
import torch
from torch import nn
from torch.nn import functional as F
try:
    from .base_networks import get_norm_layer, ResnetBlock
except:
    from base_networks import get_norm_layer, ResnetBlock
import functools
from torch.nn.utils import spectral_norm


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=True),
            nn.ReLU()
        )
        self.gamma = nn.Linear(dim, dim, bias=True)
        self.beta = nn.Linear(dim, dim, bias=True)
        self.dim = dim

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        x = x.view(b, c)
        x = self.mlp(x)
        gamma = self.gamma(x)
        beta = self.beta(x)
        return gamma, beta

class AdaLIN(nn.Module):
    def __init__(self, anime):
        super().__init__()
        self.eps = 1e-6
        self.rho = nn.Parameter(torch.FloatTensor(1).fill_(0.9))
        self.anime = anime
    def forward(self, x, gamma, beta):
        b,c,h,w = x.shape
        ins_mean = x.view(b,c, -1).mean(dim=2).view(b, c, 1, 1)
        ins_var = x.view(b,c,-1).var(dim=2) + self.eps
        ins_std = ins_var.sqrt().view(b,c ,1, 1)

        x_ins = (x - ins_mean) / ins_std

        ln_mean = x.view(b, -1).mean(dim=1).view(b, 1, 1, 1)
        ln_val = x.view(b, -1).var(dim=1).view(b, 1, 1, 1) + self.eps
        ln_std = ln_val.sqrt()

        x_ln = (x - ln_mean) / ln_std
        if self.anime:
            rho = self.rho  # for sel->anime
        else:
            rho = (self.rho - 0.1).clamp(0, 1.0)  # smoothing, for adult->age
        x_hat = rho * x_ins + (1-rho) * x_ln
        x_hat = x_hat * gamma + beta
        return x_hat

class ResBlockByAdaLIN(nn.Module):
    def __init__(self, dim, anime=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0),
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0),
        )
        self.addin_1 = AdaLIN(anime)
        self.addin_2 = AdaLIN(anime)
        self.relu = nn.ReLU()

    def forward(self, x, gamma, beta):
        x1 = self.conv1(x)
        x1 = self.relu(self.addin_1(x1, gamma, beta))

        x2 = self.conv2(x1)
        x2 = self.addin_2(x2, gamma, beta)
        return x + x2


class LayerInstanceNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.eps = 1e-6
        self.gamma = nn.Parameter(torch.FloatTensor(1, dim, 1, 1).fill_(1.0))
        self.beta = nn.Parameter(torch.FloatTensor(1, dim, 1, 1).fill_(0.0))
        self.rho = nn.Parameter(torch.FloatTensor(1, dim, 1, 1).fill_(0.0))

    def forward(self, x):
        b, c, h, w = x.shape
        ins_mean = x.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        ins_val = x.view(b, c, -1).var(dim=2).view(b, c, 1, 1) + self.eps
        ins_std = ins_val.sqrt()

        ln_mean = x.view(b, -1).mean(dim=1).view(b, 1, 1, 1)
        ln_val = x.view(b, -1).var(dim=1).view(b, 1, 1, 1) + self.eps
        ln_std = ln_val.sqrt()

        rho = torch.clamp(self.rho, 0, 1)
        x_ins = (x - ins_mean) / ins_std
        x_ln = (x - ln_mean) / ln_std

        x_hat = rho * x_ins + (1 - rho) * x_ln
        return x_hat * self.gamma + self.beta


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc,
                 ngf=64,
                 use_dropout=False, n_blocks=4,
                 padding_type='reflect', anime=False):
        assert (n_blocks >= 0)
        super(Generator, self).__init__()
        self.n_blocks = n_blocks
        instance_norm_layer = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=False)
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(ngf, affine=True),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(ngf * mult * 2, affine=True),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=instance_norm_layer, use_dropout=use_dropout,
                                  use_bias=True)]
        self.encoder = nn.Sequential(*model)

        # CAM
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.cam_w = nn.Parameter(torch.FloatTensor(ngf*mult, 1))
        nn.init.xavier_uniform_(self.cam_w)
        self.cam_bias = nn.Parameter(torch.FloatTensor(1))
        self.cam_bias.data.fill_(0)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(2*ngf*mult, ngf*mult, 1, 1),
            nn.ReLU(),
        )
        # MLP
        self.mlp = MLP(ngf*mult)

        adain_resblock = []
        for i in range(n_blocks):
            adain_resblock.append(ResBlockByAdaLIN(ngf*mult, anime))
        self.adain_resblocks = nn.ModuleList(adain_resblock)

        decoder = []
        for i in range(n_downsampling):
            decoder.append(nn.UpsamplingBilinear2d(scale_factor=2))
            decoder.append(nn.ReflectionPad2d(1))
            decoder.append(nn.Conv2d(ngf*mult, ngf*mult//2, 3, 1, 0))
            decoder.append(LayerInstanceNorm(ngf*mult//2))
            decoder.append(nn.ReLU())

            mult = mult // 2
        decoder.extend([nn.ReflectionPad2d(3),
                       nn.Conv2d(ngf, output_nc, 7, 1),
                       nn.Tanh()]
                       )
        self.decoder = nn.Sequential(*decoder)

    def cam(self, e):
        b, c, h, w = e.shape
        gap = self.gap(e).view(b, c)
        gmp = self.gmp(e).view(b, c)

        x_a = torch.matmul(gap, self.cam_w) + self.cam_bias  # for classfication loss
        x_m = torch.matmul(gmp, self.cam_w) + self.cam_bias

        x_gap = e * (self.cam_w + self.cam_bias).view(1, c, 1, 1)
        x_gmp = e * (self.cam_w + self.cam_bias).view(1, c, 1, 1)

        x = torch.cat((x_gap, x_gmp), dim=1)
        x = self.conv1x1(x)
        x_class = torch.cat((x_a, x_m), dim=1) # b, 2
        return x, x_class

    def forward(self, img):
        e = self.encoder(img)
        x, x_class = self.cam(e)
        b, c, h, w = x.shape
        gamma, beta = self.mlp(self.gap(x).view(b, c))  # 预测adain的beta, gamma
        gamma = gamma.view(b, c, 1, 1)
        beta = beta.view(b, c, 1, 1)
        for i in range(self.n_blocks):
            x = self.adain_resblocks[i](x, gamma, beta)

        x = self.decoder(x)
        return x, x_class

    def test_forward(self, x, dir=None):
        return self.forward(x)[0]


class CAM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.weight = nn.Parameter(torch.FloatTensor(dim, 1))
        nn.init.xavier_uniform_(self.weight)
        self.cam_bias = nn.Parameter(torch.FloatTensor(1))
        self.cam_bias.data.fill_(0)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1, 1),
            nn.LeakyReLU(0.2),
        )
    def forward(self, e):
        b, c, h, w = e.shape
        gap = self.gap(e).view(b, c)
        gmp = self.gmp(e).view(b, c)

        x_a = torch.matmul(gap, self.weight) + self.cam_bias  # for classfication loss
        x_m = torch.matmul(gmp, self.weight) + self.cam_bias

        x_gap = e * (self.weight + self.cam_bias).view(1, c, 1, 1)
        x_gmp = e * (self.weight + self.cam_bias).view(1, c, 1, 1)

        x = torch.cat((x_gap, x_gmp), dim=1)
        x = self.conv1x1(x)
        x_class = torch.cat((x_a, x_m), dim=1)  # b, 2
        return x, x_class

class Discriminator(nn.Module):
    def __init__(self, inc, ndf, n_blocks=6):
        super().__init__()
        # local
        local = [
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(inc, ndf, 4, 2)),
            nn.LeakyReLU(0.2),
        ]
        for i in range(1, n_blocks-2-1):
            mult = 2 ** (i-1)
            local.extend([
                nn.ReflectionPad2d(1),
                spectral_norm(nn.Conv2d(ndf*mult, ndf*mult*2, 4, 2)),
                nn.LeakyReLU(0.2),
            ])
        mult = mult * 2
        local.extend([
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(ndf*mult, ndf*mult*2, 4, 1)),
            nn.LeakyReLU(0.2)
        ])
        self.local_base = nn.Sequential(*local)
        mult = mult * 2
        self.local_cam = spectral_norm(CAM(mult*ndf))
        self.local_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(mult*ndf, 1, 4, 1)),
        )

        # global
        global_ = [
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(inc, ndf, 4, 2)),
            nn.LeakyReLU(0.2),
        ]
        for i in range(1, n_blocks-1):
            mult = 2 ** (i - 1)
            global_.extend([
                nn.ReflectionPad2d(1),
                spectral_norm(nn.Conv2d(ndf * mult, ndf * mult * 2, 4, 2)),
                nn.LeakyReLU(0.2),
            ])
        mult *= 2
        global_.extend([
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(ndf * mult, ndf * mult * 2, 4, 1)),
            nn.LeakyReLU(0.2)
        ])
        mult *= 2
        self.global_base = nn.Sequential(*global_)
        self.global_cam = spectral_norm(CAM(mult*ndf))
        self.global_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(mult * ndf, 1, 4, 1)),
        )

    def forward(self, img):
        local_base = self.local_base(img)
        local_x, local_class = self.local_cam(local_base)
        local_x = self.local_head(local_x)

        global_base = self.global_base(img)
        global_x, global_class = self.global_cam(global_base)
        global_x = self.global_head(global_x)
        # print(local_x.shape, global_x.shape, global_class.shape)
        return local_x, local_class, global_x, global_class



class UGATIT(object):
    def __init__(self, args):
        super().__init__()
        self.G_A = Generator(args.inc, args.outc, args.ngf, args.use_dropout, args.n_blocks, anime=args.anime)
        self.G_B = Generator(args.inc, args.outc, args.ngf, args.use_dropout, args.n_blocks, anime=args.anime)
        if args.training:
            self.D_A = Discriminator(args.inc, args.ndf, args.d_layers)
            self.D_B = Discriminator(args.inc, args.ndf, args.d_layers)
        self.training = args.training

    def __call__(self, inp):
        realA, realB = inp['A'], inp['B']
        if torch.cuda.is_available():
            realA, realB = realA.cuda(), realB.cuda()
        fakeB, cam_ab = self.G_A(realA)
        fakeA, cam_ba = self.G_B(realB)

        recA, _ = self.G_B(fakeB)
        recB, _ = self.G_A(fakeA)

        return realA, realB, fakeA, fakeB, recA, recB, cam_ab, cam_ba

    def train(self):
        self.D_A.train()
        self.G_A.train()
        self.D_B.train()
        self.G_B.train()

    def cuda(self):
        self.D_A.cuda()
        self.G_A.cuda()
        if self.training:
            self.D_B.cuda()
            self.G_B.cuda()

    def test_forward(self, x, dir):
        raise NotImplementedError


    def state_dict(self):
        params = {
            'G_A': self.G_A.module.state_dict(), # dist training
            'G_B': self.G_B.module.state_dict(),
        }
        if self.training:
            params['D_A'] = self.D_A.module.state_dict(),
            params['D_B'] = self.D_B.module.state_dict()
        return params

    def load_state_dict(self, weight_loc):
        weight_set = torch.load(weight_loc, map_location='cpu')
        self.G_A.load_state_dict(weight_set['G_A'])
        self.G_B.load_state_dict(weight_set['G_B'])
        if self.training:
            self.D_A.load_state_dict(weight_set['D_A'])
            self.D_B.load_state_dict(weight_set['D_B'])


if __name__ == '__main__':
    from utils.measure_model import measure_model
    # model = UGATIT(3, 3, 64, True, n_blocks=4)  # 52G
    # model.eval()
    model = Discriminator(3, 64, 6)  # 7G
    x = torch.randn(1, 3, 256, 256)
    inp = {'A':x, 'B':x}
    flops, _, _ = measure_model(model, inp=[x])
    print(flops / 1e6)








