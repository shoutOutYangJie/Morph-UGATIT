from .base_networks import define_D, define_G
import torch
from torch import nn
# from .image_pool import ImagePool

class CycleGANModel(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.netG_A = define_G(args.inc, args.outc, args.ngf, args.norm, args.use_dropout)
        self.netG_B = define_G(args.outc, args.inc, args.ngf, args.norm, args.use_dropout)

        if args.training:
            self.netD_A = define_D(args.outc, args.ndf, args.d_layers, args.norm)
            self.netD_B = define_D(args.outc, args.ndf, args.d_layers, args.norm)


    def __call__(self, inp):
        AtoB = self.args.direction == 'AtoB'
        real_A = inp['A' if AtoB else 'B']
        real_B = inp['B' if AtoB else 'A']

        if torch.cuda.is_available():
            real_A, real_B = real_A.cuda(), real_B.cuda()

        fake_B = self.netG_A(real_A)  # G_A(A)
        rec_A = self.netG_B(fake_B)  # G_B(G_A(A))
        fake_A = self.netG_B(real_B)  # G_B(B)
        rec_B = self.netG_A(fake_A)  # G_A(G_B(B))

        return real_A, real_B, fake_A, fake_B, rec_A, rec_B

    def train(self):
        self.netD_A.train()
        self.netG_A.train()
        self.netD_B.train()
        self.netG_B.train()

    def cuda(self):
        self.netD_A.cuda()
        self.netG_A.cuda()
        self.netD_B.cuda()
        self.netG_B.cuda()

    def test_forward(self, inp):
        pass

    def load_state_dict(self):
        pass

if __name__ == '__main__':
    from easydict import EasyDict
    from utils.measure_model import measure_model
    A = torch.randn(1, 3, 256, 256)
    B = torch.randn(1, 3, 256, 256)

    cfgs = {'inc': 3,
            'outc': 3,
            'ngf': 64,
            'ndf': 64,
            'norm': 'instance',
            'use_dropout': True,
            'd_layers': 3,
            'direction': 'AtoB',
            'training':True
            }
    cfgs = EasyDict(cfgs)

    model = CycleGANModel(cfgs)

    inp = {'A':A, 'B':B}
    # outp = model(inp)
    # for o in outp:
    #     print(o.shape)

    flops, _, _ = measure_model(model, inp=[inp])
    print(flops / 1e6 / 4)  # 不到50G
