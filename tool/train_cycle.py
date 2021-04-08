import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import torch
from models.cycle_gan import CycleGANModel
from models.image_pool import ImagePool
from models.loss import GANLoss
import itertools
from utils.utils import AvgMeter, init_path, get_scheduler
from torch.utils import tensorboard
import os
from torchvision.utils import make_grid
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from configs.cfgs_train import cfgs
from datasets.unpair_dataset import UnpairDataset
from torch.utils.data import DataLoader


class Train:
    def __init__(self, model, loader, sample, args):
        self.model = model
        self.loader = loader
        self.sample = sample
        self.args = args
        self.rank = args.rank
        self.each_epoch_iters = len(loader)
        self.max_iters = self.each_epoch_iters * args.total_epoch
        if args.rank == 0:
            self.td = tensorboard.SummaryWriter(args.tensorboard)

        self.init()

    def init(self):
        opt = self.args
        if not os.path.exists(opt.saved_dir):
            os.makedirs(opt.saved_dir)
        self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
        self.fake_B_pool = ImagePool(opt.pool_size)
        self.crit_cycle = torch.nn.L1Loss()
        self.crit_idt = torch.nn.L1Loss()
        self.crit_gan = GANLoss(opt.gan_mode).cuda()

        self.optim_G = torch.optim.Adam(itertools.chain(self.model.netG_A.parameters(),
                                                        self.model.netG_B.parameters()),
                                        lr=opt.lr,
                                        betas=(opt.beta1, 0.999))
        self.optim_D = torch.optim.Adam(itertools.chain(self.model.netD_A.parameters(),
                                                            self.model.netD_B.parameters()),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))  # default: 0.5
        self.optimizers = [self.optim_G, self.optim_D]

        self.schedulers = [get_scheduler(optimizer, self.args) for optimizer in self.optimizers]

    def update_G(self, inp):
        real_A, real_B, fake_A, fake_B, rec_A, rec_B = inp

        opt = self.args
        lambda_idt = opt.lambda_identity  # G:x->y; min|G(y)-y|
        if lambda_idt > 0:
            idt_A = self.model.netG_A(real_B)
            idt_B = self.model.netG_B(real_A)
            loss_itd_A = self.crit_idt(idt_A, real_B) * lambda_idt * opt.lambda_B
            loss_itd_B = self.crit_idt(idt_B, real_A) * lambda_idt * opt.lambda_A
        else:
            loss_itd_A = 0
            loss_itd_B = 0

        loss_G_A = self.crit_gan(self.model.netD_A(fake_B), True)  # G想骗过D，所以说True
        loss_G_B = self.crit_gan(self.model.netD_B(fake_A), True)

        loss_cycle_A = self.crit_cycle(rec_A, real_A) * opt.lambda_A
        loss_cycle_B = self.crit_cycle(rec_B, real_B) * opt.lambda_B

        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_itd_A + loss_itd_B
        loss_G.backward()
        return loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B

    def update_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        pred_fake = netD(fake)
        loss_real = self.crit_gan(pred_real, True)
        loss_fake = self.crit_gan(pred_fake, False)
        return loss_fake, loss_real

    def update_D(self, inp):
        real_A, real_B, fake_A, fake_B, rec_A, rec_B = inp
        fake_B = self.fake_B_pool.query(fake_B)
        loss_fake, loss_real = self.update_D_basic(self.model.netD_A, real_B, fake_B)
        loss_D_A = 0.5*(loss_fake + loss_real)
        loss_D_A.backward()

        fake_A = self.fake_A_pool.query(fake_A)
        loss_fake, loss_real = self.update_D_basic(self.model.netD_B, real_A, fake_A)
        loss_D_B = 0.5*(loss_fake + loss_real)
        loss_D_B.backward()
        return loss_D_A, loss_D_B

    def train_on_step(self, inp):
        inp = self.model(inp)
        self.realA = inp[0][:5]  # 5,c,h,w;  for tensorboard
        self.realB = inp[1][:5]
        self.fakeA = inp[2][:5]
        self.fakeB = inp[3][:5]
        self.recA = inp[4][:5]
        self.recB = inp[5][:5]

        self.set_requires_grad([self.model.netD_A, self.model.netD_B], False)
        self.optim_G.zero_grad()
        loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B = self.update_G(inp)
        self.optim_G.step()

        self.set_requires_grad([self.model.netD_A, self.model.netD_B], True)
        self.optim_D.zero_grad()
        loss_D_A, loss_D_B = self.update_D(inp)
        self.optim_D.step()

        return loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, loss_D_A, loss_D_B

    def train_on_epoches(self, epoch):
        loss_g_a_meter = AvgMeter()
        loss_g_b_meter = AvgMeter()
        loss_cyc_a_meter = AvgMeter()
        loss_cyc_b_meter = AvgMeter()
        loss_d_a = AvgMeter()
        loss_d_b = AvgMeter()
        loss_meters = [loss_g_a_meter, loss_g_b_meter, loss_cyc_a_meter, loss_cyc_b_meter, loss_d_a, loss_d_b]
        loss_names = ['G_A', 'G_B', 'Cyc_A', 'Cyc_B', 'D_A', 'D_B']

        if self.rank == 0:
            progress_bar = tqdm(self.loader, desc='Epoch train')
        else:
            progress_bar = self.loader
        for iter_idx, sample in enumerate(progress_bar):
            losses_set = self.train_on_step(sample)
            for loss, meter in zip(losses_set, loss_meters):
                dist.all_reduce(loss)
                loss = loss / self.args.gpus_num
                meter.update(loss)

            cur_lr = self.optim_G.param_groups[0]['lr']
            step = iter_idx + 1 + epoch * self.each_epoch_iters
            if self.rank == 0:
                str_content = f'epoch: {epoch:d}; lr:{cur_lr:.6f};'
                for meter, name in zip(loss_meters, loss_names):
                    str_content += f' {name}: {meter.avg:.5f};'
                progress_bar.set_postfix(
                    logger=str_content)


                if (iter_idx+1) % 200 == 0:  # tensorboard
                    # print('tensorboard logging.')
                    realA = make_grid(self.realA, nrow=5, padding=2, normalize=True, range=(-1,1))
                    realB = make_grid(self.realB, nrow=5, padding=2, normalize=True, range=(-1,1))
                    fakeA = make_grid(self.fakeA, nrow=5, padding=2, normalize=True, range=(-1,1))
                    fakeB = make_grid(self.fakeB, nrow=5, padding=2, normalize=True, range=(-1,1))
                    recA = make_grid(self.recA, nrow=5, padding=2, normalize=True, range=(-1,1))
                    recB = make_grid(self.recB, nrow=5, padding=2, normalize=True, range=(-1,1))
                    self.td.add_image('realA', realA, step)
                    self.td.add_image('fakeA', fakeA, step)
                    self.td.add_image('realB', realB, step)
                    self.td.add_image('fakeB', fakeB, step)
                    self.td.add_image('recA', recA, step)
                    self.td.add_image('recB', recB, step)
                    for name, meter in zip(loss_names, loss_meters):
                        self.td.add_scalar(name, meter.avg, step)
                    self.td.flush()

        if self.rank == 0:
            progress_bar.close()

    def train(self):
        args = self.args
        for epoch in range(args.total_epoch):
            self.sample.set_epoch(epoch)  # shuffle
            self.train_on_epoches(epoch)

            self.update_learning_rate()
            # save network
            if self.rank == 0:
                if epoch > (args.total_epoch // 2-1):
                    # TODO: correct this problem.
                    # 目前这里存储有问题，用的不是nn.module，没有model.module
                    torch.save(self.model.module, f'{self.args.saved_dir}/{epoch}.pt')
                print('finish {}-th epoch.'.format(epoch))


    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


def main_worker(rank, args):
    args.rank = rank
    # init model and dataloader
    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.gpus_num,
                            rank=args.rank)
    torch.cuda.set_device(rank)

    model = CycleGANModel(args)
    model.train()
    model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.netG_A = torch.nn.parallel.DistributedDataParallel(model.netG_A, device_ids=[rank])
    model.netG_B = torch.nn.parallel.DistributedDataParallel(model.netG_B, device_ids=[rank])
    model.netD_A = torch.nn.parallel.DistributedDataParallel(model.netD_A, device_ids=[rank])
    model.netD_B = torch.nn.parallel.DistributedDataParallel(model.netD_B, device_ids=[rank])

    dataset = UnpairDataset(args)
    sample = torch.utils.data.distributed.DistributedSampler(dataset)
    train_loader = DataLoader(dataset, args.batchsize, num_workers=args.worker, sampler=sample)
    trainer = Train(model, train_loader, sample, args)
    trainer.train()



if __name__ == '__main__':

    args = cfgs
    port_id = 10000 + np.random.randint(0, 5000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port_id)
    args.gpus_num = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=args.gpus_num, args=(args,))