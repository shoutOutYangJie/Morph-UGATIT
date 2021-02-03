import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.s_ugatit_plus import S_UGATITPlus
from configs.cfgs_s_ugatit_plus import cfgs
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets.unpair_dataset import UnpairDataset
import itertools
from utils.utils import AvgMeter, init_path, get_scheduler
from torch.utils import tensorboard
import os
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from models.image_pool import ImagePool
from models.loss import GANLoss, CAMLoss
import numpy as np

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
        self.crit_identity_preserving = torch.nn.L1Loss()
        self.crit_gan = GANLoss(opt.gan_mode).cuda()
        self.cam_loss = CAMLoss()
        self.optim_G = torch.optim.Adam(self.model.G.parameters(),
                                        lr=opt.lr,
                                        betas=(opt.beta1, 0.999))
        self.optim_D = torch.optim.Adam(itertools.chain(self.model.D_A.parameters(),
                                                        self.model.D_B.parameters()),
                                        lr=opt.lr, betas=(opt.beta1, 0.999))  # default: 0.5
        self.optimizers = [self.optim_G, self.optim_D]

        self.schedulers = [get_scheduler(optimizer, self.args) for optimizer in self.optimizers]


    def update_G(self, inp):
        real_A, real_B, fake_A, fake_B, rec_A, rec_B, cam_ab, cam_ba, real_e_ab, real_e_ba = inp
        opt = self.args
        lambda_idt = opt.lambda_identity  # G:x->y; min|G(y)-y|
        if lambda_idt > 0:  # identity loss
            b, c, h, w = real_A.shape
            z_bb = torch.randn(b, opt.z_dim * 2) * 0.2
            z_bb[:, :opt.z_dim] += 1  # 前 z1
            z_aa = torch.randn(b, opt.z_dim * 2) * 0.2
            z_aa[:, opt.z_dim:] += 1  # 后 z2
            if torch.cuda.is_available():
                z_aa = z_aa.cuda()
                z_bb = z_bb.cuda()
            # TODO: idt_A和idt_B也用D判别
            # TODO: recA, recB, idtA, idtB用人脸识别约束
            idt_A, cam_bb, _ = self.model.G(real_B, z_bb)  # b->B  for 渐变插值
            idt_B, cam_aa, _ = self.model.G(real_A, z_aa)  # A->A
            loss_itd_A = self.crit_idt(idt_A, real_B) * lambda_idt
            loss_itd_B = self.crit_idt(idt_B, real_A) * lambda_idt
        else:
            loss_itd_A = 0
            loss_itd_B = 0
        # D_A 用来检测B域的真假
        l_fake_B_logits, l_fake_B_cam_logits, \
        g_fake_B_logits, g_fake_B_cam_logits = self.model.D_A(fake_B)  # local and global

        l_fake_A_logits, l_fake_A_cam_logits, \
        g_fake_A_logits, g_fake_A_cam_logits = self.model.D_B(fake_A)
        # gan loss
        loss_G_A = self.crit_gan(l_fake_B_logits, True) + self.crit_gan(l_fake_B_cam_logits, True) + \
            self.crit_gan(g_fake_B_logits, True) + self.crit_gan(g_fake_B_cam_logits, True)
        loss_G_B = self.crit_gan(l_fake_A_logits, True) + self.crit_gan(l_fake_A_cam_logits, True) + \
            self.crit_gan(g_fake_A_logits, True) + self.crit_gan(g_fake_A_cam_logits, True)
        # cycle loss
        loss_cycle_A = self.crit_cycle(rec_A, real_A) * opt.lambda_cycle
        loss_cycle_B = self.crit_cycle(rec_B, real_B) * opt.lambda_cycle
        # cam loss
        if lambda_idt > 0:
            cam_loss_A = self.cam_loss(cam_ab, cam_bb) * opt.lambda_cam
            cam_loss_B = self.cam_loss(cam_ba, cam_aa) * opt.lambda_cam
        else:
            cam_loss_A = 0
            cam_loss_B = 0
        # identity preserving loss
        # 两个域之间相似性约束；对人脸->动漫估计不能这样做;
        # TODO: 对于差异较大的域transfer，也许可以尝试content loss
        fake_e_bb = self.model.G(fake_B, None, True)
        fake_e_aa = self.model.G(fake_A, None, True)
        loss_identity_A = self.crit_identity_preserving(fake_e_bb, real_e_ab) * opt.lambda_similarity
        loss_identity_B = self.crit_identity_preserving(fake_e_aa, real_e_ba) * opt.lambda_similarity


        loss_A = loss_G_A + loss_cycle_A + cam_loss_A + loss_itd_A + loss_identity_A # G_A的loss
        loss_B = loss_G_B + loss_cycle_B + cam_loss_B + loss_itd_B + loss_identity_B # G_B的loss

        loss_G = loss_A + loss_B
        loss_G = loss_G * 0.5  # 只有一个G
        loss_G.backward()
        return loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, cam_loss_A, cam_loss_B, loss_identity_A, loss_identity_B

    def update_D_basic(self, netD, real, fake):
        l_real_logits, l_real_cam_logits, \
        g_real_logits, g_real_cam_logits = netD(real)

        l_fake_logits, l_fake_cam_logits, \
        g_fake_logits, g_fake_cam_logits = netD(fake)

        loss_real = self.crit_gan(l_real_logits, True) + self.crit_gan(g_real_logits, True)
        loss_real_cam = self.crit_gan(l_real_cam_logits, True) + self.crit_gan(g_real_cam_logits, True)
        loss_fake = self.crit_gan(l_fake_logits, False) + self.crit_gan(g_fake_logits, False)
        loss_fake_cam = self.crit_gan(l_fake_cam_logits, False) + self.crit_gan(g_fake_cam_logits, False)
        return loss_fake, loss_real, loss_fake_cam, loss_real_cam

    def update_D(self, inp):
        real_A, real_B, fake_A, fake_B, rec_A, rec_B, _, _, _, _ = inp
        fake_B = self.fake_B_pool.query(fake_B)
        loss_fake, loss_real, loss_fake_cam, loss_real_cam \
            = self.update_D_basic(self.model.D_A, real_B, fake_B)
        loss_D_A = loss_fake + loss_real + loss_fake_cam + loss_real_cam
        loss_D_A.backward()

        fake_A = self.fake_A_pool.query(fake_A)
        loss_fake, loss_real, loss_fake_cam, loss_real_cam \
            = self.update_D_basic(self.model.D_B, real_A, fake_A)
        loss_D_B = loss_fake + loss_real + loss_fake_cam + loss_real_cam
        loss_D_B.backward()
        return loss_D_A, loss_D_B


    def train_on_step(self, inp):
        inp = self.model(inp)
        self.realA = inp[0]  # b,c,h,w;  for tensorboard
        self.realB = inp[1]
        self.fakeA = inp[2]
        self.fakeB = inp[3]
        # 先更新G
        self.set_requires_grad([self.model.D_A, self.model.D_B], False)
        self.optim_G.zero_grad()
        loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, loss_cam_A, loss_cam_B, loss_idt_A, loss_idt_B = self.update_G(inp)
        self.optim_G.step()

        self.set_requires_grad([self.model.D_A, self.model.D_B], True)
        self.optim_D.zero_grad()
        loss_D_A, loss_D_B = self.update_D(inp)
        self.optim_D.step()

        return loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, \
               loss_D_A, loss_D_B, loss_cam_A, loss_cam_B, loss_idt_A, loss_idt_B

    def train_on_epoch(self, epoch):
        loss_meters = []
        loss_names = ['G_A', 'G_B', 'Cyc_A', 'Cyc_B', 'D_A', 'D_B', 'CAM_A', 'CAM_B', 'IDT_A', 'IDT_B']
        for _ in range(len(loss_names)):
            loss_meters.append(AvgMeter())
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
                # for meter, name in zip(loss_meters, loss_names):
                #     str_content += f' {name}: {meter.avg:.5f};'
                progress_bar.set_postfix(
                    logger=str_content)
                if (step % 200) == 0:  # tensorboard
                    realA = make_grid(self.realA, padding=2, normalize=True, range=(-1, 1))
                    realB = make_grid(self.realB, padding=2, normalize=True, range=(-1, 1))
                    fakeA = make_grid(self.fakeA, padding=2, normalize=True, range=(-1, 1))
                    fakeB = make_grid(self.fakeB, padding=2, normalize=True, range=(-1, 1))
                    self.td.add_image('realA', realA, step)
                    self.td.add_image('fakeA', fakeA, step)
                    self.td.add_image('realB', realB, step)
                    self.td.add_image('fakeB', fakeB, step)
                    for name, meter in zip(loss_names, loss_meters):
                        self.td.add_scalar(name, meter.avg, step)
                    self.td.flush()
        if self.rank == 0:
            progress_bar.close()

    def train(self):
        args = self.args
        for epoch in range(args.total_epoch):
            self.sample.set_epoch(epoch)  # shuffle
            self.train_on_epoch(epoch)

            self.update_learning_rate()
            # save network
            if self.rank == 0:
                if epoch > (args.total_epoch // 2 - 1):
                    torch.save(self.model.state_dict(), f'{self.args.saved_dir}/{epoch}.pt')
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

    model = S_UGATITPlus(args)
    model.train()
    model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.G = torch.nn.parallel.DistributedDataParallel(model.G, device_ids=[rank])
    model.D_A = torch.nn.parallel.DistributedDataParallel(model.D_A, device_ids=[rank])
    model.D_B = torch.nn.parallel.DistributedDataParallel(model.D_B, device_ids=[rank])

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