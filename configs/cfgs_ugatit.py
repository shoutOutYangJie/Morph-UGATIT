from easydict import EasyDict

cfgs = {
    # model
    'inc': 3,
    'outc': 3,
    'ngf': 64,
    'ndf': 64,
    'use_dropout': False,
    'n_blocks': 4,
    'd_layers': 6,
    'training': True,
    # dataset
    # {adult, child}
    'dirA': '/share/yangjie08/datasets/age_adults',
    'dirB': '/share/yangjie08/datasets/generated_babies-stylegan2',
    # {real face, anime}
    'anime': False,  # to ensure dataset alignment.
    # 'dirA': '/share/yangjie08/datasets/selfie2anime/trainA',
    # 'dirB': '/share/yangjie08/datasets/selfie2anime/trainB',

    'direction': 'AtoB',
    'load_size': 256,
    'batchsize': 1,
    'worker': 5,
    # training
    'total_epoch': 100,  # 100->200
    'tensorboard': '/share/yangjie08/cycle_gan_log',
    'resume': '',   # resume training.
    'start_epoch': 0,  # if resume, please set start epoch.
    # 'saved_dir': '/home/yangjie08/My-CycleGAN/ckpt_adult_child',
    'saved_dir': '/home/yangjie08/My-CycleGAN/ckpt_sel_anime_ugatit',
    'pool_size': 10,
    'gan_mode': 'lsgan',
    'lr': 5e-4,
    'beta1': 0.5,
    'lr_decay_epoch': 50,
    'lr_policy': 'linear',
    'lambda_identity': 10,
    'lambda_cycle': 10,
    'lambda_cam': 1000,
}
cfgs = EasyDict(cfgs)


test_cfgs = EasyDict({
    # model
    'inc': 3,
    'outc': 3,
    'ngf': 64,
    'ndf': 64,
    'use_dropout': False,
    'n_blocks': 4,
    'd_layers': 6,
    'training': False,
    # 'saved_dir': '/share/yangjie08/result_ugatit',
})