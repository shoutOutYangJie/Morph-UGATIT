from easydict import EasyDict

cfgs = {
    # model
    'inc': 3,
    'outc': 3,
    'ngf': 64,
    'ndf': 64,
    'z_dim': 32,
    'use_dropout': False,
    'n_blocks': 4,
    'd_layers': 6,
    'training': True,
    # dataset
    'anime': False,
    'dirA': '/share/yangjie08/datasets/age_adults',
    'dirB': '/share/yangjie08/datasets/generated_babies-stylegan2',
    'direction': 'AtoB',
    'load_size': 256,
    'batchsize': 1,
    'worker': 5,
    # training
    'total_epoch': 100,
    'tensorboard': '/share/yangjie08/s_ugatit_plus',
    'saved_dir': '/home/yangjie08/My-CycleGAN/ckpt_s_ugatit_plus',
    'pool_size': 10,
    'gan_mode': 'lsgan',
    'lr': 2e-4,
    'beta1': 0.5,
    'lr_decay_epoch': 50,
    'lr_policy': 'linear',
    'lambda_identity': 10,
    'lambda_cycle': 10,
    'lambda_cam': 1000,  # CAM loss is unnecessary.
    'lambda_similarity': 1.0,  # 或许可以更小
}
cfgs = EasyDict(cfgs)


test_cfgs = EasyDict({
    # model
    'inc': 3,
    'outc': 3,
    'ngf': 64,
    'ndf': 64,
    'z_dim': 32,
    'use_dropout': False,
    'n_blocks': 4,
    'd_layers': 6,
    'training': False,
})