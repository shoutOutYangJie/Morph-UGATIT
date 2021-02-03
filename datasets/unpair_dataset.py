from torch.utils.data import Dataset
import os
from utils.data_aug import get_transform, get_transform_for_anime
from PIL import Image
import random



class UnpairDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dirA = args.dirA
        self.dirB = args.dirB
        self.A_path = sorted(os.listdir(self.dirA))
        self.B_path = sorted(os.listdir(self.dirB))
        self.A_size = len(self.A_path)
        self.B_size = len(self.B_path)
        print('A has {:d} images, at {}'.format(self.A_size, self.dirA))
        print('B has {:d} images, at {}'.format(self.B_size, self.dirB))

        AtoB = args.direction == 'AtoB'
        inc = args.inc if AtoB else args.outc
        outc = args.outc if AtoB else args.inc
        if args.anime:
            self.transform_A = get_transform_for_anime(args)
            self.transform_B = get_transform_for_anime(args)
        else:
            self.transform_A = get_transform(args, grayscale=(inc==1))
            self.transform_B = get_transform(args, grayscale=(outc==1))

    def __getitem__(self, idx):
        A_path = self.A_path[idx % self.A_size]
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_path[index_B]
        A_img = Image.open(os.path.join(self.dirA, A_path)).convert('RGB')
        B_img = Image.open(os.path.join(self.dirB, B_path)).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        return {'A':A, 'B': B}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

if __name__ == '__main__':
    import cv2
    from configs.cfgs_train import cfgs
    # cfgs.dirB = '/Users/yangjie08/dataset/youtube_vos/train/JPEGImages/0a2f2bd294'
    # cfgs.dirA = '/Users/yangjie08/dataset/youtube_vos/train/JPEGImages/0a2f2bd294'
    dataset = UnpairDataset(cfgs)
    for sample in dataset:
        A, B = sample['A'], sample['B']
        print(A.shape, B.shape)
        A = (A*0.5) + 0.5
        B = B*0.5 + 0.5
        A = A.permute(1, 2, 0).flip([2]).numpy()
        B = B.permute(1, 2, 0).flip([2]).numpy()
        cv2.imshow('A', A)
        cv2.imshow('B', B)
        cv2.waitKey(100)

