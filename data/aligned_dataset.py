import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import cv2


class AlignedDataset(BaseDataset):
    """ A dataset calss for pairing images.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of A, B.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        """ Initialize the dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
        """
        self.opt = opt
        self.root = self.opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase)
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_A))
        self.C_paths = sorted(make_dataset(self.dir_A))
        self.D_paths = sorted(make_dataset(self.dir_A))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.C_size = len(self.C_paths)
        self.D_size = len(self.D_paths)

    def __getitem__(self, index):
        """ Return a data point and its metadata information.
        Params:
            index -- a random integer for data indexing
        Returns a dictionary that contains A, B, C, D and their paths.
        B (tensor) -- an image in the input domain.
        B_path (str) -- image paths
        """
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        C_path = self.C_paths[index]
        D_path = self.D_paths[index]
        BLUE = [255,255,255]

        B_img = cv2.imread(A_path)
        if self.B_size >= self.opt.datasetSize:
            b = B_img[:,:,0]
            g = B_img[:,:,1]
            r = B_img [:,:,2]
            B_img = np.dstack((r,g,b))
            h, w, channel = B_img.shape
            # Random uniform cropping
            w_offset = int(np.random.uniform(0, max(0, w - self.opt.fineSize - 1)))
            h_offset = int(np.random.uniform(0, max(0, h - self.opt.fineSize - 1)))

            B = B_img[h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            # Apply image interpolation
            A = cv2.resize(B, (B.shape[1]//2, B.shape[0]//2), interpolation = cv2.INTER_AREA)
            C = cv2.resize(B, (B.shape[1]//4, B.shape[0]//4), interpolation = cv2.INTER_AREA)
            D = cv2.resize(B, (B.shape[1]//8, B.shape[0]//8), interpolation = cv2.INTER_AREA)
            # Apply transform (image to tensor)
            A = transforms.ToTensor()(A)
            B = transforms.ToTensor()(B)
            C = transforms.ToTensor()(C)
            D = transforms.ToTensor()(D)
            # Normalization
            A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
            B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
            C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C)
            D = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(D)
            # Augmentation
            if (not self.opt.no_flip) and random.random() < 0.5:
                #for 2x:
                # idx_1 = [i for i in range(A.size(2) - 1, -1, -1)]
                #for 4x:
                idx_1 = [i for i in range(C.size(2) - 1, -1, -1)]
                #for 8x:
                # idx_1 = [i for i in range(D.size(2) - 1, -1, -1)]
                idx_2 = [i for i in range(B.size(2) - 1, -1, -1)]
                idx_1 = torch.LongTensor(idx_1)
                idx_2 = torch.LongTensor(idx_2)
                #for 2x:
                # A = A.index_select(2, idx_1)
                #for 4x:
                C = C.index_select(2, idx_1)
                #for 8x:
                # D = D.index_select(2, idx_1)
                B = B.index_select(2, idx_2)

            if self.opt.which_direction == 'BtoA':
                input_nc = self.opt.output_nc
                output_nc = self.opt.input_nc
            else:
                input_nc = self.opt.input_nc
                output_nc = self.opt.output_nc

            return {'A': A, 'B': B, 'C': C, 'D': D,
                    'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path, 'D_paths': D_path}
        else:

            BLUE = [255,255,255]
            b = B_img[:,:,0]
            g = B_img[:,:,1]
            r = B_img [:,:,2]
            B_img = np.dstack((r,g,b))
            # B_img = cv2.copyMakeBorder(B_img,100,150,100,150,cv2.BORDER_CONSTANT,value=BLUE)
            h, w, channel = B_img.shape
            left = (w - self.opt.fineSize)//2
            top = (h - self.opt.fineSize)//2
            right = (w + self.opt.fineSize)//2
            bottom = (h + self.opt.fineSize)//2
            B = B_img[top:bottom, left:right]

            A = cv2.resize(B, (B.shape[1]//2, B.shape[0]//2), interpolation = cv2.INTER_AREA)
            C = cv2.resize(B, (B.shape[1]//4, B.shape[0]//4), interpolation = cv2.INTER_AREA)
            D = cv2.resize(B, (B.shape[1]//8, B.shape[0]//8), interpolation = cv2.INTER_AREA)

            A = transforms.ToTensor()(A)
            B = transforms.ToTensor()(B)
            C = transforms.ToTensor()(C)
            D = transforms.ToTensor()(D)

            A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
            B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
            C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C)
            D = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(D)

            if self.opt.which_direction == 'AtoB':
                input_nc = self.opt.output_nc
                output_nc = self.opt.input_nc
            else:
                input_nc = self.opt.input_nc
                output_nc = self.opt.output_nc


            return {'A': A, 'B':B, 'C': C, 'D':D,
                'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path, 'D_paths': D_path}

    def __len__(self):
        return max(self.A_size, self.B_size, self.C_size, self.D_size)

    def name(self):
        return 'AlignedDataset'
