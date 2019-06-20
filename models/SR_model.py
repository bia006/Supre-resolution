import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable


class SRModel(BaseModel):
    def name(self):
        return 'SRModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the SR paper
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='upsample')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        if self.isTrain:

            # specify the training losses you want to print out. The program will call base_model.get_current_losses
            self.loss_names = ['G_L1', 'G_VGG']
            # specify the images you want to save/display. The program will call base_model.get_current_visuals
            self.visual_names = ['fake_B_LR', 'fake_B', 'real_B']
            # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
            self.model_names = ['G']
        else:
            self.visual_names = ['fake_B','real_B', 'real_D']
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            # self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1_C = networks.L1_Charbonnier_loss()
            self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_C = input['C' if AtoB else 'C'].to(self.device)
        self.real_D = input['D' if AtoB else 'D'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_input_test(self, input):
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        # for 2X:
        # self.fake_B = self.netG(self.real_A)
        # self.fake_B_LR = networks.up_loss(self.real_A, 2, 4)
        # # for 4x:
        self.fake_B = self.netG(self.real_C)
        self.fake_B_LR = networks.up_loss(self.real_C, 4, 4)
        # for 8x:
        # self.fake_B = self.netG(self.real_D)
        # self.fake_B_LR = networks.up_loss(self.real_D, 8, 4)

    def test(self):
        # for 2x:
        # self.fake_B = self.netG(self.real_A)
        # for 4x:
        self.fake_B = self.netG(self.real_C)
        # for 8x:
        # self.fake_B = self.netG(self.real_D)

    def backward_G(self):
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1_C(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * self.opt.lambda_feat
        self.loss_G = self.loss_G_VGG + self.loss_G_L1
        # feature matching loss
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def load(self, label):
        self.load_networks(label)
