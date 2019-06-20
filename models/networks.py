import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import cv2
import math
import numpy as np
from skimage.measure import compare_ssim
from . import networks

###############################################################################
# Helper Functions
###############################################################################

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'upsample_2x':
        netG = Upsampling_2x(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'upsample_4x':
        netG = Upsampling_4x(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'upsample_8x':
        netG = Upsampling_8x(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)

# Bilinear filter
def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


# Just for visualization in visdom
def up_loss (target, upscale_factor, batchSize):

    neighbors, avg = [], []
    target = target.data.cpu().numpy()
    target = target[0,0,:,:]
    import numpy as np
    b = cv2.resize(target, (target.shape[0]*upscale_factor, target.shape[1]*upscale_factor), interpolation = cv2.INTER_CUBIC)
    B = torch.FloatTensor(b)
    B = Variable(B)

    y = B[None, None, :, :]
    real_HR = Variable(y.data, requires_grad=True)
    for i in range (0, batchSize - 1):
        return real_HR.cuda()

##############################################################################
# Classes
##############################################################################
# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.sum(error)
        return loss

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

#### VGG19 from torchvision
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return h_relu1, h_relu4

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class Upsampling_2x(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect'):
            super(Upsampling_2x, self).__init__()
            self.input_nc = input_nc
            self.output_nc = output_nc
            self.ngf = ngf
            if type(norm_layer) == functools.partial:
                use_bias = norm_layer.func == nn.InstanceNorm2d
            else:
                use_bias = norm_layer == nn.InstanceNorm2d

            self.conv_block = ResnetBlock(ngf, padding_type, norm_layer, use_dropout, use_bias)
            self.in_layer = nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)
            self.down = nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1, bias=use_bias)
            self.up = nn.ConvTranspose2d(ngf, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
            self.intermediate = nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias)
            self.out = nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=use_bias)
            self.vgg = Vgg19()
            self.dropout = nn.Dropout(0.3)
            self.padding = nn.ReflectionPad2d(3)
            self.leakyrelu = nn.LeakyReLU(0.2, True)
            # self.relu = nn.ReLU(True)
            self.tanh = nn.Tanh()

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                if isinstance(m, nn.ConvTranspose2d):
                    c1, c2, h, w = m.weight.data.size()
                    weight = get_upsample_filter(h)
                    m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, input):

        h_relu1, h_relu4 = self.vgg(input)
        in_layer = self.leakyrelu(self.padding(self.in_layer(input)))
        up_in1 = self.leakyrelu(self.up(in_layer))
        for i in range(2):
            in_layer = self.leakyrelu(self.down(in_layer))
        for i in range(3):
            in_layer = self.conv_block(in_layer)
        res_1 = self.dropout(in_layer)
        for i in range(3):
            res_1 = self.leakyrelu(self.up(res_1))
        up_1 = res_1 + up_in1
        up_1 = self.leakyrelu(self.padding(self.intermediate(up_1)))
        up_out = self.tanh(self.out(up_1))

        return up_out

class Upsampling_4x(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect'):
            super(Upsampling_4x, self).__init__()
            self.input_nc = input_nc
            self.output_nc = output_nc
            self.ngf = ngf
            if type(norm_layer) == functools.partial:
                use_bias = norm_layer.func == nn.InstanceNorm2d
            else:
                use_bias = norm_layer == nn.InstanceNorm2d

            self.conv_block = ResnetBlock(ngf, padding_type, norm_layer, use_dropout, use_bias)
            self.in_layer = nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)
            self.down = nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1, bias=use_bias)
            self.up = nn.ConvTranspose2d(ngf, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
            self.intermediate = nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias)
            self.out = nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=use_bias)
            self.vgg = Vgg19()
            self.dropout = nn.Dropout(0.3)
            self.padding = nn.ReflectionPad2d(3)
            self.leakyrelu = nn.LeakyReLU(0.2, True)
            # self.relu = nn.ReLU(True)
            self.tanh = nn.Tanh()

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                if isinstance(m, nn.ConvTranspose2d):
                    c1, c2, h, w = m.weight.data.size()
                    weight = get_upsample_filter(h)
                    m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, input):

        h_relu1, h_relu4 = self.vgg(input)
        in_layer = self.leakyrelu(self.padding(self.in_layer(input)))
        up_in1 = self.leakyrelu(self.up(in_layer))
        for i in range(2):
            in_layer = self.leakyrelu(self.down(in_layer))
        for i in range(3):
            in_layer = self.conv_block(in_layer)
        res_1 = self.dropout(in_layer)
        for i in range(3):
            res_1 = self.leakyrelu(self.up(res_1))
        up_1 = res_1 + up_in1
        down_2 = self.leakyrelu(self.intermediate(up_1))
        up_in2 = self.leakyrelu(self.up(up_1))
        for i in range(2):
            down_2 = self.leakyrelu(self.down(down_2))
        for i in range(3):
            down_2 = self.conv_block(down_2)
        res_2 = self.dropout(down_2)
        for i in range(3):
            res_2 = self.leakyrelu(self.up(res_2))
        up_2 = res_2 + up_in2
        up_2 = self.leakyrelu(self.padding(self.intermediate(up_2)))
        up_out = self.tanh(self.out(up_2))

        return up_out

class Upsampling_8x(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect'):
            super(Upsampling_8x, self).__init__()
            self.input_nc = input_nc
            self.output_nc = output_nc
            self.ngf = ngf
            if type(norm_layer) == functools.partial:
                use_bias = norm_layer.func == nn.InstanceNorm2d
            else:
                use_bias = norm_layer == nn.InstanceNorm2d

            self.conv_block = ResnetBlock(ngf, padding_type, norm_layer, use_dropout, use_bias)
            self.in_layer = nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)
            self.down = nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1, bias=use_bias)
            self.up = nn.ConvTranspose2d(ngf, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
            self.intermediate = nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias)
            self.out = nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=use_bias)
            self.vgg = Vgg19()
            self.dropout = nn.Dropout(0.3)
            self.padding = nn.ReflectionPad2d(3)
            self.leakyrelu = nn.LeakyReLU(0.2, True)
            # self.relu = nn.ReLU(True)
            self.tanh = nn.Tanh()

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                if isinstance(m, nn.ConvTranspose2d):
                    c1, c2, h, w = m.weight.data.size()
                    weight = get_upsample_filter(h)
                    m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, input):

        h_relu1, h_relu4 = self.vgg(input)
        in_layer = self.leakyrelu(self.padding(self.in_layer(input)))
        up_in1 = self.leakyrelu(self.up(in_layer))
        for i in range(2):
            in_layer = self.leakyrelu(self.down(in_layer))
        for i in range(3):
            in_layer = self.conv_block(in_layer)
        res_1 = self.dropout(in_layer)
        for i in range(3):
            res_1 = self.leakyrelu(self.up(res_1))
        up_1 = res_1 + up_in1
        down_2 = self.leakyrelu(self.intermediate(up_1))
        up_in2 = self.leakyrelu(self.up(up_1))
        for i in range(2):
            down_2 = self.leakyrelu(self.down(down_2))
        for i in range(3):
            down_2 = self.conv_block(down_2)
        res_2 = self.dropout(down_2)
        for i in range(3):
            res_2 = self.leakyrelu(self.up(res_2))
        up_2 = res_2 + up_in2
        down_3 = self.leakyrelu(self.intermediate(up_2))
        up_in3 = self.leakyrelu(self.up(up_2))
        for i in range (2):
            down_3 = self.leakyrelu(self.down(down_3))
        for i in range(3):
            down_3 = self.conv_block(down_3)
        res_3 = self.dropout(down_3)
        for i in range(3):
            res_3 = self.leakyrelu(self.up(res_3))
        up_3 = res_3 + up_in3
        up_3 = self.leakyrelu(self.padding(self.intermediate(up_3)))
        up_out = self.tanh(self.out(up_3))

        return up_out
