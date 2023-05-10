import torch
import torch.nn as nn
from torch.nn.functional import l1_loss
from numpy import random
from torcheval import metrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# cycle loss, using l1_loss
def cycle_loss(real_a, cycle_a, real_b, cycle_b):
    return l1_loss(real_a, cycle_a) + l1_loss(real_b, cycle_b)

# using an image pool to buffer images, ideally, helps the generator stay ahead of the discriminator 
# This image pool implementation comes from the original CycleGAN paper's PyTorch implementation:
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master
class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


# This implementation of a resnet block was adapted from
# https://github.com/wangguanan/Pytorch-Image-Translation-GANs/
class ResidualBlock(nn.Module):
    '''Residual Block with Instance Normalization'''

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='reflect', device=device),
            nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False, device=device),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='reflect', device=device),
            nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False, device=device),
        )

    def forward(self, x):
        return self.model(x) + x

# We adapt this resnet generator from
# https://github.com/wangguanan/Pytorch-Image-Translation-GANs/
# though we experiment with different model parameters, such as kernel size, bias, 
# padding modes, and instance normalization
class Generator(nn.Module):
    '''Generator with Down sampling, Several ResBlocks and Up sampling.
       Down/Up Samplings are used for less computation.
    '''

    def __init__(self, conv_dim, layer_num):
        super(Generator, self).__init__()

        layers = []

        # input layer
        layers.append(nn.Conv2d(in_channels=3, out_channels=conv_dim, kernel_size=7, stride=1, padding=3, bias=True, padding_mode='reflect', device=device))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=False, track_running_stats=False, device=device))
        layers.append(nn.ReLU(inplace=True))

        # down sampling layers
        current_dims = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(current_dims, current_dims*2, kernel_size=3, stride=2, padding=1, bias=True, device=device))
            layers.append(nn.InstanceNorm2d(current_dims*2, affine=False, track_running_stats=False, device=device))
            layers.append(nn.ReLU(inplace=True))
            current_dims *= 2

        # Residual Layers
        for i in range(layer_num):
            layers.append(ResidualBlock(current_dims, current_dims))

        # up sampling layers
        for i in range(2):
            layers.append(nn.ConvTranspose2d(current_dims, current_dims//2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True, device=device))
            layers.append(nn.InstanceNorm2d(current_dims//2, affine=False, track_running_stats=False, device=device))
            layers.append(nn.ReLU(inplace=True))
            current_dims = current_dims//2

        # output layer
        layers.append(nn.Conv2d(current_dims, 3, kernel_size=7, stride=1, padding=3, bias=True, padding_mode='reflect', device=device))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, x,):
        return self.model(x)

# We adapt this discriminator from
# https://github.com/wangguanan/Pytorch-Image-Translation-GANs/
class Discriminator(nn.Module):
    '''Discriminator with PatchGAN'''

    def __init__(self, image_size, conv_dim, layer_num):
        super(Discriminator, self).__init__()

        layers = []

        # input layer
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1, device=device))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        current_dim = conv_dim

        # hidden layers
        for i in range(layer_num-1):
            layers.append(nn.Conv2d(current_dim, current_dim*2, kernel_size=4, stride=2, padding=1, device=device))
            layers.append(nn.InstanceNorm2d(current_dim*2, device=device))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_dim *= 2

        layers.append(nn.Conv2d(current_dim, current_dim*2, kernel_size=4, stride=1, padding=1, device=device))
        layers.append(nn.InstanceNorm2d(current_dim*2, device=device))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        current_dim *= 2

        self.model = nn.Sequential(*layers)

        # output layer
        self.conv_src = nn.Conv2d(current_dim, 1, kernel_size=4, stride=1, padding=1, bias=True, device=device)

    def forward(self, x):
        x = self.model(x)
        out_src = self.conv_src(x)
        return out_src
    
# We follow a similar CycleGAN class and training loop as
# https://github.com/Asthestarsfalll/Symbolic-Music-Genre-Transfer-with-CycleGAN-for-pytorch
class CycleGAN(nn.Module):

    def __init__(self, mode='train', lamb=10):
        super(CycleGAN, self).__init__()

        # Train for training, or A2B or B2A for evaluation
        assert mode in ["train", "A2B", "B2A"]
        self.mode = mode

        # Generator for domain A->B
        self.G_A2B = Generator(conv_dim=64, layer_num=6) # 6 resnet blocks, as in the CycleGAN paper
        
        # Generator for domain B->A
        self.G_B2A = Generator(conv_dim=64, layer_num=6) # 6 resnet blocks
        
        # Discriminators for domains A and B
        self.D_A = Discriminator(image_size=256, conv_dim=64, layer_num=3)
        self.D_B = Discriminator(image_size=256, conv_dim=64, layer_num=3)

        self.l2loss = nn.MSELoss(reduction="mean") # LSGAN

        self.criterionIdt = nn.L1Loss()  # We use L1 loss for cycle and identity losses

        self.lamb = lamb # Weighting of cycle loss

        # Image pools
        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)

    # Provides test accuracies for the discriminators on real images and ones generated
    # by our generators
    def test(self, real_A, real_B):
        with torch.no_grad():
            acc = metrics.functional.binary_accuracy
            fake_B = self.G_A2B(real_A)
            
            fake_A = self.G_B2A(real_B)
            
            DA_fake = self.D_A(fake_A)

            DB_fake = self.D_B(fake_B)

            DA_real = self.D_A(real_A)

            DB_real = self.D_B(real_B)

            fake_A_acc = acc(DA_fake.flatten(), torch.zeros(DA_fake.flatten().shape, device=device))
            
            fake_B_acc = acc(DB_fake.flatten(), torch.zeros(DB_fake.flatten().shape, device=device))

            real_A_acc = acc(DA_real.flatten(), torch.ones(DA_real.flatten().shape, device=device))

            real_B_acc = acc(DB_real.flatten(), torch.ones(DB_real.flatten().shape, device=device))

            return fake_A_acc, fake_B_acc, real_A_acc, real_B_acc


    def forward(self, real_A, real_B):
        fake_B = self.G_A2B(real_A)
        cycle_A = self.G_B2A(fake_B)

        fake_A = self.G_B2A(real_B)
        cycle_B = self.G_A2B(fake_A)

        if self.mode == 'train':
            DA_fake = self.D_A(fake_A)
            DB_fake = self.D_B(fake_B)

            # Cycle loss
            c_loss = self.lamb * cycle_loss(real_A, cycle_A, real_B, cycle_B)

            # Identity loss
            i_loss_a2b = l1_loss(real_A, fake_B) * self.lamb * .3
            i_loss_b2a = l1_loss(real_B, fake_A) * self.lamb * .3

            # Generator losses
            if device == 'cuda':
                g_A2B_loss = self.l2loss(DB_fake, torch.ones_like(DB_fake).cuda()) + c_loss + i_loss_a2b
                g_B2A_loss = self.l2loss(DA_fake, torch.ones_like(DA_fake).cuda()) + c_loss + i_loss_b2a
            else: 
                g_A2B_loss = self.l2loss(DB_fake, torch.ones_like(DB_fake).cuda()) + c_loss + i_loss_a2b
                g_B2A_loss = self.l2loss(DA_fake, torch.ones_like(DA_fake).cuda()) + c_loss + i_loss_b2a

            DA_real = self.D_A(real_A)
            DB_real = self.D_B(real_B)

            # buffer helps prevent oscillation in training
            fake_A = self.fake_A_pool.query(fake_A.clone().detach())
            fake_B = self.fake_B_pool.query(fake_B.clone().detach())

            DA_fake = self.D_A(fake_A)
            DB_fake = self.D_B(fake_B)

            # Discriminator loss
            if device == 'cuda':
                d_A_loss_real = self.l2loss(DA_real, torch.ones_like(DA_real).cuda())
                d_A_loss_fake = self.l2loss(DA_fake, torch.zeros_like(DA_fake).cuda())
                d_A_loss = (d_A_loss_real + d_A_loss_fake) / 2
                d_B_loss_real = self.l2loss(DB_real, torch.ones_like(DB_real).cuda())
                d_B_loss_fake = self.l2loss(DB_fake, torch.zeros_like(DB_fake).cuda())
                d_B_loss = (d_B_loss_real + d_B_loss_fake) / 2
            else:
                d_A_loss_real = self.l2loss(DA_real, torch.ones_like(DA_real))
                d_A_loss_fake = self.l2loss(DA_fake, torch.zeros_like(DA_fake))
                d_A_loss = (d_A_loss_real + d_A_loss_fake) / 2
                d_B_loss_real = self.l2loss(DB_real, torch.ones_like(DB_real))
                d_B_loss_fake = self.l2loss(DB_fake, torch.zeros_like(DB_fake))
                d_B_loss = (d_B_loss_real + d_B_loss_fake) / 2

            return (c_loss, g_A2B_loss, g_B2A_loss, d_A_loss, d_B_loss)

        elif self.mode == 'A2B':
                return fake_B, cycle_A
        elif self.mode == 'B2A':
                return fake_A, cycle_B