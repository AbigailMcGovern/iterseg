import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F


# globals
decoder_instructions = {
            5: {
                'in' : 256 * 2, 
                'out': 128
            }, 
            6: {
                'in' : 128 * 2, 
                'out': 64
            }, 
            7: {
                'in' : 64 * 2, 
                'out': 32
            }, 
        }


# convolution module
class ConvModule(nn.Module):
    def __init__(
                 self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding0=1, 
                 padding1=1, 
                 final='relu'
                 ):
        """
        Convolution module with 2x batch normaisation --> convolution. 
        First activation function is ReLU. Second activation defaults to ReLU
        but options also include SoftMax and Sigmoid.

        Parameters
        ----------
        in_channels: int 
            Number of input channels for data
        out_channels: int
            Number of desired output channels
        kernel_size: int or tuple of int
            Size of convolution kernel
        stride: int or tuple of int
            Size of stride to use during convolution
        padding0: int or tuple of int
            Size with which to pad data for first convolution
        padding1: int or tuple of int
            Size with which to pad data for second convolution
        final: str
            Option for final activation function: 
            'relu', 'softmax', or 'sigmoid'
        """
        super(ConvModule, self).__init__()

        # Convolutions
        # ------------
        self.conv0 = nn.Conv3d(
                               in_channels, 
                               out_channels, 
                               kernel_size=kernel_size, 
                               stride=stride, 
                               padding=padding0
                               )
        self.conv1 = nn.Conv3d(
                               out_channels, 
                               out_channels, 
                               kernel_size=kernel_size, 
                               stride=stride, 
                               padding=padding1
                               )

        # Batch Normailsation
        # -------------------
        self.batch0 = nn.BatchNorm3d(out_channels)
        self.batch1 = nn.BatchNorm3d(out_channels)

        # Activation
        # ----------
        self.relu0 = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.sm = nn.Softmax()
        self.final = final


    def forward(self, x):
        # First convolution
        x = self.conv0(x)
        x = self.batch0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.batch1(x)
        if self.final == 'relu':
            x = self.relu1(x)
        elif self.final == 'softmax':
            x = self.sm(x)
        elif self.final == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.final == 'tanh':
            x = torch.tanh(x)
        return x


class ResConvModule(nn.Module):

    def __init__(
                 self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding0=1, 
                 padding1=1, 
                 final='relu'):
        """
        """
        super(ResConvModule, self).__init__()


# trial UNet
class UNet(nn.Module):

    def __init__(
                 self, 
                 in_channels=1, 
                 out_channels=5, 
                 down_factors=(1, 2, 2), 
                 up='convolution', 
                 downsample_1_at_bottom=True, 
                 chan_final_activations=None
                 ):
        '''
        Anisotropic U-net

        Parameters
        ----------
        in_channels: int
        out_channels: int
        down_factors: tuple of int
            Factors by which to downsample in encoder
        up: str
            'bilinear': use bilinear/nearest neighbour interpolations for up 
            sampling in decoder
            'convolution': use inverse convolutions with learnable parameters
        '''
        super(UNet, self).__init__()
        self.forked = isinstance(out_channels, tuple)
        if not self.forked:
            out_channels = (out_channels,)
        else:
            self.forks = len(out_channels)
        self.out_channels = out_channels
        # Max pooling 
        # -----------
        # encoder: downsample 4 times
        # Get the padding for the max pool
        #   Must be at most half of 
        p = [np.floor_divide(df, 2).astype(int) for df in down_factors]
        p = tuple(p)
        # max pool layers
        self.d0 = nn.MaxPool3d(
                               down_factors, 
                               stride=down_factors, 
                               padding=(0, 1, 1)
                               )
        self.d1 = nn.MaxPool3d(
                               down_factors, 
                               stride=down_factors, 
                               padding=p
                               )
        self.d2 = nn.MaxPool3d(
                               down_factors, 
                               stride=down_factors, 
                               padding=(0, 1, 1)
                               )
        # the input will be downsampled in previously un
        new_down = self.new_down_factors(down_factors, downsample_1_at_bottom)
        self.d3 = nn.MaxPool3d(
                               new_down, 
                               stride=new_down, 
                               padding=(0, 1, 1)
                               )

        # Convolutions
        # ------------
        # encoder colvolutions:
        self.c0 = ConvModule(in_channels, 32)
        self.c1 = ConvModule(32, 64)
        self.c2 = ConvModule(64, 128)
        self.c3 = ConvModule(128, 256)
        self.c4 = ConvModule(256, 256)

        # decoder convolutions
        for i, c in enumerate(out_channels):
            for key in decoder_instructions.keys():
                n, in_c, = key, decoder_instructions[key]['in']
                out_c = decoder_instructions[key]['out']
                cmd = f'self.c{n}_{i} = ConvModule({in_c}, {out_c})'
                exec(cmd)
            if chan_final_activations is not None:
                final = chan_final_activations[i]
            else:
                final = 'sigmoid'
            cmd = f'self.c8_{i} = ConvModule(32 * 2, {c}, final=\'{final}\')'
            exec(cmd)

        # Upsampling
        # ----------
        # Inverse convolutions
        if up == 'convolution':
            self.up0 = nn.ConvTranspose3d(
                                          256, 
                                          256, 
                                          kernel_size=new_down, 
                                          stride=new_down, 
                                          groups=256)
            self.up1 = nn.ConvTranspose3d(
                                          128, 
                                          128, 
                                          kernel_size=down_factors, 
                                          stride=down_factors, 
                                          groups=128
                                          )
            self.up2 = nn.ConvTranspose3d(
                                          64, 
                                          64, 
                                          kernel_size=down_factors, 
                                          stride=down_factors, 
                                          groups=64
                                          )
            self.up3 = nn.ConvTranspose3d(
                                          32, 
                                          32, 
                                          kernel_size=down_factors, 
                                          stride=down_factors, 
                                          groups=32
                                          )
        elif up == 'bilinear':
            self.up0 = lambda x: self.bilinear_interpolation(x, down_factors)
            self.up1 = lambda x: self.bilinear_interpolation(x, down_factors)
            self.up2 = lambda x: self.bilinear_interpolation(x, down_factors)
            self.up3 = lambda x: self.bilinear_interpolation(x, down_factors)
        else:
            raise ValueError('Valid options for up param are convolution and bilinear')


    def bilinear_interpolation(self, x, down_factors):
        '''
        THIS NEEDS DEBUGGING!!
        '''
        print(x.shape)
        x = torch.squeeze(x, 0)
        print(x.shape)
        x = F.interpolate(
                          x, 
                          mode='tconv', 
                          scale_factor=down_factors
                          )
        x = torch.unsqueeze(x, 0)
        return x


    def new_down_factors(self, down_factors, downsample_1_at_bottom):
        if downsample_1_at_bottom:
            # downsample in last pool layer if down factor has been 1
            # for an axis
            new_down = []
            for df in down_factors:
                if df == 1:
                    new_down.append(2)
                else:
                    new_down.append(df)
            new_down = tuple(new_down) # probs not necessary, I like it, sue me
        else:
            new_down = down_factors
        return new_down


    def forward(self, x):
        # Encoder
        # -------
        x, c0, c1, c2, c3 = self.encoder(x)
 
        # Decoder
        # -------
        if self.forked:
            x = self.forked_decoder(x, c0, c1, c2, c3)
        else:
            x = self.decoder(x, c0, c1, c2, c3)
        return x

    
    def encoder(self, x):
        # Encoder
        # -------
        c0 = self.c0(x)
        x = self.d0(c0)
        c1 = self.c1(x)
        x = self.d1(c1)
        c2 = self.c2(x)
        x = self.d2(c2)
        c3 = self.c3(x)
        x = self.d3(c3)
        x = self.c4(x)
        return x, c0, c1, c2, c3


    def forked_decoder(self, x, c0, c1, c2, c3):
        # Decoder
        # -------
        started = False
        for i in range(self.forks):
            x = x.clone()
            if not started:
                x0 = self.decoder(x, c0, c1, c2, c3, i=i)
                started = True
            else:
                x1 = self.decoder(x, c0, c1, c2, c3, i=i)
                x0 = torch.cat((x0, x1), dim=1)
        return x0


    def decoder(self, x, c0, c1, c2, c3, i=0):
        x = self.up0(x)
        # quick dumb hack for concatenation 
        x = x[:, :, :, :-1, :-1]
        x = torch.cat([x, c3], 1)
        if i == 0:
            x = self.c5_0(x)
            x = self.up1(x)
            x = x[:, :, :, :-1, :-1]
            x = torch.cat([x, c2], 1)
            x = self.c6_0(x)
            x = self.up2(x)
            x = x[:, :, :, :-1, :-1]
            x = torch.cat([x, c1], 1)
            x = self.c7_0(x)
            x = self.up3(x)
            x = x[:, :, :, 1:-1, 1:-1]
            x = torch.cat([x, c0], 1)
            x = self.c8_0(x)
        elif i == 1:
            x = self.c5_1(x)
            x = self.up1(x)
            x = x[:, :, :, :-1, :-1]
            x = torch.cat([x, c2], 1)
            x = self.c6_1(x)
            x = self.up2(x)
            x = x[:, :, :, :-1, :-1]
            x = torch.cat([x, c1], 1)
            x = self.c7_1(x)
            x = self.up3(x)
            x = x[:, :, :, 1:-1, 1:-1]
            x = torch.cat([x, c0], 1)
            x = self.c8_1(x)
            # so on and so forth? Couldn't make the below work
        #cmd = f'x = self.c8_{i}(x)'
        #exec(cmd)
        return x

    




class ForkedUNet(UNet):
    def __init__(self, in_channels=1, fork_channels=(8, 2)):
        super(ForkedUNet, self).__init__(in_channels=in_channels, out_channels=fork_channels)
        self.forks = len(fork_channels)


    def forward(self, x):
        # Encoder
        # -------
        x, c0, c1, c2, c3 = self.encoder(x)
        print('forked unet')
 
        # Decoder
        # -------
        started = False
        for i in range(self.forks):
            x = x.clone()
            if not started:
                x0 = self.decoder(x, c0, c1, c2, c3, i=i)
                print('out shape: ', x0.shape)
            else:
                x1 = self.decoder(x, c0, c1, c2, c3, i=i)
                print('out shape: ', x1.shape)
                x0 = torch.cat(x0, x1, dim=1)
        return x0



if __name__ == '__main__':
    ip = torch.randn(1, 1, 10, 256, 256)
    unet = UNet()
    o = unet(ip)
    # output has shape (1, 3, 10, 256, 256)

    # c0 -> torch.Size([1, 32, 10, 256, 256])
    # d0, c1 -> torch.Size([1, 64, 10, 129, 129])
    # d1, c2 -> torch.Size([1, 128, 10, 65, 65])
    # d2, c3 -> torch.Size([1, 256, 10, 33, 33])
    # d3, c4 -> torch.Size([1, 256, 10, 17, 17]
    # u0 -> torch.Size([1, 256, 10, 33, 33])
    # u1, c5 -> torch.Size([1, 128, 10, 65, 65])
    # u2, c6 -> torch.Size([1, 64, 10, 129, 129])
    # u3, c7 -> torch.Size([1, 32, 10, 256, 256])
    # c8 -> torch.Size([1, 3, 10, 256, 256])