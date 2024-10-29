# On the model of a work by Yamil Vindas,
# by Mathilde Dupouy, June 2024

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import floor
from torchsummary import summary

class ConvolutionalEncoder(nn.Module):
    """
    A convolutional encoder made of 4 convolutional layers.
    """
    def __init__(self, input_channels = 3, output_channels = 4, k_size = 3):
        """
        Arguments:
        ----------
        input_channels: int
        Number of channels of the input (3 for a RGB image)

        output_channels: int
        Number of channels of the output, i.e. the latent space

        k_size: int
        Convolutional kernels size
        """
        super(ConvolutionalEncoder, self).__init__()
        self.k_size = k_size

        # Activation functions
        self.activation_fn = nn.LeakyReLU()

        # Pooling layer to reduce dimension
        self.poolingLayer = nn.MaxPool2d(kernel_size=2, stride=2)

        # First pattern
        in_channels, out_channels = input_channels, 8
        self.e_conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size, stride=1, padding=1, dilation=1)
        self.e_batchNorm_1 = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.99)

        # Second pattern
        in_channels, out_channels = out_channels, 16
        self.e_conv_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size, stride=1, padding=1)
        self.e_batchNorm_2 = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.99)

        # Third pattern
        in_channels, out_channels = out_channels, 16
        self.e_conv_3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size, stride=1, padding=1)
        self.e_batchNorm_3 = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.99)

        # Fourth pattern
        in_channels, out_channels = out_channels, output_channels
        self.e_conv_4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size, stride=1, padding=1)
        self.e_batchNorm_4 = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.99)

    def forward(self, input):
        # First pattern
        # print("Input shape: ", input.shape)
        x = self.activation_fn(self.e_batchNorm_1(self.e_conv_1(input)))
        x = self.poolingLayer(x)
        # print("Data shape after first pattern encoder: ", x.shape)

        # Second pattern
        x = self.activation_fn(self.e_batchNorm_2(self.e_conv_2(x)))
        x = self.poolingLayer(x)
        # print("Data shape after second pattern encoder: ", x.shape)

        # Third pattern
        x = self.activation_fn(self.e_batchNorm_3(self.e_conv_3(x)))
        x = self.poolingLayer(x)
        # print("Data shape after third pattern encoder: ", x.shape)

        # Fourth pattern
        x = self.activation_fn(self.e_batchNorm_4(self.e_conv_4(x)))
        output = self.poolingLayer(x)
        # print("Data shape after fourth pattern encoder: ", x.shape)

        return output


class ConvolutionalDecoder(nn.Module):
    """
    A convolutional decoder made of 4 transpose convolutional layers.
    """
    def __init__(self, input_channels = 4, output_channels = 3, k_size = 3):
        """
        Arguments:
        ----------
        input_channels: int
        Number of channels of the input (generally the latent space)

        output_channels: int
        Number of channels of the output (3 for a RGB image)

        k_size: int
        Convolutional kernels size
        """
        super(ConvolutionalDecoder, self).__init__()
        self.k_size = k_size

        # Activation functions
        self.activation_fn = nn.LeakyReLU()
        self.last_activation = nn.Sigmoid()

        # First pattern
        in_channels, out_channels = input_channels, 16
        self.d_conv_4 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size, stride=2, padding=1, output_padding=1)
        self.d_batchNorm_4 = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.99)

        # Second pattern
        in_channels, out_channels = out_channels, 16
        self.d_conv_3 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size, stride=2, padding=1, output_padding=1)
        self.d_batchNorm_3 = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.99)


        # Third pattern
        in_channels, out_channels = out_channels, 8
        self.d_conv_2 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size, stride=2, padding=1, output_padding=1)
        self.d_batchNorm_2 = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.99)

        # Fourth pattern
        in_channels, out_channels = out_channels, output_channels
        self.d_conv_1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size, stride=2, padding=1, output_padding=1)
        self.d_batchNorm_1 = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.99)


    def forward(self, input):
        # First pattern
        x = self.activation_fn(self.d_batchNorm_4(self.d_conv_4(input)))
        # print("Data shape after second pattern decoder: ", x.shape)

        # Second pattern
        x = self.activation_fn(self.d_batchNorm_3(self.d_conv_3(x)))
        # print("Data shape after second pattern decoder: ", x.shape)

        # Third pattern
        x = self.activation_fn(self.d_batchNorm_2(self.d_conv_2(x)))
        # print("Data shape after third pattern decoder: ", x.shape)

        # Fourth pattern
        output = self.last_activation(self.d_batchNorm_1(self.d_conv_1(x)))
        # print("Data shape after fourth pattern decoder: ", x.shape)

        return output

class ConvolutionalAE(nn.Module):
    """
    A convolutional auto encoder made of 4 layers of a convolutional encoder and 4 layers of a convolutional decoder.
    If the input height and width is not a multiple of 2⁴, the output is interpolated to the input size. We suggest to crop
    to an adequate size before to avoid side effect of interpolation.

    Attributes
    ----------
    encoder: ConvolutionalEncoder
    instantiation of the nn.Module class of a convolutional encoder

    decoder: ConvolutionalDecoder
    instantiation of the nn.Module class of a convolutional decoder

    latent_representation: torch.Tensor
    compressed representation (output of the bottleneck) on the cpu of the last input
    """
    def __init__(self, input_channels = 3, latent_channels = 4, k_size = 3):
        super(ConvolutionalAE, self).__init__()
        self.k_size = k_size

        self.encoder = ConvolutionalEncoder(input_channels=input_channels, output_channels=latent_channels, k_size=self.k_size)
        self.decoder = ConvolutionalDecoder(input_channels=latent_channels, output_channels=input_channels, k_size=self.k_size)

        self.latent_representation = None

    def forward(self, input):
        latent = self.encoder(input)
        self.latent_representation = latent.detach().to('cpu')
        output = self.decoder(latent)

        # Last interpolation to retrieve initial size (not necessary if initial shape is a multiple of  2⁴)
        output = F.interpolate(output, size=(input.shape[-2], input.shape[-1]))

        return (output,)


###############################################################################
###############################################################################

if __name__=="__main__":
    from torchsummary import summary

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Creating dummy data
    # nb_channels, h_in, w_in = 3, 214, 100
    nb_channels, h_in, w_in = 3, 192, 384
    # nb_channels, h_in, w_in = 3, 16, 16
    data = torch.randn((1, nb_channels, h_in, w_in)).to(device)

    # Creating the model
    model = ConvolutionalAE(input_channels=nb_channels, latent_channels=4, k_size = 7).to(device)

    # Summary of the model
    summary(model, (nb_channels, h_in, w_in))

    # # Parameters of the network
    # for name, param in model.named_parameters():
    #     print(name)

    # Evaluating the model
    output = model(data)
    latent_repr = model.latent_representation
    original_dim = nb_channels*h_in*w_in
    reduced_dim = latent_repr.shape[1]*latent_repr.shape[2]*latent_repr.shape[3] # element 0 ofcompressedRepr.shape is the number of samples in the batch

    print(f"Original sample dimension: {data.shape}")
    print(f"Reduced sample dimension: {latent_repr.shape}")
    print(f"Output sample dimension: {output[0].shape}")
    print(f"Compression rate of {original_dim/reduced_dim}")