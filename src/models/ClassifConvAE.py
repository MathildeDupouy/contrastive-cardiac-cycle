# On the model of a work by Yamil Vindas,
# by Mathilde Dupouy, June 2024

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import floor
from torchsummary import summary
from Models.ConvolutionalAE import ConvolutionalEncoder, ConvolutionalDecoder

class ClassifConvAE(nn.Module):
    """
    A convolutional auto encoder made of 4 layers of a convolutional encoder and 4 layers of a convolutional decoder
    + a classification head on the latent space, in the form of a fully connected layer.
    If the input height and width is not a multiple of 2⁴, the output is interpolated to the input size. We suggest to crop
    input image to an adequate size before training to avoid side effect of interpolation.

    Attributes
    ----------
    encoder: ConvolutionalEncoder
    instantiation of the nn.Module class of a convolutional encoder

    decoder: ConvolutionalDecoder
    instantiation of the nn.Module class of a convolutional decoder

    latent_representation: torch.Tensor
    compressed representation (output of the bottleneck) on the cpu of the last input
    """
    def __init__(self, input_shape, n_classes, input_channels = 3, latent_channels = 4):
        """
        Arguments:
        ----------
        input_shape: tuple, list or torch.Size
        The input shape of the data in the order (channel, height, width)

        n_classes: int
        Size of the classification head output

        input_channels: int
        Number of channels of the input (3 for a RGB image)

        latent_channels: int
        Number of channels of the latent space vector
        """
        super(ClassifConvAE, self).__init__()

        self.latent_representation = None

        # Auto encoder layers
        self.encoder = ConvolutionalEncoder(input_channels=input_channels, output_channels=latent_channels)
        self.decoder = ConvolutionalDecoder(input_channels=latent_channels, output_channels=input_channels)

        # Classification heads
        self.n_classes = n_classes
        self.latent_shape = (latent_channels, int(input_shape[1] / (2 ** 4)), int(input_shape[2] / (2 ** 4)))

        self.fcs = nn.ModuleList([nn.Linear(in_features=self.latent_shape[0]*self.latent_shape[1]*self.latent_shape[2], out_features=self.n_classes[i]) for i in range(len(n_classes))])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        """
        Arguments:
        ----------
        input: torch.Tensor
        Input of the model to forward

        Returns:
        ---------
        tuple of torch.Tensor
        First argument is the autoencoder output, then the classifier outputs
        """
        latent = self.encoder(input)
        self.latent_representation = latent.detach().to('cpu')

        output_AE = self.decoder(latent)
        # Last interpolation to retrieve initial size (not necessary if initiail shape is a multiple of  2⁴)
        output_AE = F.interpolate(output_AE, size=(input.shape[-2], input.shape[-1]))

        output_classif = []
        latent = latent.reshape((latent.shape[0], -1))
        for fc in self.fcs:
            if not self.training:  # If in evaluation mode, apply Softmax
                output_classif.append(self.softmax(fc(latent)))
            else:
                output_classif.append(fc(latent))
            # output = F.log_softmax(self.dropout(self.fc(x)), dim=1) # Not necessary if using CrossEntropyLoss


        return (output_AE,) + tuple(output_classif)

###############################################################################
###############################################################################

if __name__=="__main__":
    from torchsummary import summary

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Creating dummy data
    nb_channels, h_in, w_in = 3, 224, 96
    # nb_channels, h_in, w_in = 3, 16, 16
    data = torch.randn((1, nb_channels, h_in, w_in)).to(device)

    # Creating the model
    model = ClassifConvAE(input_shape=(nb_channels, h_in, w_in), n_classes = [4], input_channels=nb_channels, latent_channels=4).to(device)

    # Summary of the model
    # summary(model, (nb_channels, h_in, w_in))

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
    print(f"Compression rate of {original_dim/reduced_dim}")

    # print(f"Output of the model: {output}")
    print(f"Output sample dimension: {output[0].shape} {output[1].shape}")