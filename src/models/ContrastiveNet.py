# On the model of a work from the Github repository adambielski/siamese-triplet,
# by Mathilde Dupouy, July 2024

import torch
import torch.nn as nn
from torchsummary import summary

class ContrastiveNet(nn.Module):
    """
    A generic network based on an encoder to be trained with a contrastive learning framework.

    Attributes
    ----------
    embedding_net: nn.Module
    instantiation of the nn.Module class of the encoder architecture

    latent_representation: torch.Tensor
    compressed representation (output of the bottleneck) on the cpu of the last input
    """
    def __init__(self, embedding_net):
        """
        Arguments:
        ----------
        embedding_net: nn.Module
        instantiation of the nn.Module class of the encoder architecture
        """
        super(ContrastiveNet, self).__init__()
        self.embedding_net = embedding_net

        self.latent_representation = None

    def forward(self, x_anchor, x_pos=None, x_neg=None):
        """
        Arguments:
        ----------
        x_anchor: torch.Tensor
        Input of the network, seen as an anchor in the contrastive learning framework

        x_pos: torch.Tensor
        Batch of positive samples associated to the anchor, used for training only

        x_neg: torch.Tensor
        Batch of negative samples associated to the anchor, used for training only

        Returns:
        --------
        Inputs projections in the latent space
        """
        output_anchor = self.embedding_net(x_anchor)
        self.latent_representation = output_anchor
        if self.training:
            assert x_pos is not None, "[ContrastiveNet] No positive sample(s) given for training."
            assert x_neg is not None, "[ContrastiveNet] No negative sample(s) given for training."
            output_pos = torch.empty((x_pos.shape[0], x_pos.shape[1], output_anchor.shape[1], output_anchor.shape[2], output_anchor.shape[3])).to(x_anchor.device)
            for i, pos_batch in enumerate(x_pos):
                output_pos[i] = self.embedding_net(pos_batch)
            output_neg = torch.empty((x_neg.shape[0], x_neg.shape[1], output_anchor.shape[1], output_anchor.shape[2], output_anchor.shape[3])).to(x_anchor.device)
            for i, neg_batch in enumerate(x_neg):
                output_neg[i] = self.embedding_net(neg_batch)
            return output_anchor, output_pos, output_neg
        else:
            return output_anchor

    def get_embedding(self, x):
        return self.embedding_net(x)
    
    def get_encoder(self):
        model_copy = type(self.embedding_net)()
        model_copy.load_state_dict(self.embedding_net.state_dict())
        return model_copy
    
class ContrastiveNetAE(nn.Module):
    """
    A generic network based on an encoder to be trained
    with a contrastive learning framework combined with an autoencoder task.

    Attributes
    ----------
    encoder: nn.Module
    instantiation of an encoder

    decoder: nn.Module
    instantiation of a decoder

    latent_representation: torch.Tensor
    compressed representation (output of the bottleneck) on the cpu of the last input
    """
    def __init__(self, encoder, decoder):
        super(ContrastiveNetAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.latent_representation = None

    def forward(self, x_anchor, x_pos=None, x_neg=None):
        """
        Arguments:
        ----------
        x_anchor: torch.Tensor
        Input of the network, seen as an anchor in the contrastive learning framework

        x_pos: torch.Tensor
        Batch of positive samples associated to the anchor, used for training only

        x_neg: torch.Tensor
        Batch of negative samples associated to the anchor, used for training only

        Returns:
        --------
        Inputs projection in the latent space
        """
        output_anchor = self.encoder(x_anchor)
        reconstruction = self.decoder(output_anchor)
        self.latent_representation = output_anchor
        if self.training:
            assert x_pos is not None, "[ContrastiveNet] No positive sample(s) given for training."
            assert x_neg is not None, "[ContrastiveNet] No negative sample(s) given for training."
            output_pos = torch.empty((x_pos.shape[0], x_pos.shape[1], output_anchor.shape[1], output_anchor.shape[2], output_anchor.shape[3])).to(x_anchor.device)
            for i, pos_batch in enumerate(x_pos):
                output_pos[i] = self.encoder(pos_batch)
            output_neg = torch.empty((x_neg.shape[0], x_neg.shape[1], output_anchor.shape[1], output_anchor.shape[2], output_anchor.shape[3])).to(x_anchor.device)
            for i, neg_batch in enumerate(x_neg):
                output_neg[i] = self.encoder(neg_batch)
            return reconstruction, output_anchor, output_pos, output_neg
        else:
            return reconstruction, output_anchor

    def get_embedding(self, x):
        return self.encoder(x)
    
    def get_encoder(self):
        model_copy = type(self.encoder)()
        model_copy.load_state_dict(self.encoder.state_dict())
        return model_copy


###############################################################################
###############################################################################

if __name__=="__main__":
    import torch
    from torchsummary import summary
    from Models.ConvolutionalAE import ConvolutionalEncoder

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Creating dummy data
    # nb_channels, h_in, w_in = 3, 214, 100
    nb_channels, h_in, w_in = 3, 224, 96
    # nb_channels, h_in, w_in = 3, 16, 16
    data = (torch.randn((1, nb_channels, h_in, w_in)).to(device), torch.randn((1, nb_channels, h_in, w_in)).to(device), torch.randn((1, nb_channels, h_in, w_in)).to(device))
    latent_channels = 4

    # Creating the embedding net
    encoder = ConvolutionalEncoder(input_channels=nb_channels, output_channels=latent_channels, k_size=3).to(device)
    # Creating the ContrastiveNet
    model = ContrastiveNet(encoder)
    # Summary of the model
    summary(model, data)

    # # Parameters of the network
    # for name, param in model.named_parameters():
    #     print(name)

    # Evaluating the model
    output = model(*data)
    original_dim = nb_channels*h_in*w_in

    print(f"Output sample dimension: {output[0].shape}")
