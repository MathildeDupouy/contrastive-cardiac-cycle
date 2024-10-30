# On the model of a work from the Github repository adambielski/siamese-triplet,
# by Mathilde Dupouy, July 2024
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class ContrastiveLossCosSim(nn.Module):
    """
    Contrastive loss from Khosla 2020 - Supervised contrastive learning using cosine similarity rather than dot product.
    Instantiated with a temperature as argument.
    Forward is called with a Tensor anchor, and a positive and negative tensor which are batches of tensors with the same shapes as the anchor.
    """
    def __init__(self, temperature):
        super(ContrastiveLossCosSim, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negative, size_average=True):
        anchor_flat = anchor.reshape((anchor.shape[0], -1))
        positive_flat = positive.reshape((positive.shape[0], positive.shape[1], -1))
        negative_flat = negative.reshape((negative.shape[0], negative.shape[1], -1))

        softmax = torch.nn.Softmax(dim=0)
        losses = torch.empty((anchor_flat.shape[0], 1))
        for i, a in enumerate(anchor_flat):
            pos_vector = torch.matmul(positive_flat[i], a) / (self.temperature * torch.linalg.vector_norm(positive_flat[i], dim = 1) * torch.linalg.vector_norm(anchor_flat[i], dim = 0))
            neg_vector = torch.matmul(negative_flat[i], a) / (self.temperature * torch.linalg.vector_norm(negative_flat[i], dim = 1) * torch.linalg.vector_norm(anchor_flat[i], dim = 0))
            softmax_vector = softmax(torch.concat((pos_vector, neg_vector), dim=0))
            losses[i] = -torch.log(softmax_vector[:pos_vector.shape[0]]).mean()
        return losses.mean() if size_average else losses.sum()
    
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss from Khosla 2020 - Supervised contrastive learning using cosine similarity rather than dot product.
    Instantiated with a temperature as argument.
    Forward is called with a Tensor anchor, and a positive and negative tensor which are batches of tensors with the same shapes as the anchor.
    """
    def __init__(self, temperature):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negative, size_average=True):
        eps = 1e-14
        anchor_flat = anchor.reshape((anchor.shape[0], -1))
        positive_flat = positive.reshape((positive.shape[0], positive.shape[1], -1))
        negative_flat = negative.reshape((negative.shape[0], negative.shape[1], -1))
        anchor_flat = torch.nn.functional.normalize(anchor_flat, p=2.0, dim = 1)
        # positive_flat = torch.nn.functional.normalize(positive_flat, p=2.0, dim = 2)
        # negative_flat = torch.nn.functional.normalize(negative_flat, p=2.0, dim = 2)

        softmax = torch.nn.Softmax(dim=0)
        losses = torch.empty((anchor_flat.shape[0], 1))
        for i, a in enumerate(anchor_flat):
            pos_vector = torch.matmul(positive_flat[i], a) / self.temperature
            neg_vector = torch.matmul(negative_flat[i], a) / self.temperature
            max_vec = torch.max(torch.concat((pos_vector, neg_vector), dim=0)).detach()
            pos_vector = pos_vector - max_vec
            neg_vector = neg_vector - max_vec
            # print("pos_vector", pos_vector)
            # print("neg_vector", neg_vector)
            softmax_vector = softmax(torch.concat((pos_vector, neg_vector), dim=0))
            losses[i] = -torch.log(softmax_vector[:pos_vector.shape[0]] + eps).mean()
            # print("ContrastiveLoss", softmax_vector[:pos_vector.shape[0]], losses[i])
        if torch.isnan(losses).any():
            raise ValueError("[ContrastiveLoss] Loss is nan, stopping the training.")
        return losses.mean() if size_average else losses.sum()
    
###############################################################################
###############################################################################

if __name__=="__main__":
    import torch
    from torchsummary import summary
    from Models.ConvolutionalAE import ConvolutionalEncoder

    test_anchor = torch.rand(size=(1, 3, 128, 128))
    test_pos = torch.rand(size = (1, 2, 3, 128, 128))
    # test_pos[0, 0, :] = test_anchor
    # test_pos[0, 1, :] = test_anchor
    test_neg = torch.rand(size = (1, 1, 3, 128, 128))

    loss_fn = ContrastiveLoss(temperature=1.)
    print(loss_fn(test_anchor, test_pos, test_neg))


    