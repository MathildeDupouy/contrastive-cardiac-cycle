# On the model of a work by Yamil Vindas,
# by Mathilde Dupouy, June 2024


import torch.nn as nn
import numpy as np
from numpy import floor

def weights_init(m):
    """
    Xavier Normal initialization based on the layer type.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
        # nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)