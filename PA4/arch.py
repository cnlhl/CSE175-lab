#
# arch.py
#
# This script implements three Python classes for three different artificial
# neural network architectures: no hidden layer, one hidden layer, and two
# hidden layers. Note that this script requires the installation of the
# PyTorch (torch) Python package.
#
# This content is protected and may not be shared, uploaded, or distributed.
#
# PLACE ANY COMMENTS, INCLUDING ACKNOWLEDGMENTS, HERE.
# ALSO, PROVIDE ANSWERS TO THE FOLLOWING TWO QUESTIONS.
#
# Which network architecture achieves the lowest training set error?
# the third one AnnTwoHid() acheives the lowest training set error

# Which network architecture tends to exhibit the best testing set accuracy?
# also the third one AnnTwoHid() tends to exhibit the best testing set accuracy
#
# PLACE YOUR NAME AND THE DATE HERE
# Haolin Li 12/1/2023

# PyTorch - Deep Learning Models
import torch.nn as nn
import torch.nn.functional as F


# Number of input features ...
input_size = 4
# Number of output classes ...
output_size = 3


class AnnLinear(nn.Module):
    """Class describing a linear artificial neural network, with no hidden
    layers, with inputs directly projecting to outputs."""

    def __init__(self):
        super().__init__()
        # PLACE NETWORK ARCHITECTURE CODE HERE
        self.fc1 = nn.Linear(input_size, output_size)
        

    def forward(self, x):
        # PLACE YOUR FORWARD PASS CODE HERE
        fc1 = self.fc1(x)
        y_hat = fc1
        return y_hat


class AnnOneHid(nn.Module):
    """Class describing an artificial neural network with one hidden layer,
    using the rectified linear (ReLU) activation function."""

    def __init__(self):
        super().__init__()
        # PLACE NETWORK ARCHITECTURE CODE HERE
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, output_size)

    def forward(self, x):
        # PLACE YOUR FORWARD PASS CODE HERE
        fc1 = self.fc1(x)
        relu = F.relu(fc1)
        fc2 = self.fc2(relu)
        y_hat = fc2
        return y_hat


class AnnTwoHid(nn.Module):
    """Class describing an artificial neural network with two hidden layers,
    using the rectified linear (ReLU) activation function."""

    def __init__(self):
        super().__init__()
        # PLACE NETWORK ARCHITECTURE CODE HERE
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 12)
        self.fc3 = nn.Linear(12, output_size)

    def forward(self, x):
        # PLACE YOUR FORWARD PASS CODE HERE
        fc1 = self.fc1(x)
        relu1 = F.relu(fc1)
        fc2 = self.fc2(relu1)
        relu2 = F.relu(fc2)
        fc3 = self.fc3(relu2)
        y_hat = fc3
        return y_hat
