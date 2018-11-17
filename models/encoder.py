import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class MyBarEncoder(torch.nn.Module):
    def __init__(self, seq_len, z_dim, use_cuda=False):
        super(MyBarEncoder, self).__init__()
        self.seq_len = seq_len
        self.z_dim = z_dim
        self.use_cuda = use_cuda
        #####################################
        # INSERT YOUR CODE HERE
        # initialize your VAE encoder model
        #####################################
        self.fc_1 = nn.Linear(self.seq_len, self.seq_len // 2)
        self.fc_21 = nn.Linear(self.seq_len // 2, self.z_dim)
        self.fc_22 = nn.Linear(self.seq_len // 2, self.z_dim)
        #####################################
        # END OF YOUR CODE
        #####################################

    def __repr__(self):
        """
        String Representation of class
        :return: string, class representation
        """
        return 'MyBarEncoder'

    def forward(self, score_tensor):
        """
        Performs the forward pass of the model, overrides torch method
        :param score_tensor: torch Variable
                (batch_size, measure_seq_len)
        :return: torch distribution
                (batch_size, self.z_dim)
        """
        #####################################
        # INSERT YOUR CODE HERE
        # forward pass of the VAE encoder
        #####################################
        # initial embedding
        score_tensor = score_tensor.type(torch.FloatTensor)
        embedded = F.relu(self.fc_1(score_tensor))

        # compute the mean
        mu = self.fc_21(embedded)

        # compute the logvar
        logvar = self.fc_22(embedded)
        sigma = torch.exp(0.5 * logvar)

        # use these to create a torch.distribution object
        z_distribution = Normal(mu, sigma)
        #####################################
        # END OF YOUR CODE
        #####################################
        return z_distribution
