import torch
import torch.nn as nn
import torch.nn.functional as F


class MyBarDecoder(torch.nn.Module):
    def __init__(self, seq_len, z_dim, num_notes, use_cuda=False):
        super(MyBarDecoder, self).__init__()
        self.use_cuda = use_cuda
        self.seq_len = seq_len
        self.z_dim = z_dim
        self.num_notes = num_notes
        #####################################
        # INSERT YOUR CODE HERE
        # initialize your VAE decoder model
        #####################################
        self.fc_1 = nn.Linear(self.z_dim, self.seq_len // 2)
        self.fc_2 = nn.Linear(self.seq_len // 2, self.seq_len)
        self.conv = nn.Conv1d(1, self.num_notes, 1)
        #####################################
        # END OF YOUR CODE
        #####################################

    def __repr__(self):
        """
        String Representation of class
        :return: string, class representation
        """
        return 'MyBarDecoder'

    def forward(self, z, score_tensor, train):
        """
        Performs the forward pass of the model, overrides torch method
        :param z: torch tensor,
                (batch_size, self.z_dim)
        :param score_tensor: torch tensor
                (batch_size, measure_seq_len)
        :param train: bool, performing training if true
        :return: weights: torch Variable, softmax over all possible outputs
                (batch_size, measure_seq_len, num_notes)
        :return: samples: torch Variable, , selected note for each tick
                (batch_size, 1, measure_seq_len)
        """
        #####################################
        # INSERT YOUR CODE HERE
        # forward pass of the VAE decoder
        #####################################
        # use the z to reconstruct the input
        # you may score_tensor if you are using a
        # recurrent decoder
        linear = F.sigmoid(self.fc_2(F.relu(self.fc_1(z))))
        conv = self.conv(linear.unsqueeze(1))
        weights = F.softmax(conv, dim=1).permute(0, 2, 1)
        samples = torch.argmax(weights, dim=2)

        #####################################
        # END OF YOUR CODE
        #####################################
        return weights, samples





