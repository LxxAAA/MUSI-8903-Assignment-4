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
        self.linear = nn.Linear(self.z_dim, self.seq_len)
        self.hidden_dim = 128
        self.num_layers = 2
        self.lstm = nn.LSTM(1, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.seq_len * self.hidden_dim, self.seq_len)
        self.conv = nn.Conv1d(1, self.num_notes, 1)
        self.init_params()
        #####################################
        # END OF YOUR CODE
        #####################################

    def __repr__(self):
        """
        String Representation of class
        :return: string, class representation
        """
        return 'MyBarDecoder'

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
        return

    def init_hidden_and_cell(self, batch_size):
        hidden = torch.zeros((self.num_layers, batch_size, self.hidden_dim))
        cell = torch.zeros((self.num_layers, batch_size, self.hidden_dim))
        if self.use_cuda and torch.cuda.is_available():
            hidden, cell = hidden.cuda(), cell.cuda()
        return hidden, cell

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

        h0, c0 = self.init_hidden_and_cell(z.size(0))
        linear_out = self.linear(z)
        lstm_out, _ = self.lstm(linear_out.unsqueeze(-1), (h0, c0))
        fc_out = self.fc(lstm_out.contiguous().view(z.size(0), -1))
        conv = self.conv(fc_out.unsqueeze(1))
        weights = F.softmax(conv, dim=1).permute(0, 2, 1)
        samples = torch.argmax(weights, dim=2)


        #####################################
        # END OF YOUR CODE
        #####################################
        return weights, samples





