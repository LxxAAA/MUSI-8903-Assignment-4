import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

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
        self.hidden_dim = 64
        self.num_layers = 2

        enc_mid_point = (z_dim + self.hidden_dim) // 2
        self.fc_encode_1 = nn.Linear(z_dim, enc_mid_point)
        self.fc_encode_2 = nn.Linear(enc_mid_point, self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)
        
        dec_mid_point = (self.hidden_dim + num_notes) // 2
        self.fc_decode_1 = nn.Linear(self.hidden_dim, dec_mid_point)
        self.fc_decode_2 = nn.Linear(dec_mid_point, num_notes)
        # self.conv = nn.Conv1d(1, self.num_notes, 1)
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
        # you may use score_tensor if you are using a
        # recurrent decoder
        # pdb.set_trace()
        weights = []
        samples = []

        encoded = self.fc_encode_2(F.relu(self.fc_encode_1(z)))
        h, c = self.init_hidden_and_cell(z.size(0))
        for _ in range(self.seq_len):
            lstm_out, (h, c) = self.lstm(encoded, (h, c))
            # full_lstm_out = torch.cat([full_lstm_out, lstm_out], dim=1)
            decoded = self.fc_decode_2(F.relu(self.fc_decode_1(lstm_out)))
            distr = F.softmax(decoded, dim=2)
            weights.append(distr)
            samples.append(distr.squeeze(1).multinomial(1))

        # pdb.set_trace()
        # weights = F.softmax(decoded, dim=2)
        # samples = weights.multinomial(1)
        weights = torch.cat(weights, dim=1)
        samples = torch.cat(samples, dim=1)
        #####################################
        # END OF YOUR CODE
        #####################################
        return weights, samples





