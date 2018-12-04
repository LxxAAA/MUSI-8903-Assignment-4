import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import pdb

class MyBarEncoder(torch.nn.Module):
    def __init__(self, seq_len, z_dim, num_notes, use_cuda=False):
        super(MyBarEncoder, self).__init__()
        self.use_cuda = use_cuda
        self.seq_len = seq_len
        self.z_dim = z_dim
        self.num_notes = num_notes
        #####################################
        # INSERT YOUR CODE HERE
        # initialize your VAE encoder model
        #####################################
        self.embedding_dim = 16
        self.hidden_dim = 64
        self.num_dirs = 2
        self.num_layers = 2
        
        self.embedder = nn.Embedding(num_notes, self.embedding_dim)

        # mid = (self.embedding_dim + self.hidden_dim) // 2
        self.fc_encode = nn.Linear(self.embedding_dim, self.hidden_dim)
        # self.fc_encode_2 = nn.Linear(mid, self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, 
                batch_first=True, bidirectional=(self.num_dirs == 2))

        mid_point = (self.hidden_dim * self.num_dirs + self.z_dim) // 2
        self.fc_decode_1 = nn.Linear(self.hidden_dim * self.num_dirs, mid_point)
        
        self.fc_mean = nn.Linear(mid_point, self.z_dim)
        self.fc_stdev = nn.Linear(mid_point, self.z_dim)
        self.init_params()
        #####################################
        # END OF YOUR CODE
        #####################################

    def __repr__(self):
        """
        String Representation of class
        :return: string, class representation
        """
        return 'MyBarEncoder'

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
        return

    def init_hidden_and_cell(self, batch_size):
        hidden = torch.zeros((self.num_layers * self.num_dirs, batch_size, self.hidden_dim))
        cell = torch.zeros((self.num_layers * self.num_dirs, batch_size, self.hidden_dim))
        if self.use_cuda and torch.cuda.is_available():
            hidden, cell = hidden.cuda(), cell.cuda()
        return hidden, cell

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
        # pdb.set_trace()
        # pdb.set_trace() 
        h0, c0 = self.init_hidden_and_cell(score_tensor.size(0))
        score_tensor = score_tensor.type(torch.LongTensor)

        embedded = self.embedder(score_tensor)
        # encoded = self.fc_encode_2(F.relu(self.fc_encode_1(embedded)))
        encoded = self.fc_encode(embedded)
        lstm_out, _ = self.lstm(encoded, (h0, c0))
        lstm_out = lstm_out.contiguous()[:, -1, :].unsqueeze(1)
        # compute the mean
        mu = self.fc_mean(F.relu(self.fc_decode_1(lstm_out)))
        # compute the logvar
        logvar = self.fc_stdev(F.relu(self.fc_decode_1(lstm_out)))
        sigma = torch.exp(0.5 * logvar)
        # use these to create a torch.distribution object
        z_distribution = Normal(mu, sigma)
        #####################################
        # END OF YOUR CODE
        #####################################
        return z_distribution
