import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

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
        self.embedding_dim = 8
        self.embedder = nn.Embedding(self.num_notes, self.embedding_dim)
        self.hidden_dim = 128
        self.num_layers = 2
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc_1 = nn.Linear(self.seq_len * self.hidden_dim, self.seq_len // 2)
        self.fc_21 = nn.Linear(self.seq_len // 2, self.z_dim)
        self.fc_22 = nn.Linear(self.seq_len // 2, self.z_dim)
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
        hidden = torch.zeros((self.num_layers, batch_size, self.hidden_dim))
        cell = torch.zeros((self.num_layers, batch_size, self.hidden_dim))
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
        score_tensor = score_tensor.type(torch.LongTensor)

        embedded = self.embedder(score_tensor)
        h0, c0 = self.init_hidden_and_cell(score_tensor.size(0))
        lstm_out, _ = self.lstm(embedded, (h0, c0))

        fc_out = F.relu(self.fc_1(lstm_out.contiguous().view(score_tensor.size(0), -1)))

        # compute the mean
        mu = self.fc_21(fc_out)

        # compute the logvar
        logvar = self.fc_22(fc_out)
        sigma = torch.exp(0.5 * logvar)

        # use these to create a torch.distribution object
        z_distribution = Normal(mu, sigma)
        #####################################
        # END OF YOUR CODE
        #####################################
        return z_distribution
