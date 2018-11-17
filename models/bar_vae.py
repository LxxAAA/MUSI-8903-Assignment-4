import os
import torch
from torch import nn, distributions
from models.encoder import MyBarEncoder
from models.decoder import MyBarDecoder


class MyBarVAE(nn.Module):
    def __init__(self, dataset, use_cuda=False):
        """
        Initializes the MyBarVAE class object

        :param dataset: BarDataset object
        """
        super(MyBarVAE, self).__init__()
        self.num_beats_per_measure = 3  # Hardcoded for 3 by 4 measures
        self.measure_seq_len = 18  # Hardcoded for 3 by 4 measures
        self.num_ticks_per_beat = int(self.measure_seq_len / self.num_beats_per_measure)
        self.num_notes = len(dataset.note2index_dicts)
        self.use_cuda = use_cuda
        #####################################
        # INSERT YOUR CODE HERE
        # initialize your VAE model
        #####################################
        # size of the latent space
        self.z_dim = 10  # feel free to change this

        # you may declare and pass arguments below
        # for your encoder & decoder
        # do NOT the change the names below though
        self.encoder = MyBarEncoder(self.measure_seq_len, self.z_dim, use_cuda=self.use_cuda)
        self.decoder = MyBarDecoder(self.measure_seq_len, self.z_dim, use_cuda=self.use_cuda)
        #####################################
        # END OF YOUR CODE
        #####################################

        # location to save model
        self.filepath = os.path.join('models/saved/',
                                     self.__repr__())

    def __repr__(self):
        """
        String Representation of class
        :return: string, class representation
        """
        return 'MyBarVAE'

    def forward(self, measure_score_tensor, train=True):
        """
        Implements the forward pass of the MyBarVAE model
        :param measure_score_tensor: torch Variable,
                (batch_size, measure_seq_length)
        :param train: bool, if True perform training
        :return: weights, torch Variable, softmax over all possible outputs
                 (batch_size, self.measure_seq_length, self.num_notes)
        :return: samples, torch Variable, selected note for each tick
                 (batch_size, 1, self.measure_seq_len)
        :return: z_dist, torch distribution object, output distribution of the encoder,
                 (batch_size, self.z_dim)
        :return: prior_dist, torch distribution object, prior distribution
        """
        # check input dimensions
        seq_len = measure_score_tensor.size(1)
        assert (seq_len == self.measure_seq_len)

        # compute output of encoding layer
        # should return a torch.distributions object
        z_dist = self.encoder(measure_score_tensor)

        # sample from output distribution
        z_tilde = z_dist.rsample()

        #####################################
        # COMPLETE THE CODE BLOCK BELOW
        #####################################
        # compute prior distribution
        # should also be a torch.distributions object
        prior_dist = None # initialize this
        #####################################
        # END OF YOUR CODE
        #####################################

        # compute output of the decoder
        weights, samples = self.decoder(
            z=z_tilde,
            score_tensor=measure_score_tensor,
            train=train
        )

        return weights, samples, z_dist, prior_dist

    def save(self):
        """
        Saves the model
        :return:
        """
        torch.save(self.state_dict(), self.filepath)
        print('Model saved')

    def load(self, cpu=False):
        """
        Loads the model
        :param cpu: bool, specifies if the model should be loaded on the CPU
        :return:
        """
        if cpu:
            self.load_state_dict(
                torch.load(
                    self.filepath,
                    map_location=lambda storage,
                    loc: storage
                )
            )
        else:
            self.load_state_dict(torch.load(self.filepath))
        print('Model loaded')
