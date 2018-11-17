import torch


class MyBarDecoder(torch.nn.Module):
    def __init__(self, use_cuda=False):
        super(MyBarDecoder, self).__init__()
        self.use_cuda = use_cuda
        #####################################
        # INSERT YOUR CODE HERE
        # initialize your VAE decoder model
        #####################################
        pass
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
        pass
        #####################################
        # END OF YOUR CODE
        #####################################
        return weights, samples





