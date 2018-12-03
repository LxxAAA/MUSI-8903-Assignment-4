from random import randint
import torch
from utils.helpers import *
from utils.trainer import VAETrainer


class VAETester(object):
    def __init__(self, dataset, model, use_cuda):
        self.dataset = dataset
        self.model = model
        self.model.eval()
        self.decoder = self.model.decoder
        self.batch_size = 1
        self.train = False
        self.measure_seq_len = self.dataset.beat_subdivisions * 3
        self.use_cuda = use_cuda

    def test_model(self, batch_size):
        """
        Runs the model on the test set
        :param batch_size: int, number of datapoints in minibatch
        :return:
        """
        (_, gen_val, gen_test) = self.dataset.data_loaders(
            batch_size=batch_size,
            split=(0.70, 0.20)
        )
        print('Num Test Batches: ', len(gen_test))
        mean_loss_test, mean_accuracy_test = self.loss_and_acc_test(gen_test)
        print('Testing Model')
        print(
            '\tTest Loss:', mean_loss_test,
            '\tTest Accuracy:', mean_accuracy_test * 100, '%'
        )

    def eval_interpolation(self):
        (_, gen_val, gen_test) = self.dataset.data_loaders(
            batch_size=self.batch_size,
            split=(0.70, 0.20)
        )

        gen_it_test = gen_test.__iter__()
        for _ in range(randint(0, len(gen_test))):
            tensor_score1 = next(gen_it_test)[0]

        gen_it_val = gen_val.__iter__()
        for _ in range(randint(0, len(gen_val))):
            tensor_score2 = next(gen_it_val)[0]

        tensor_score1 = to_cuda_variable(tensor_score1.long(), self.use_cuda)
        tensor_score2 = to_cuda_variable(tensor_score2.long(), self.use_cuda)
        self.test_interpolation(tensor_score1, tensor_score2, 10)

    def decode_mid_point(self, z1, z2, n):
        """
        Decodes the mid-point of two latent vectors
        :param z1: torch tensor, (1, self.z_dim)
        :param z2: torch tensor, (1, self.z_dim)
        :param n: int, number of points for interpolation
        :return: concat_tensor_score, torch tensor,
                 (1, (n+2) * measure_seq_len)
        """
        assert (n >= 1 and isinstance(n, int))
        # compute the score_tensors for z1 and z2
        dummy_score_tensor = to_cuda_variable(torch.zeros(self.batch_size, self.measure_seq_len), self.use_cuda)
        _, sam1 = self.decoder(z1, dummy_score_tensor, self.train)
        _, sam2 = self.decoder(z2, dummy_score_tensor, self.train)
        concat_tensor_score = sam1
        ##################################
        # INSERT YOUR CODE HERE
        # complete the interpolation code below
        ##################################
        # find the interpolation points and run through decoder
        # concatenate to results concat_tensor_score
        incr_difference = z2.sub(z1) / (n + 1)

        for i in range(n):
            interp_z = z1 + (incr_difference * (i + 1))
            _, interp_sam = self.decoder(interp_z, dummy_score_tensor, self.train)
            torch.cat((concat_tensor_score, interp_sam), 1)

        ##################################
        # END OF YOUR CODE
        ##################################
        concat_tensor_score = torch.cat((concat_tensor_score, sam2), 1).view(1, -1)
        return concat_tensor_score

    def test_interpolation(self, tensor_score1, tensor_score2, n=10):
        """
        Tests the interpolation in the latent space for two random points
        :param tensor_score1: torch tensor, (1, measure_seq_len)
        :param tensor_score2: torch tensor, (1, measure_seq_len)
        :param n: int, number of points for interpolation
        :return:
        """
        z_dist1 = self.model.encoder(tensor_score1)
        z_dist2 = self.model.encoder(tensor_score2)
        z1 = z_dist1.loc
        z2 = z_dist2.loc
        tensor_score = self.decode_mid_point(z1, z2, n)
        print(tensor_score)
        score = self.dataset.get_score_from_tensor(tensor_score.cpu())
        score.show()
        return score

    def loss_and_acc_test(self, data_loader):
        """
        :param data_loader: torch data loader object
        :return: mean_loss, float
        :return: mean_accuracy, float
        """
        mean_loss = 0
        mean_accuracy = 0
        for sample_id, batch in enumerate(data_loader):
            score_tensor = to_cuda_variable_long(batch[0], self.use_cuda)
            ##################################
            # INSERT YOUR CODE HERE
            # complete the steps below
            ##################################
            # compute forward pass of VAE model
            weights, samples, z_dist, prior_dist = self.model(
                measure_score_tensor=score_tensor,
                train=False
            )

            # compute reconstruction & kld losses
            recons_loss = VAETrainer.mean_crossentropy_loss(weights=weights, targets=score_tensor)
            kld_loss = VAETrainer.compute_kld_loss(z_dist, prior_dist)
            mean_loss += recons_loss + kld_loss

            # compute accuracy
            mean_accuracy += VAETrainer.mean_accuracy(weights=weights, targets=score_tensor)
            ##################################
            # END OF YOUR CODE
            ##################################
        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        )
