import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from utils.helpers import *


class VAETrainer(object):
    """
    Class for training a VAE model
    """
    def __init__(self, dataset,
                 model,
                 lr=1e-4,
                 use_cuda=False):
        """
        Initializes the trainer class
        :param dataset: torch Dataset object
        :param model: torch.nn object
        :param lr: float, learning rate
        """
        self.dataset = dataset
        self.model = model
        # initialize optimizer for trainer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )
        self.use_cuda = use_cuda

    def train_model(self, batch_size, num_epochs):
        """
        Trains the model
        :param batch_size: int, number of datapoints in the minibatch
        :param num_epochs: int, number of epochs to train for
        :return: None
        """
        # set-up log parameters
        log_parameters = VAETrainer.log_init()

        # get training and validation dataloaders
        generator_train, generator_val, _ = self.dataset.data_loaders(
            batch_size=batch_size,
            split=(0.70, 0.20)
        )
        print('Num Train Batches: ', len(generator_train))
        print('Num Valid Batches: ', len(generator_val))

        # train epochs
        for epoch_index in range(num_epochs):
            # update training scheduler
            self.update_scheduler(epoch_index)

            # run training loop on training data
            self.model.train()
            mean_loss_train, mean_accuracy_train = self.loss_and_acc_on_epoch(
                data_loader=generator_train,
                train=True)

            # run evaluation loop on validation data
            self.model.eval()
            mean_loss_val, mean_accuracy_val = self.loss_and_acc_on_epoch(
                data_loader=generator_val,
                train=False)

            # log epoch stats
            log_parameters = VAETrainer.log_epoch_stats(
                log_parameters,
                epoch_index,
                mean_loss_train,
                mean_accuracy_train,
                mean_loss_val,
                mean_accuracy_val
            )

            # print epoch stats
            self.print_epoch_stats(
                epoch_index,
                num_epochs,
                mean_loss_train,
                mean_accuracy_train,
                mean_loss_val,
                mean_accuracy_val
            )
            # save model
            self.model.save()
        # save training stats
        if log_parameters is not None:
            log_filename = 'runs/training_stats_log.txt'
            f = open(log_filename, 'w')
            f.write(str(log_parameters))
            f.close()
            print('Saved training stats log')

    def loss_and_acc_on_epoch(self, data_loader, train=True):
        """
        Computes the loss and accuracy for an epoch
        :param data_loader: torch dataloader object
        :param train: bool, performs the backward pass and gradient descent if TRUE
        :return: loss values and accuracy percentages
        """
        mean_loss = 0
        mean_accuracy = 0
        for sample_id, batch in tqdm(enumerate(data_loader)):
            # process batch data
            batch_data = to_cuda_variable_long(batch[0], self.use_cuda)

            # zero the gradients
            self.optimizer.zero_grad()

            # compute loss for batch
            loss, accuracy = self.loss_and_acc_for_batch(
                batch_data, train=train
            )

            # compute backward and step if train
            if train:
                loss.backward()
                self.optimizer.step()

            # compute mean loss and accuracy
            mean_loss += to_numpy(loss.mean())
            if accuracy is not None:
                mean_accuracy += to_numpy(accuracy)

        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        )

    def loss_and_acc_for_batch(self, batch, train=True):
        """
        Computes the loss and accuracy for the batch
        Must return (loss, accuracy) as a tuple, accuracy can be None
        :param batch: torch Variable,
        :param train: bool, True is backward pass is to be performed
        :return: scalar loss value, scalar accuracy value
        """
        score = batch
        # perform forward pass of model
        weights, samples, z_dist, prior_dist = self.model(
            measure_score_tensor=score,
            train=train
        )

        # compute loss
        recons_loss = self.mean_crossentropy_loss(weights=weights, targets=score)
        kld_loss = self.compute_kld_loss(z_dist, prior_dist)
        loss = recons_loss + kld_loss

        # compute accuracy
        accuracy = self.mean_accuracy(weights=weights, targets=score)
        return loss, accuracy

    def update_scheduler(self, epoch_num):
        """
        Updates the learning rate of the optimizer
        :param epoc_num: int, index corresponding to the epoch number
        :return: None
        """
        ##################################
        # INSERT YOUR CODE HERE
        # to update the learning rate
        ##################################
        DECAY = 0.0001
        new_lr = None

        for param_group in self.optimizer.param_groups:
            # Basic decay
            param_group['lr'] = param_group['lr'] * 1 / (1 + DECAY * epoch_num)
            new_lr = param_group['lr']

        print("New learning rate:", new_lr)
        ##################################
        # END OF YOUR CODE
        ##################################

    @staticmethod
    def mean_crossentropy_loss(weights, targets):
        """
        Evaluates the cross entropy loss
        :param weights: torch Variable,
                (batch_size, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, seq_len)
        :return: recons_loss, torch tensor,
                 single float, cross-entropy loss
        """
        #####################################
        # INSERT YOUR CODE HERE
        #####################################
        # define the loss criterion
        # compute a mean cross entropy loss
        vocab_size = int(weights.size()[-1])
        weighting = torch.ones([vocab_size])
        weighting[21] = 0.2 # fuck note continuations
        loss_fn = nn.NLLLoss(weight=weighting)
        recons_loss = torch.nn.NLLLoss()(weights.view(-1, weights.size()[-1]), targets.view(-1))
        #####################################
        # END OF YOUR CODE
        #####################################
        return recons_loss

    @staticmethod
    def mean_accuracy(weights, targets):
        """
        Evaluates the mean accuracy in prediction
        :param weights: torch Variable,
                (batch_size, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, seq_len)
        :return accuracy, torch tensor,
                single float, reconstruction accuracy
        """
        #####################################
        # INSERT YOUR CODE HERE
        #####################################
        # compute the reconstruction accuracy
        recon = weights.argmax(dim=2)
        recon = recon.type(torch.LongTensor)
        num_equal = torch.eq(recon, targets)
        total_equal = num_equal.sum()
        accuracy = torch.tensor(float(total_equal) / targets.numel())

        #####################################
        # END OF YOUR CODE
        #####################################
        return accuracy

    @staticmethod
    def compute_kld_loss(z_dist, prior_dist, beta=0.5):
        """
        Computes the KL-divergence loss for the given arguments

        :param z_dist: torch.nn.distributions object, output of the encoder
        :param prior_dist: torch.nn.distributions, prior Normal distribution
        :param beta: hyperparameter, how much weightage should be given
        :return: kld_loss, torch tensor,
                 single float, kl divergence loss
        """
        #####################################
        # INSERT YOUR CODE HERE
        #####################################
        # compute the kl-divergence
        kld_loss = beta * torch.mean(torch.distributions.kl.kl_divergence(z_dist, prior_dist))
        #####################################
        # END OF YOUR CODE
        #####################################
        return kld_loss

    @staticmethod
    def log_init():
        """
        Initializes the log element

        :return: None
        """
        log_parameters = {
            'x': [],
            'loss_train': [],
            'acc_train': [],
            'loss_val': [],
            'acc_val': []
        }
        return log_parameters

    @staticmethod
    def log_epoch_stats(
            log_parameters,
            epoch_index,
            loss_train,
            acc_train,
            loss_val,
            acc_val
        ):
        """
        Logs the epoch statistics
        :param log_parameters: dict, container for epoch stats
        :param epoch_index: int,
        :param loss_train: float,
        :param acc_train: float,
        :param loss_val: float,
        :param acc_val: float
        :return: log_parameters
        """
        log_parameters['x'].append(epoch_index)
        log_parameters['loss_train'].append(loss_train)
        log_parameters['acc_train'].append(acc_train)
        log_parameters['loss_val'].append(loss_val)
        log_parameters['acc_val'].append(acc_val)
        return log_parameters

    @staticmethod
    def print_epoch_stats(
            epoch_index,
            num_epochs,
            mean_loss_train,
            mean_accuracy_train,
            mean_loss_val,
            mean_accuracy_val
        ):
        """
        Prints the epoch statistics
        :param epoch_index: int,
        :param num_epochs: int,
        :param mean_loss_train: float,
        :param mean_accuracy_train:float,
        :param mean_loss_val: float,
        :param mean_accuracy_val: float
        :return: None
        """
        print(
            'Train Epoch:', epoch_index + 1, '/', num_epochs)
        print(
            '\tTrain Loss:', mean_loss_train,
            '\tTrain Accuracy:', mean_accuracy_train * 100, '%'
        )
        print(
            '\tValid Loss:', mean_loss_val,
            '\tValid Accuracy:', mean_accuracy_val * 100, '%'
        )

