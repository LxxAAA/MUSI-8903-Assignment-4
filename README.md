In this assignment, you will use PyTorch to build a Variational AutoEncoder (VAE) model which will be trained learn to reconstruct bars of folk music. 

Download the starter code from the link provided on Canvas. In this folder, you will find the following: 
* `./dat/`: This directory contains the saved `torch.TensorDataset` object (named as `BarDataset`) and a text file `dicts.txt` which contain the note to index and index to note dictionaries.
* `./dataloaders/`: This directory contains the `bar_dataset.py` file which implements the `BarDataset` class. All the methods of this class are already implemented for your use. 
* `./models/`: This directory contains your 3 model files: a) `bar_vae.py` which implements your VAE model and its forward pass, b) `encoder.py` which implements the encoder of your VAE, and c) `decoder.py` which implements the decoder of your VAE. This also contains a `saved/` directory which will save your trained models.
* `./runs/`: This directory will contain the `.txt` file which will log your training statistics.
* `./utils/`: This directory contains 3 files: a) `helpers.py` which contain some pre-implemented helper functions, b) `trainer.py` which implements the `VAETrainer` class, and c) `tester.py` which implements the `VAETester` class.
* `script_train.py`: This is the training script with arguments for model initialization and hyperparameters. You will have to run this script after completing the required code blocks to train the model. 
* `README.md`: this readme file which you are now reading. 

Like in assignment 3, we have provided the data scaffolding and you will need to write code within the spaces allocated. Do NOT edit any other part of the code in any way.

You will be training one VAE model for this task and implementing the encoder, decoder and forward pass for the model. There are no restrictions on the model type. You may implement convolutional / recurrent / fully connected layers as part of your encoder and decoder architectures. You are also free to choose the dimensionality of your latent space. 

## Part 1: Setup (0 points)
You will need to setup a few things before you start.
1. Open a new terminal window. Activate the conda environment you are have been using for the previous assignments and install the `tqdm` package by running:
 ```
 conda install tqdm
 ```
2. Then install the `music21` package by navigating to a suitable place on your computer and running the following commands in the terminal.
 ```
 git clone https://github.com/cuthbertLab/music21.git
 cd music21
 python setup.py install
 ```
 This will clone the `music21` repository on the location you navigated and install the python package.
3. You will also need to install MuseScore on your system. Follow the instructions given at https://musescore.org/en/handbook/installation. You would have to configure MuseScore to be used with the command line. 

## Part 2: VAE Models (50 points)
Implement your VAE model by completing the `__init__()` and `forward()` methods in `bar_vae.py`, `encoder.py` and `decoder.py` located in the `./models/` directory. Follow the instructions carefully and make sure you pay attention to the type of objects and the dimensionality of the tensors / variables to be returned by the methods.

## Part 3: Training Methods (30 points)
Complete the following methods in the `VAETrainer` class in the `./utils/trainer.py` file. 
* `update_scheduler()`: Updates the learing rate depending upon the epoch number.
* `mean_crossentropy_loss()`: Computes the mean cross entropy loss for the VAE output.
* `mean_accuracy()`: Comptues the mean reconstruction accuracy for the VAE output.
* `compute_kld_loss()`: Computes the KL-Divergence loss between the encoder output and the prior distributions. 

## Part 4: Testing Methods (20 points)
Complete the following methods in the `VAETester` class in the `./utils/tester.py` file. 
* `loss_and_acc_test()`: Computes the loss and accuracy for the test set.
* `decode_mid_point()`: Computes n-points on the line joining two given latent vectors and passes the resulting latent vectors through the decoder. The function should then concatenate all these tensors and return a single tensor as output. 

Finally, run the training script `script_train.py` to train and save your VAE model and the training log. 

## Submission Format
Submit all the following in a zip file named `Assign4_Group#` where `#` is your group number:

* `./dat/`: Containing the `BarDataset` and the `dicts.txt` files.
* `./dataloaders/`: Containing the `bar_dataset.py` dataloader file
* `./models/`: Directory containing your implemented VAE models. The `./models/saved/` folder should contain your best saved model. 
* `./runs/`: Directory containing the training log of your best VAE model (only one `.txt` file).
* `./utils/`: Directory containing the completed `trainer.py` and `tester.py` files. Should also contain the `helpers.py` file. 
* `script_train.py`: Updated with the hyperparameters that you have used for training. 



