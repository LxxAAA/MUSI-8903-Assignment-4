import argparse
import torch
from models.bar_vae import MyBarVAE
from dataloaders.bar_dataset import BarDataset
from utils.trainer import VAETrainer
from utils.tester import VAETester

# Training settings
parser = argparse.ArgumentParser(description='Assign4: Bar VAE')
# Hyper-Parameters
parser.add_argument('--lr', type=float, metavar='LR', default=0.001,
                    help='learning rate')
parser.add_argument('--num_epochs', type=int, metavar='N', default=10,
                    help='number of epochs to train')
parser.add_argument('--batch_size', default=128)
parser.add_argument('--train', default=True)
# Other configuration
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# initialize dataset
dataset = BarDataset()

# initialize model
model = MyBarVAE(dataset, use_cuda=args.cuda)
if args.cuda:
    model.cuda()

# initialize trainer
trainer = VAETrainer(
    dataset=dataset,
    model=model,
    lr=args.lr,
    use_cuda=args.cuda
)

# train and test model
if args.train:
    # train model
    trainer.train_model(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs
    )
else:
    # load model
    model.load()
    if args.cuda:
        model.cuda()
    model.eval()

tester = VAETester(
    dataset=dataset,
    model=model,
    use_cuda=args.cuda
)
tester.test_model(batch_size=args.batch_size)
tester.eval_interpolation()
