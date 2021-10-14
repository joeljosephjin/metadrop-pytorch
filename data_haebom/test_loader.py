import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0,
    help='GPU id')
parser.add_argument('--mode', type=str, default='meta_train',
    help='either meta_train or meta_test')
parser.add_argument('--savedir', type=str, default=None,
    help='save directory')
parser.add_argument('--save_freq', type=int, default=1000,
    help='save frequency')
parser.add_argument('--n_train_iters', type=int, default=60000,
    help='number of meta-training iterations')
parser.add_argument('--n_test_iters', type=int, default=1000,
    help='number of meta-testing iterations')
parser.add_argument('--dataset', type=str, default='omniglot',
    help='either omniglot or mimgnet')
parser.add_argument('--way', type=int, default=20,
    help='number of classes per task')
parser.add_argument('--shot', type=int, default=1,
    help='number of training examples per class')
parser.add_argument('--query', type=int, default=5,
    help='number of test examples per class')
parser.add_argument('--metabatch', type=int, default=16,
    help='number of tasks per each meta-iteration')
parser.add_argument('--meta_lr', type=float, default=1e-3,
    help='meta learning rate')
parser.add_argument('--inner_lr', type=float, default=0.1,
    help='inner-gradient stepsize')
parser.add_argument('--n_steps', type=int, default=5,
    help='number of inner-gradient steps')
parser.add_argument('--n_test_mc_samp', type=int, default=1,
    help='number of MC samples to evaluate the expected inner-step loss')
parser.add_argument('--maml', action='store_true', default=False,
    help='whether to convert this model back to the base MAML or not')
args = parser.parse_args()

from data import Data

data = Data(args)

#old model
for episode in range(10): # i'm assuming each episode carries a single new task
    data_episode = data.generate_episode(args, meta_training=True, n_episodes=args.metabatch)
    xtr, ytr, xte, yte = data_episode
    print(len(xtr))
    # for loop of map (16 iters?)
    # for n_steps range(5):
        # print()

# new model
for episode in range(10):
    for n_task in range(5):
        pass
        # dataloader_task = ...




