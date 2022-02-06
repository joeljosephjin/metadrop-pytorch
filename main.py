import argparse
import os
from collections import OrderedDict

import torch
import torch.nn as nn

torch.manual_seed(0)
torch.cuda.manual_seed(0)

import numpy as np
np.random.seed(0)

print('tensor:', torch.rand([1,2]))

from model import MetaLearner
from evaluate import accuracy

from tqdm import tqdm
from statistics import mean

from data_haebom.data import Data

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',default='data/Omniglot',help="Directory containing the dataset")
parser.add_argument('--model_dir',default='experiments/base_model',help="Directory containing params.json")
parser.add_argument('--restore_file',default=None,help="Optional, init model weight file")  # 'best' or 'train'
parser.add_argument('--seed',default=1)
parser.add_argument('--dataset',default="Omniglot")
parser.add_argument('--meta_lr',default=1e-3, type=float)
parser.add_argument('--task_lr',default=1e-1, type=float)
parser.add_argument('--num_episodes',default=8, type=int)
parser.add_argument('--num_classes',default=5, type=int)
parser.add_argument('--num_samples',default=1, type=int)
parser.add_argument('--num_query',default=10, type=int)
parser.add_argument('--num_steps',default=100, type=int)
parser.add_argument('--num_inner_tasks',default=8, type=int)
parser.add_argument('--num_train_updates',default=1, type=int)
parser.add_argument('--num_eval_updates',default=1, type=int)
parser.add_argument('--save_summary_steps',default=100, type=int)
parser.add_argument('--num_workers',default=1, type=int)
parser.add_argument('--phi',default=False, action="store_true")
parser.add_argument('--wandb',default=False, action="store_true")


def train_and_evaluate(models,
                       meta_train_classes,
                       meta_test_classes,
                       task_type,
                       meta_optimizer,
                       loss_fn,
                       args):

    model = models['model']

    # params information
    num_classes = args.num_classes
    num_samples = args.num_samples
    num_query = args.num_query
    num_inner_tasks = args.num_inner_tasks
    task_lr = args.task_lr
    start_time = 0

    data = Data(args)

    # for episode in tqdm(range(args.num_episodes)):
    for episode in range(args.num_episodes):
        # Run inner loops to get adapted parameters (theta_t`)
        data_episode = data.generate_episode(args, meta_training=True, n_episodes=num_inner_tasks)
        meta_loss = 0
        accs = []
        xtr, ytr, xte, yte = data_episode
        for n_task in range(num_inner_tasks):

            xtri, ytri, xtei, ytei = xtr[n_task], ytr[n_task], xte[n_task], yte[n_task]
            X_sup, Y_sup = xtri, ytri
            X_sup, Y_sup = X_sup.reshape([-1, 1, 28, 28]).to(args.device), Y_sup.to(args.device) # [5, 784]

            adapted_params = model.cloned_state_dict()

            for _ in range(0, args.num_train_updates):
                Y_sup_hat = model(X_sup, adapted_params)
                loss = loss_fn(Y_sup_hat, torch.argmax(Y_sup, dim=1))

                grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)
                for (key, val), grad in zip(adapted_params.items(), grads):
                    adapted_params[key] = val - task_lr * grad

            X_meta, Y_meta = xtei, ytei
            X_meta, Y_meta = X_meta.reshape([-1, 1, 28, 28]).to(args.device), Y_meta.to(args.device) # [5, 784]

            Y_meta_hat = model(X_meta, adapted_params)

            accs.append(accuracy(Y_meta_hat.data.cpu().numpy(), torch.argmax(Y_meta, dim=1).data.cpu().numpy()))
            loss_t = loss_fn(Y_meta_hat, torch.argmax(Y_meta, dim=1))

            meta_loss += loss_t

        meta_loss /= float(num_inner_tasks)

        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        print('episode:', episode, 'accs (mean of tasks):', mean(accs))


if __name__ == '__main__':
    args = parser.parse_args()

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if 'Omniglot' in args.data_dir and args.dataset == 'Omniglot':
        args.in_channels = 1
        meta_train_classes, meta_test_classes = None, None
        task_type = None

    model = MetaLearner(args).to(args.device)
    loss_fn = nn.NLLLoss()

    meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)

    models = {'model':model}

    train_and_evaluate(models, meta_train_classes, meta_test_classes, task_type, meta_optimizer, loss_fn, args)
