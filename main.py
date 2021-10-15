import argparse
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from model import MetaLearner
from model import Net
from model import phiNet
from data.dataloader import split_omniglot_characters
from data.dataloader import load_imagenet_images
from data.dataloader import OmniglotTask
from data.dataloader import ImageNetTask
from data.dataloader import fetch_dataloaders
from evaluate import evaluate
from evaluate import accuracy

import numpy as np

from time import time

from data_haebom.data import Data

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',default='data/Omniglot',help="Directory containing the dataset")
parser.add_argument('--model_dir',default='experiments/base_model',help="Directory containing params.json")
parser.add_argument('--restore_file',default=None,help="Optional, init model weight file")  # 'best' or 'train'
parser.add_argument('--seed',default=1)
parser.add_argument('--dataset',default="Omniglot")
parser.add_argument('--meta_lr',default=1e-3, type=float)
parser.add_argument('--task_lr',default=1e-1, type=float)
parser.add_argument('--num_episodes',default=10000, type=int)
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
    if args.wandb:
        wandb.init(project='metadrop-pytorch', entity='joeljosephjin', config=vars(args))

    model = models['model']
    if args.phi:
        phi_net = models['phi_net']
        phi_optimizer = models['phi_optimizer']

    # params information
    num_classes = args.num_classes
    num_samples = args.num_samples
    num_query = args.num_query
    num_inner_tasks = args.num_inner_tasks
    task_lr = args.task_lr
    start_time = 0

    data = Data(args)

    for episode in range(args.num_episodes):
        # Run inner loops to get adapted parameters (theta_t`)
        data_episode = data.generate_episode(args, meta_training=True, n_episodes=num_inner_tasks)
        meta_loss = 0
        accs = []
        xtr, ytr, xte, yte = data_episode
        for n_task in range(num_inner_tasks):
            # task = task_type(meta_train_classes, num_classes, num_samples, num_query)
            # dataloaders = fetch_dataloaders(['train', 'test', 'meta'], task)
            # dl_sup = dataloaders['train']
            # X_sup, Y_sup = dl_sup.__iter__().next()

            xtri, ytri, xtei, ytei = xtr[n_task], ytr[n_task], xte[n_task], yte[n_task]
            X_sup, Y_sup = xtri, ytri
            X_sup, Y_sup = X_sup.reshape([-1, 1, 28, 28]).to(args.device), Y_sup.to(args.device) # [5, 784]

            adapted_params = model.cloned_state_dict()
            if args.phi:
                phi_adapted_params = phi_net.cloned_state_dict()

            for _ in range(0, args.num_train_updates):
                if args.phi:
                    Y_sup_hat = model(X_sup, adapted_params, phi_adapted_params)
                else:
                    Y_sup_hat = model(X_sup, adapted_params)
                loss = loss_fn(Y_sup_hat, torch.argmax(Y_sup, dim=1))

                grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)
                for (key, val), grad in zip(adapted_params.items(), grads):
                    adapted_params[key] = val - task_lr * grad

            # dl_meta = dataloaders['meta']
            # X_meta, Y_meta = dl_meta.__iter__().next()
            # X_meta, Y_meta = X_meta.to(args.device), Y_meta.to(args.device)

            X_meta, Y_meta = xtei, ytei
            X_meta, Y_meta = X_meta.reshape([-1, 1, 28, 28]).to(args.device), Y_meta.to(args.device) # [5, 784]

            if args.phi:
                Y_meta_hat = model(X_meta, adapted_params, phi_adapted_params)
            else:
                Y_meta_hat = model(X_meta, adapted_params)

            accs.append(accuracy(Y_meta_hat.data.cpu().numpy(), Y_meta.data.cpu().numpy()))
            loss_t = loss_fn(Y_meta_hat, torch.argmax(Y_meta, dim=1))

            meta_loss += loss_t
            
        meta_loss /= float(num_inner_tasks)

        meta_optimizer.zero_grad()
        if args.phi:
            phi_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        if args.phi:
            phi_optimizer.step()

        # Evaluate model on new task
        # Evaluate on train and test dataset given a number of tasks (args.num_steps)
        if (episode + 1) % args.save_summary_steps == 0:
            train_loss, train_acc = evaluate(models, loss_fn, meta_train_classes,
                            task_lr, task_type, args,
                            'train')
            test_loss, test_acc = evaluate(models, loss_fn, meta_test_classes,
                                    task_lr, task_type, args,
                                    'test')

            if args.wandb:
                wandb.log({"episode":episode, "test_acc":test_acc, "train_acc":train_acc,\
                            "test_loss":test_loss,"train_loss":train_loss})
            print('episode: {:0.2f}, acc: {:0.2f}, test_acc: {:0.2f}, train_acc: {:0.2f}, time: {:0.2f}, test_loss: {:0.2f}, train_loss: {:0.2f}'\
                    .format(episode, np.mean(accs), test_acc, train_acc, time()-start_time, test_loss, train_loss))
            start_time = time()

if __name__ == '__main__':
    args = parser.parse_args()

    if args.wandb:
        import wandb


    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if 'Omniglot' in args.data_dir and args.dataset == 'Omniglot':
        args.in_channels = 1
        meta_train_classes, meta_test_classes = split_omniglot_characters(args.data_dir, args.seed)
        task_type = OmniglotTask
    elif ('miniImageNet' in args.data_dir or 'tieredImageNet' in args.data_dir) and args.dataset == 'ImageNet':
        args.in_channels = 3
        meta_train_classes, meta_test_classes = load_imagenet_images(args.data_dir)
        task_type = ImageNetTask
    else:
        raise ValueError("I don't know your dataset")

    model = MetaLearner(args).to(args.device)
    loss_fn = nn.NLLLoss()

    # args.phi = False

    meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)

    if args.phi:
        phi_net = phiNet(args.in_channels, args.num_classes, dataset=args.dataset).to(args.device)
        phi_optimizer = torch.optim.Adam(phi_net.parameters(), lr=args.meta_lr)
        models = {'model':model, 'phi_net':phi_net, 'phi_optimizer':phi_optimizer}
    else:
        models = {'model':model}

    train_and_evaluate(models, meta_train_classes, meta_test_classes, task_type, meta_optimizer, loss_fn, args)
