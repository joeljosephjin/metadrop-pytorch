import numpy as np
from torch import tensor
import argparse


class Data:
  def __init__(self, args):
    self.N = 20 # total num instances per class
    self.K_mtr = 4800 # total num meta_train classes
    self.K_mte = 1692 # total num meta_test classes

    x_mtr, x_mte = np.load('./data/omniglot/train.npy'), np.load('./data/omniglot/test.npy')
    self.x_mtr, self.x_mte = np.reshape(x_mtr, [4800,20,28*28*1]), np.reshape(x_mte, [1692,20,28*28*1])

  def generate_episode(self, args, meta_training=True, n_episodes=1, classes=None):
    generate_label = lambda way, n_samp: np.repeat(np.eye(way), n_samp, axis=0)
    n_way, n_shot, n_query = args.num_classes, args.num_samples, args.num_query
    (K, x) = (self.K_mtr, self.x_mtr) if meta_training else (self.K_mte, self.x_mte)

    xtr, ytr, xte, yte = [], [], [], []
    for t in range(n_episodes):
      # sample WAY classes
      if classes is None:
        classes = np.random.choice(range(K), size=n_way, replace=False)

      xtr_t, xte_t = [], []
      for k in list(classes):
        # sample SHOT and QUERY instances
        idx = np.random.choice(range(self.N), size=n_shot+n_query, replace=False)
        x_k = x[k][idx]
        xtr_t.append(x_k[:n_shot])
        xte_t.append(x_k[n_shot:])

      xtr.append(np.concatenate(xtr_t, 0))
      xte.append(np.concatenate(xte_t, 0))
      ytr.append(generate_label(n_way, n_shot))
      yte.append(generate_label(n_way, n_query))

    xtr, ytr = tensor(np.stack(xtr, 0)), tensor(np.stack(ytr, 0))
    xte, yte = tensor(np.stack(xte, 0)), tensor(np.stack(yte, 0))
    return [xtr, ytr, xte, yte]


def parser_fn():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',default='data/Omniglot',help="Directory containing the dataset")
    parser.add_argument('--model_dir',default='experiments/base_model',help="Directory containing params.json")
    parser.add_argument('--restore_file',default=None,help="Optional, init model weight file")  # 'best' or 'train'
    parser.add_argument('--seed',default=1)
    parser.add_argument('--dataset',default="Omniglot")
    parser.add_argument('--meta_lr',default=1e-3, type=float)
    parser.add_argument('--task_lr',default=1e-1, type=float)
    parser.add_argument('--num_episodes',default=9, type=int)
    parser.add_argument('--num_classes',default=5, type=int)
    parser.add_argument('--num_samples',default=1, type=int)
    parser.add_argument('--num_query',default=10, type=int)
    parser.add_argument('--num_steps',default=100, type=int)
    parser.add_argument('--num_inner_tasks',default=8, type=int)
    parser.add_argument('--num_train_updates',default=1, type=int)
    parser.add_argument('--num_eval_updates',default=1, type=int)
    parser.add_argument('--save_summary_steps',default=100, type=int)
    parser.add_argument('--num_workers',default=1, type=int)

    return parser


def accuracy(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)