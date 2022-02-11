import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

from model import MetaLearner
from statistics import mean
from utils import Data, parser_fn, accuracy

args = parser_fn().parse_args()
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

args.in_channels = 1

model = MetaLearner(args).to(args.device)
loss_fn = nn.NLLLoss()

meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)

num_inner_tasks = args.num_inner_tasks
task_lr = args.task_lr

data = Data(args)

mean_accs = []

for episode in range(args.num_episodes):
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

    mean_accs.append(mean(accs))
    print('episode:', episode, 'accs (mean of tasks):', mean(accs))

print('All good?:', (mean_accs==[0.355, 0.3075, 0.3825, 0.47250000000000003, 0.6925, 0.4875, 0.4675, 0.5425, 0.8175]))
