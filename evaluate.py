import copy

import torch
import numpy as np
from collections import OrderedDict
from data.dataloader import fetch_dataloaders


def evaluate(models, loss_fn, meta_classes, task_lr, task_type, args,
             split):
    """
    Evaluate the model on `num_steps` batches.
    
    Args:
        model: (MetaLearner) a meta-learner that is trained on MAML
        loss_fn: a loss function
        meta_classes: (list) a list of classes to be evaluated in meta-training or meta-testing
        task_lr: (float) a task-specific learning rate
        task_type: (subclass of FewShotTask) a type for generating tasks
        metrics: (dict) a dictionary of functions that compute a metric using 
                 the output and labels of each batch
        params: (Params) hyperparameters
        split: (string) 'train' if evaluate on 'meta-training' and 
                        'test' if evaluate on 'meta-testing' TODO 'meta-validating'
    """
    # summary for current eval loop
    losses = []
    accs = []

    model = models['model']
    
    if args.phi:
        phi_net = models['phi_net']

    # compute metrics over the dataset
    for episode in range(args.num_steps):
        # Make a single task
        task = task_type(meta_classes, args.num_classes, args.num_samples, args.num_query)
        dataloaders = fetch_dataloaders(['train', 'test'], task)
        dl_sup, dl_que = dataloaders['train'], dataloaders['test']
        X_sup, Y_sup = dl_sup.__iter__().next()
        X_que, Y_que = dl_que.__iter__().next()

        X_sup, Y_sup = X_sup.to(args.device), Y_sup.to(args.device)
        X_que, Y_que = X_que.to(args.device), Y_que.to(args.device)

        # Direct optimization
        # net_clone = copy.deepcopy(model)
        adapted_params = model.cloned_state_dict()
        if args.phi:
            phi_adapted_params = phi_net.cloned_state_dict()

        # optim = torch.optim.SGD(net_clone.parameters(), lr=task_lr)
        for _ in range(args.num_eval_updates):
            if args.phi:
                Y_sup_hat = model(X_sup, adapted_params, phi_adapted_params)
            else:
                Y_sup_hat = model(X_sup, adapted_params)

            loss = loss_fn(Y_sup_hat, Y_sup)

            grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)
            for (key, val), grad in zip(adapted_params.items(), grads):
                adapted_params[key] = val - task_lr * grad

        # Y_que_hat = model(X_que)

        if args.phi:
            Y_que_hat = model(X_que, adapted_params, phi_adapted_params)
        else:
            Y_que_hat = model(X_que, adapted_params)

        loss = loss_fn(Y_que_hat, Y_que)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        Y_que_hat, Y_que = Y_que_hat.data.cpu().numpy(), Y_que.data.cpu().numpy()

        losses.append(loss.item())
        accs.append(accuracy(Y_que_hat, Y_que))

    return np.mean(losses), np.mean(accs)

def accuracy(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)