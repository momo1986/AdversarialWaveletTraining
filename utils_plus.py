#import apex.amp as amp
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.autograd import Variable


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform_nonorm = transforms.Compose([
        transforms.ToTensor()
    ])
    num_workers = 2
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
    test_dataset_nonorm = datasets.CIFAR10(
        dir_, train=False, transform=test_transform_nonorm, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    test_loader_nonorm = torch.utils.data.DataLoader(
        dataset=test_dataset_nonorm,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader, test_loader_nonorm

def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    
    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, use_CWloss=False):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            if use_CWloss:
                loss = CW_loss(output, y)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts, step=2, use_CWloss=False):
    epsilon = (8 / 255.) / std
    alpha = (step / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, use_CWloss=use_CWloss)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss/n, pgd_acc/n

def evaluate_mim(test_loader, model, num_steps=20, decay_factor=1.0):
    test_loss = 0
    test_acc = 0
    n = 0
    print(std)
    epsilon = (8.0 / 255.0)/std
    step_size = (2.0 / 255.0)/std
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            #output = model(X)
            X_pgd = Variable(X.data, requires_grad=True)
            delta = torch.zeros_like(X).cuda()
            for i in range(len(epsilon)):
                delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True
            X_pgd = Variable(X_pgd.data + delta, requires_grad=True)
            previous_grad = torch.zeros_like(X.data)
            for _ in range(num_steps):
                opt = torch.optim.SGD([X_pgd], lr=1e-3)
                opt.zero_grad()

                with torch.enable_grad():
                    loss = torch.nn.CrossEntropyLoss()(model(X_pgd),y)
                loss.backward()
                grad = X_pgd.grad.data / torch.mean(torch.abs(X_pgd.grad.data), [1,2,3], keepdim=True)
                previous_grad = decay_factor * previous_grad + grad
                X_pgd = Variable(X_pgd.data + step_size * previous_grad.sign(), requires_grad=True)
                eta = clamp(X_pgd.data - X.data, -epsilon, epsilon)
                X_pgd = Variable(X.data + eta, requires_grad=True)
                X_pgd = Variable(torch.clamp(X_pgd, -1.0, 1.0), requires_grad=True)
            test_loss += loss.item() * y.size(0)
            test_acc += (model(X_pgd).max(1)[1] == y).float().sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n


def evaluate_fgsm(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    print(std)
    epsilon = (8.0 / 255.0)/std
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            #output = model(X)
 
            X_fgsm = Variable(X.data, requires_grad=True)
            opt = torch.optim.SGD([X_fgsm], lr=1e-3)
            opt.zero_grad()
            with torch.enable_grad():
                #loss = F.cross_entropy(model(X_fgsm), y)
                loss = torch.nn.CrossEntropyLoss()(model(X_fgsm),y)
            loss.backward()
            X_fgsm = Variable(torch.clamp(X_fgsm.data + epsilon * X_fgsm.grad.data.sign(), -1.0, 1.0), requires_grad=True)
            test_loss += loss.item() * y.size(0)
            test_acc += (model(X_fgsm).max(1)[1] == y).float().sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n

def evaluate_new_fgsm(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    epsilon = (8 / 255.)/std
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            delta = torch.zeros_like(X).cuda()
            for i in range(len(epsilon)):
                delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True
 
            output = model(X + delta)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            #delta.grad.zero_()
    return test_loss/n, test_acc/n

def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n
