"""Bucle de entrenamiento y evaluacion."""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nn import functional as F
import tqdm

import src.data.loaders as loaders

from src.utils.tracking_demo import registrar

OPTIMIZADORES = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MNIST = {
    "epochs": 10,
    "train_losses": [],
    "train_accuracies": [],
    "test_losses": [],
    "test_accuracies": []
}

def train_mnist(model, train_loader, optimizer, device) -> nn.Module:
    # Definimos la localización del dataaset
    input_path = 'MNIST/raw/'
    training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte')
    training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte')
    test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte')
    test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte')
    
    mnist_dataloader = loaders.MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    mnist_dataloader.load_data()

    mnist_train, mnist_test = mnist_dataloader.get_datasets() # van a ser nuestros datasets de entrenamiento y testing!
    train_loader = DataLoader(mnist_train, batch_size=32)
    test_loader = DataLoader(mnist_test, batch_size=32)

    model.train()
    correct = 0
    total = 0
    total_loss = 0 # acumulamos la pérdida/loss para la época/epoch
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0) # Acumulamos weighted loss

        # Calculamos accuracy de entrenamiento
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)

        pbar.set_postfix(loss=f'{loss.item():.4f}')

    avg_loss = total_loss / total # Calculamos loss medio para la época/epoch
    acc = 100. * correct / total
    print(f'Training Accuracy: {acc:.2f}%')
    return avg_loss, acc

def test_model(model, test_loader, device):
    model.eval() # evaluamos el modelo
    test_loss = 0
    correct = 0
    with torch.no_grad(): # no queremos realizar el descenso de gradientes/gradient descent durante la evaluación!
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # obtenemos la perdida/loss de test
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset) #normalizamos perdida/loss de test
    acc = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.0f}%)\n')
    return test_loss, acc


def train_cifar_10():
    return

def test_mnist():
    return

def test_cifar_10():
    return
