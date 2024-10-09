import torch, argparse
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import NeuralNetwork
import os, sys

def parser():

    args_parser = argparse.ArgumentParser(description='Sample code to train on paperspace gradient. Modified from https://github.com/pytorch/tutorials/blob/main/beginner_source/basics/quickstart_tutorial.py')

    args_parser.add_argument('--evaluation', action='store_true', help='Evaluate given model.')
    args_parser.add_argument('--eval_model_path', type=str, help='Path to the model to be used for evaluation.')
    args_parser.add_argument('--save_path', type=str, default='exps/exp1', help='Path to save the trained model.')

    args = args_parser.parse_args()

    return args

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':

    args = parser()

    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=4)

    # Get device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    if args.evaluation:
        model.load_state_dict(torch.load(args.eval_model_path, weights_only=True))
        test(test_dataloader, model, loss_fn)
        sys.exit()

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    torch.save(model.state_dict(), f"{args.save_path}/model.pth")
    print(f"Saved PyTorch Model State to {args.save_path}/model.pth")