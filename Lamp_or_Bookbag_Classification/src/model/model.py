import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import os
import sys
import argparse
from pathlib import Path


class model():
    def __init__(self, args):
        self.root = args.input
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        # GPU/CPU
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        # Using pretrained Resnet for training
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        self.model.fc = nn.Linear(num_ftrs, 2)
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        # Stochastic Gradient Desc as optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9)
        self.NUM_EPOCHS = 10
        self.BEST_MODEL_PATH = f"{args.output}/best_model.pth"
        self.best_accuracy = 0.0

    def transforms(self):
        # Define data transforms to perform on the dataset
        data_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return data_transform

    def load(self):
        # Load the dataset
        dataset = datasets.ImageFolder(
            root=self.root, transform=self.transforms())
        # testdata ratio (20% for test, 80% for training)
        test_split = .2
        dataset_size = len(dataset)
        test_len = int(np.floor(test_split * dataset_size))
        # Use torch randomsplit to split the dataset into train and test sets
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset, [dataset_size-test_len, test_len], generator=torch.Generator().manual_seed(42))
        # Use torch dataloader to create respective dataloaders for training and validation
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=16, shuffle=True, num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=16, shuffle=True, num_workers=4)

    def train_test(self):
        self.load()
        for epoch in range(self.NUM_EPOCHS):

            for inputs, labels in iter(self.train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            test_error_count = 0.0
            for inputs, labels in iter(self.test_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                test_error_count += float(
                    torch.sum(torch.abs(labels - outputs.argmax(1))))

            test_accuracy = 1.0 - \
                float(test_error_count) / float(len(self.test_dataset))
            print('%d: %f' % (epoch, test_accuracy))
            if test_accuracy > self.best_accuracy:
                torch.save(self.model.state_dict(), self.BEST_MODEL_PATH)
                self.best_accuracy = test_accuracy


def build_parser():
    # Use build parser for setting commandline arguments (including default values)
    outputpath = "../../data/outputs/"
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Path to root folder (inside which there are folders for each class)")
    parser.add_argument("-o", "--output", default=outputpath,
                        help="Path to output directory")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    # Grabs the current directory
    dirname = os.path.dirname(__file__)
    # Builds the path to save location
    args.output = os.path.join(dirname, args.output)
    args.input = os.path.join(dirname, args.input)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model(args).train_test()


if __name__ == "__main__":
    main()
