# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.d = d
        self.h = h
        self.k = k
        alpha = 1/math.sqrt(d)
        self.W0 = Parameter(torch.empty(h, d).uniform_(-alpha, alpha)) #(h, d)
        self.b0 = Parameter(torch.empty(h).uniform_(-alpha, alpha))
        self.W1 = Parameter(torch.empty(k, h).uniform_(-alpha, alpha)) #(k, h)
        self.b1 = Parameter(torch.empty(k).uniform_(-alpha, alpha))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        hidden_1 = x @ self.W0.T  + self.b0
        hidden_1_activation = relu(hidden_1)
        output = hidden_1_activation @ self.W1.T  + self.b1
        return output


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        alpha = 1/math.sqrt(d)
        self.W0 = Parameter(torch.empty(h0, d).uniform_(-alpha, alpha).T) #(d, h0)
        self.b0 = Parameter(torch.empty(h0).uniform_(-alpha, alpha))
        self.W1 = Parameter(torch.empty(h1, h0).uniform_(-alpha, alpha).T)
        self.b1 = Parameter(torch.empty(h1).uniform_(-alpha, alpha))
        self.W2 = Parameter(torch.empty(k, h1).uniform_(-alpha, alpha).T)
        self.b2 = Parameter(torch.empty(k).uniform_(-alpha, alpha))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        hidden_1 = relu(x @ self.W0 + self.b0)
        hidden_2 = relu(hidden_1 @ self.W1 + self.b1)
        output = hidden_2 @ self.W2 + self.b2
        return output


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    model.train()
    history:List[float] = []
    accuracy = 0
    epoches = 0
    while accuracy < 0.99:
        total_loss = 0
        print(epoches)
        epoches += 1
        for x, y in tqdm(train_loader):
            optimizer.zero_grad()
            y_pred = model(x)

            loss = cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        history.append(total_loss/len(train_loader))
        accuracy = accuracy_score(model, train_loader)
    return history

def accuracy_score(model, dataloader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for CrossEntropy.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is an integer representing a correct class to a corresponding observation.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to MSE accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    model.eval()
    correct_num = 0
    total_num = 0
    for x, y in dataloader:
        y_pred = model(x)
        pred_idx = torch.argmax(y_pred, 1)
        total_num += len(y)
        correct_num += (pred_idx == y).sum().item()

    accuracy = correct_num / total_num
    return accuracy

def test_loss(model, test_loader):
    total_loss = 0
    for x, y in test_loader:
        y_pred = model(x)
        loss = cross_entropy(y_pred, y)
        total_loss += loss.item()
    return total_loss/ len(test_loader), total_loss





@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    ## prepare for dataset and loader
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    train_dataset = TensorDataset(x, y)
    test_dataset = TensorDataset(x_test, y_test)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    lr = 0.001

    """
    ## F1 model
    F1_model = F1(64, 784, 10)
    history = train(F1_model, Adam(F1_model.parameters(), lr), train_loader)
    plt.figure("hw3-A5-a F1 training loss")
    plt.plot(history)
    plt.xlabel("epoches")
    plt.ylabel("loss")

    plt.show()
    avg_test_loss_F1, total_loss_F1 = test_loss(F1_model, test_loader)
    test_acc_F1=accuracy_score(F1_model, test_loader)
    total_params = sum(p.numel() for p in F1_model.parameters() if p.requires_grad)
    print(f'F1 model: lr={lr}, batch size={batch_size}, test accuracy={test_acc_F1}, average test loss={avg_test_loss_F1}, total test loss={total_loss_F1}')
    print(f'total parameters count:{total_params}')
    """


    ## F2_model
    F2_model = F2(32, 32, 784, 10)
    history_F2 = train(F2_model, Adam(F2_model.parameters(), lr), train_loader)
    plt.figure("hw3-A5-b F2 training loss")
    plt.plot(history_F2)
    plt.xlabel("epoches")
    plt.ylabel("loss")
    plt.show()
    avg_test_loss_F2, total_loss_F2 = test_loss(F2_model, test_loader)
    test_acc_F2=accuracy_score(F2_model, test_loader)
    total_params = sum(p.numel() for p in F2_model.parameters() if p.requires_grad)
    print(f'F2 model: lr={lr}, batch size={batch_size}, test accuracy={test_acc_F2}, average test loss={avg_test_loss_F2}, total test loss={total_loss_F2}')
    print(f'total parameters count:{total_params}')


if __name__ == "__main__":
    main()
