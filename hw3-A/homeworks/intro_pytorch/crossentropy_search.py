if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from losses import CrossEntropyLossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from .optimizers import SGDOptimizer
    from .losses import CrossEntropyLossLayer
    from .train import plot_model_guesses, train

from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


@problem.tag("hw3-A")
def crossentropy_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the CrossEntropy problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers
    NOTE: Each model should end with a Softmax layer due to CrossEntropyLossLayer requirement.

    Notes:
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.

    Args:
        dataset_train (TensorDataset): Dataset for training.
        dataset_val (TensorDataset): Dataset for validation.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    class LinearModel(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.linear = LinearLayer(input_size, output_size)
            self.softmax = SoftmaxLayer()
        
        def forward(self, inputs):
            x = self.linear(inputs)
            x = self.softmax(x)
            return x
    
    class OneHiddenLayer(nn.Module):
        def __init__(self, input_size, output_size, hidden_size, activation_func):
            super().__init__()
            self.linear0 = LinearLayer(input_size, hidden_size)
            self.activation = activation_func
            self.linear1 = LinearLayer(hidden_size, output_size)
            self.softmax = SoftmaxLayer()
            
        def forward(self, inputs):
            x = self.activation(self.linear0(inputs))
            x = self.linear1(x)
            x = self.softmax(x)
            return x

    class TwoHiddenLayer(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, activation_1, activation_2):
            super().__init__()
            self.linear_0 = LinearLayer(input_size, hidden_size)
            self.linear_1 = LinearLayer(hidden_size, hidden_size)
            self.linear_2 = LinearLayer(hidden_size, output_size)
            self.activation_1 = activation_1
            self.activation_2 = activation_2
            self.softmax = SoftmaxLayer()
        
        def forward(self, input_data):
            hidden_1_layer = self.activation_1(self.linear_0(input_data))
            hidden_2_layer = self.activation_2(self.linear_1(hidden_1_layer))
            output_layer = self.linear_2(hidden_2_layer)
            output_layer = self.softmax(output_layer)
            return output_layer


    input_sample, _ = dataset_train[0]
    input_feature_size = input_sample.shape[0]
    output_size = 2
    lr= 10 ** -4
    batch_size = (2 ** 5)
    models = {
        "Linear": LinearModel(input_feature_size, output_size),
        "Sigmoid": OneHiddenLayer(input_feature_size, output_size, 2, SigmoidLayer()),
        "ReLU": OneHiddenLayer(input_feature_size, output_size, 2, ReLULayer()),
        "SigmoidReLU": TwoHiddenLayer(input_feature_size, 2, 2, SigmoidLayer(), ReLULayer()),
        "ReLUSigmoid": TwoHiddenLayer(input_feature_size, 2, 2, ReLULayer(), SigmoidLayer())
    }
    result = {}
    for model_name, model in models.items():
        result[model_name] = {}
        train_loader = DataLoader(dataset_train, int(batch_size), shuffle=True)
        val_loader = DataLoader(dataset_val, int(batch_size))
        history = train(train_loader, model, CrossEntropyLossLayer(), SGDOptimizer(model.parameters(), lr), val_loader, epochs=200)
        print(model_name)
        result[model_name]['train'] = history['train']
        result[model_name]['val'] = history['val']
        result[model_name]["model"] = model
    return result


@problem.tag("hw3-A")
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

@problem.tag("hw3-A", start_line=7)
def main():
    """
    Main function of the Crossentropy problem.
    It should:
        1. Call crossentropy_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me Crossentropy loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y))
    dataset_val = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    dataset_test = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))

    ce_configs = crossentropy_parameter_search(dataset_train, dataset_val)
    plt.figure("hw3-A4-b-ce")
    for model_name, model_info in ce_configs.items():
        plt.plot(model_info["train"], label=f'{model_name} train loss')
        plt.plot(model_info["val"], label=f'{model_name} val loss')
    plt.legend()
    plt.xlabel("Epoches")
    plt.ylabel("Cross Entropy Loss")
    plt.title("CE plot")
    plt.show()



if __name__ == "__main__":
    main()
