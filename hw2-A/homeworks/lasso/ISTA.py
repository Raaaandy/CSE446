from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(
        X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
    
    """

    new_bias = bias - 2 * eta * np.sum(np.matmul(X, weight) + bias - y)
    # (X @ weight + bias - y) -> (n, ) array, means error of each observation [x_i^T w+b-y_i];
    new_weight = weight - 2 * eta * (X.T @ (X @ weight + bias - y))
    condition_less = new_weight < (-2 * eta * _lambda)
    condition_more = new_weight > 2 * eta * _lambda
    condition_zero = np.logical_and(new_weight >= (-2 * eta * _lambda), new_weight <= 2 * eta * _lambda)
    new_weight[condition_less] += 2 * eta * _lambda
    new_weight[condition_more] -= 2 * eta * _lambda
    new_weight[condition_zero] = 0

    return new_weight, new_bias


@problem.tag("hw2-A")
def loss(
        X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized SSE loss.

    Args:
        X (np.ndfloatarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """

    loss_sse = np.sum(np.square(y - (X @ weight + bias)))
    norm_L1 = _lambda * np.sum(np.abs(weight))
    return loss_sse + norm_L1


@problem.tag("hw2-A", start_line=5)
def train(
        X: np.ndarray,
        y: np.ndarray,
        _lambda: float = 0.01,
        eta: float = 0.00001,
        convergence_delta: float = 1e-4,
        start_weight: np.ndarray = None,
        start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (float, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
        start_bias = 0

    old_w: Optional[np.ndarray] = np.copy(start_weight)
    old_b: Optional[np.ndarray] = start_bias
    new_b = start_bias
    new_w = np.copy(start_weight)

    while True:
        old_b = new_b
        old_w = np.copy(new_w)
        new_w, current_b = step(X, y, weight=new_w, bias=new_b, _lambda=_lambda, eta=eta)
        if convergence_criterion(new_w, old_w, new_b, old_b, convergence_delta):
            break

    return new_w, current_b


@problem.tag("hw2-A")
def convergence_criterion(
        weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight and bias has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compare it to convergence delta.
    It should also calculate the maximum absolute change between the bias and old_b, and compare it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of gradient descent.
        old_w (np.ndarray): Weight from previous iteration of gradient descent.
        bias (float): Bias from current iteration of gradient descent.
        old_b (float): Bias from previous iteration of gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight and bias has not converged yet. True otherwise.
    """

    return (np.max(np.abs(weight - old_w)) < convergence_delta) or (np.abs(bias - old_b) < convergence_delta)


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    n = 500
    d = 1000
    k = 100
    # sigma = 1
    w = np.zeros((d,))
    w[:100] = (np.arange(k) + 1) / k
    epsilon = np.random.normal(0, 1, n)
    X = np.random.normal(0, 1, size=(n, d))
    y = np.matmul(X, w) + epsilon

    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    X = (X - x_mean) / x_std

    lambda_choice = np.dot(y.T - np.mean(y), X)
    max_lambda = np.max(np.abs(lambda_choice)) * 2
    ratio = 2

    current_lambda = max_lambda
    lambda_train = []
    weights = np.zeros((d,))
    nonzero_count = []
    FDR = []
    TPR = []
    while np.count_nonzero(weights) <= (d - 10):
        lambda_train.append(current_lambda)
        print("We are using lambda = ", current_lambda)
        weights, bias = train(X, y, _lambda=current_lambda)
        nonzero_count.append(np.count_nonzero(weights))
        FDR.append(np.count_nonzero(weights[k:d]) / (d ))
        TPR.append(np.count_nonzero(weights[:k]) / k)
        print("non-zero count: ", np.count_nonzero(weights))
        current_lambda = current_lambda / ratio



    plt.figure(1)
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Non-zero counts')
    plt.title('A5-a Lambda vs non-zero element number')
    plt.plot(lambda_train, nonzero_count)
    plt.figure(2)
    plt.title('A5-b FDR vs TPR')
    plt.xlabel('FDR')
    plt.ylabel('TPR')
    plt.plot(FDR, TPR)
    plt.show()

    norm = plt.Normalize(np.min(lambda_train), np.max(lambda_train))
    sc = plt.scatter(FDR, TPR, c=lambda_train, norm=norm)
    cbar = plt.colorbar(sc)
    cbar.set_label('Lambda')
    plt.xlabel('FDRs')
    plt.ylabel('TPRs')
    plt.title('FDRs vs TPRs with Lambda as Color')
    plt.show()


if __name__ == "__main__":
    main()
