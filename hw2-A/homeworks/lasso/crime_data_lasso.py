if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem
from ISTA import loss


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets

    df_train, df_test = load_dataset("crime")  # type: ignore
    train_y = df_train['ViolentCrimesPerPop'].values
    test_y = df_test['ViolentCrimesPerPop'].values
    train_X = df_train.drop('ViolentCrimesPerPop', axis=1).values
    test_X = df_test.drop('ViolentCrimesPerPop', axis=1).values

    lambda_choice = np.dot(train_y.T - np.mean(train_y), train_X)
    max_lambda = np.max(np.abs(lambda_choice)) * 2
    current_lambda = max_lambda
    lambda_train = []

    ratio = 2
    nonzero_count = []
    agePct12t29 = df_train.drop('ViolentCrimesPerPop', axis=1).columns.get_loc("agePct12t29")
    pctWSocSec = df_train.drop('ViolentCrimesPerPop', axis=1).columns.get_loc("pctWSocSec")
    pctUrban = df_train.drop('ViolentCrimesPerPop', axis=1).columns.get_loc("pctUrban")
    agePct65up = df_train.drop('ViolentCrimesPerPop', axis=1).columns.get_loc("agePct65up")
    householdsize = df_train.drop('ViolentCrimesPerPop', axis=1).columns.get_loc("householdsize")
    train_loss = []
    test_loss = []
    i1 = []
    i2 = []
    i3 = []
    i4 = []
    i5 = []
    prev_w = np.zeros((95,))
    bias = 0
    B = []
    while current_lambda >= 0.01:
        weights, bias = train(train_X, train_y, current_lambda, start_weight=prev_w, start_bias=bias)
        train_loss.append(np.sum((train_X @ weights + bias - train_y)**2)/ train_X.shape[0])
        test_loss.append(np.sum((test_X @ weights + bias - test_y)**2)/ test_X.shape[0])

        prev_w = np.copy(weights)
        B.append(bias)
        i1.append(weights[agePct12t29])
        i2.append(weights[pctWSocSec])
        i3.append(weights[pctUrban])
        i4.append(weights[agePct65up])
        i5.append(weights[householdsize])
        nonzero_count.append(np.count_nonzero(weights))
        print("non-zero count: ", np.count_nonzero(weights))
        print("We are using lambda = ", current_lambda)
        lambda_train.append(current_lambda)
        current_lambda = current_lambda / ratio
    plt.figure(1)
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Non-zero counts')
    plt.title('A6-c Lambda vs non-zero element number')
    plt.plot(lambda_train, nonzero_count)

    plt.figure(2)
    plt.xscale('log')
    plt.plot(lambda_train, i1, label='agePct12t29')
    plt.plot(lambda_train, i2, label='pctWSocSec')
    plt.plot(lambda_train, i3, label='pctUrban')
    plt.plot(lambda_train, i4, label='agePct65up')
    plt.plot(lambda_train, i5, label='householdsize')
    plt.xlabel('Lambda')
    plt.ylabel('Coefficient Value')
    plt.title('6d: Regularization Paths')
    plt.legend()

    plt.figure(3)

    plt.xscale('log')
    plt.plot(lambda_train, train_loss, label='train loss')
    plt.plot(lambda_train, test_loss, label='test loss')
    plt.xlabel('Lambda')
    plt.ylabel('mean squared error')
    plt.title("a6-e")
    plt.legend()
    plt.show()

    weights, bias = train(train_X, train_y, 30, start_weight=np.zeros((95,)), start_bias=0)
    print(np.argmax(weights), weights[np.argmax(weights)], df_train.columns[np.argmax(weights)])
    print(np.argmin(weights), weights[np.argmin(weights)], df_train.columns[np.argmin(weights)])

if __name__ == "__main__":
    main()
