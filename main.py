import torch
from kan import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
def miningData():
    df=pd.read_csv('data/origindata.csv')
    df = df[['new_cases', 'new_deaths', 'total_cases', 'handwashing_facilities']]
    df = df.dropna()
    X = df[['new_deaths', 'total_cases', 'handwashing_facilities']]
    y = df[['new_cases']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float64).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float64).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float64).to(device)  # or torch.long for classification
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float64).to(device)
    dataset = {}
    dataset['train_input'] = X_train_tensor
    dataset['test_input'] = X_test_tensor
    dataset['train_label'] = y_train_tensor
    dataset['test_label'] = y_test_tensor
    return dataset
def createModel():
    grid_input_str = input(
        "Please press input for grid-sizes (Ex: 3,10,20 or 3 10 20): ")
    try:
        grid_input_str = grid_input_str.replace(',', ' ')
        grids = np.array([int(x) for x in grid_input_str.split() if x.strip()])
        print(f"Grid input: {grids}")
    except ValueError:
        print("Error: input is not available")
    except Exception as e:
        print(f"Have error about grid-size: {e}")
    while True:
        steps_str = input('Please enter a number for steps: ')
        try:
            steps = int(steps_str)
            break
        except ValueError:
            print("Invalid input. Please enter an integer for steps.")
    while True:
        k_str = input('Please enter a number for k (spline order): ')
        try:
            k = int(k_str)
            break
        except ValueError:
            print("Invalid input. Please enter an integer for k.")

    print(f"Steps: {steps}, Type: {type(steps)}")
    print(f"k: {k}, Type: {type(k)}")


train_losses = []
test_losses = []
for i in range(grids.shape[0]):
    if i == 0:
        model = KAN(width=[dataset['train_input'].shape[1], [3, 2], 1], grid=grids[i], k=k, seed=1, device=device)
    if i != 0:
        model = model.refine(grids[i])
    results = model.fit(dataset, opt="LBFGS", steps=steps)
    train_losses += results['train_loss']
    test_losses += results['test_loss']
plt.plot(train_losses)
plt.plot(test_losses)
plt.legend(['train', 'test'])
plt.ylabel('RMSE')
plt.xlabel('step')
plt.yscale('log')
n_params = 3 * grids
train_vs_G = train_losses[(steps-1)::steps]
test_vs_G = test_losses[(steps-1)::steps]
plt.plot(n_params, train_vs_G, marker="o")
plt.plot(n_params, test_vs_G, marker="o")
plt.plot(n_params, 100*n_params**(2.), ls="--", color="black")
plt.xscale('log')
plt.yscale('log')
plt.legend(['train', 'test', r'$N^{2}$'])
plt.xlabel('number of params')
plt.ylabel('RMSE')
x = torch.rand(5, 5).cuda()
miningData()