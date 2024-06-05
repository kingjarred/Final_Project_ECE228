'''
===========
Main Script
===========
''' 

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from rnnObject import rnnApprox, train_rnn, approximate_function
from cgSolver import cgSolver

# define the ill-conditioned pdes
def wave_equation(x): return np.sin(10 * x)
def heat_equation(x): return np.exp(-x**2 / 0.05)
def poissons_equation(x): return x**6 - x**2

# generate data for training the rnn
x_train = np.linspace(0, 7, 10000)
wave_train = wave_equation(x_train)
heat_train = heat_equation(x_train)
poissons_train = poissons_equation(x_train)
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1).unsqueeze(1)
wave_train_tensor = torch.tensor(wave_train, dtype=torch.float32).unsqueeze(-1)
heat_train_tensor = torch.tensor(heat_train, dtype=torch.float32).unsqueeze(-1)
poissons_train_tensor = torch.tensor(poissons_train, dtype=torch.float32).unsqueeze(-1)

# initialize and train the rnn
rnn_model = rnnApprox()
criterion, optimizer = nn.MSELoss(), torch.optim.Adam(rnn_model.parameters(), lr=0.01)
wave_losses = train_rnn(rnn_model, criterion, optimizer, x_train_tensor, wave_train_tensor)
heat_losses = train_rnn(rnn_model, criterion, optimizer, x_train_tensor, heat_train_tensor)
poissons_losses = train_rnn(rnn_model, criterion, optimizer, x_train_tensor, poissons_train_tensor)

# use rnn to approximate the preconditioning function
wave_approx = np.array([approximate_function(np.array([x]), rnn_model) for x in x_train])
heat_approx = np.array([approximate_function(np.array([x]), rnn_model) for x in x_train])
poissons_approx = np.array([approximate_function(np.array([x]), rnn_model) for x in x_train])

# construct very ill-conditioned matrices for the pdes
A_wave = np.diag(np.linspace(1, 1e-4, 10000))
A_heat = np.diag(np.linspace(1, 1e-4, 10000))
A_poissons = np.diag(np.linspace(1, 1e-4, 10000))

# solve using conjugate gradient
wave_solution = cgSolver(A_wave, wave_approx)
heat_solution = cgSolver(A_heat, heat_approx)
poissons_solution = cgSolver(A_poissons, poissons_approx)

# plot the results, limiting the x-axis to 6
equations = ['wave equation', 'heat equation', 'poissons equation']
exact_solutions = [wave_train, heat_train, poissons_train]
cg_solutions = [wave_solution, heat_solution, poissons_solution]

plt.figure(figsize=(15, 5))
for i, (title, exact, cg) in enumerate(zip(equations, exact_solutions, cg_solutions)):
    plt.subplot(1, 3, i + 1)
    plt.plot(x_train, exact, label='exact')
    plt.plot(x_train, cg, 'r--', label='cg')
    plt.xlim(0, 6)
    plt.xlabel('spatial variable (x)')
    plt.ylabel('solution value (y)')
    plt.legend()
    plt.title(title)
plt.show()

# plot mse losses
epochs = range(1, 1001)
plt.figure(figsize=(15, 5))
for i, (losses, title) in enumerate(zip([wave_losses, heat_losses, poissons_losses], equations)):
    plt.subplot(1, 3, i + 1)
    plt.plot(epochs, losses, label=f'{title} mse loss')
    plt.xlabel('epoch')
    plt.ylabel('mse loss')
    plt.legend()
    plt.title(f'{title} training loss')
plt.show()
