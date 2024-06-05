# Neural Net-Based Preconditioners for PDE Solvers
### ECE 228, UC San Diego, Spring 2024
### J. Sampayan, Oren Yang, Tony Guan
***
**Motivation**: To use learning preconditioners of different neural network architectures to solve a linear ill-conditioned partial differential equations (PDE) that is then passed into a conjugate gradient (CG) solver, to be solved numerically. 

**Background**: This repo uses a graphic, recurring, and convolutional neural network architecture for 3 different preconditioners. The main PDE solver is a classical numerical matrix solver, a conjugate gradient PDE solver. The methods were tested on 3 ill-conditioned PDE's, the heat, wave, and Poisson equation. 

**Contents**:
  * ***PDE_GNN.ipynb*** - Is the Graphics Neural Net (GNN) preconditioner file authored by **T. Guan**.
  * ***PDE_RNN*** - Is the Recurring Neural Net (RNN) preconditioner file authored by **J. Sampayan**.
