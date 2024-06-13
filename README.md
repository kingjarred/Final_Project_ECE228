### Neural Net-Based Preconditioners for PDE Solvers
 * **Authors** - J. Sampayan, Tony Guan, and Oren Yang
 * **Affiliation** - ECE 228 - Machine Learning and Physical Applications, Team 3, UC San Diego, Spring 2024
 * **Motivation** - To use learning preconditioners of different neural network architectures to solve linear ill-conditioned partial differential equations (PDE) that is then passed into a conjugate gradient (CG) PDE solver, numerically. The goal is to show that learning preconditioners can decrease computational costs for PDE solvers.
 * **Background** - This repo uses a graphic, recurring, and convolutional neural network architecture-based precondition (3 total preconditioners). The PDE solver used is a classical numerical matrix solver, a simple conjugate gradient PDE solver. The methods were tested on 3 ill-conditioned PDEs, the heat, wave, and Poisson equation. Those 3 PDE's with very ill-conditioned matrices where each passed into the 3 difference preconditioners then into the CG solver. The methods were validated against linear classical preconditioners.

***

### Contents
  * ***PDE_RNN*** - Is the Recurring Neural Net (RNN) preconditioner folder authored by **J. Sampayan**.
  * ***PDE_GNN*** - Is the Graphics Neural Net (GNN) preconditioner folder authored by **T. Guan**.
  * ***PDE_CNN*** - Is the Convolutional Neural Net (CNN) preconditioner folder authored by **O. Yang**.

***

### Citations
 1. Li, Y., Chen, H., & Sun, L. (2023). Learning Preconditioners for Conjugate Gradient PDE Solvers. Proceedings of the 40th International Conference on Machine Learning. PMLR 202. Retrieved from [https://sites.google.com/view/neuralPCG](https://sites.google.com/view/neuralPCG).
 2. Belbute-Peres, F., Economon, T., & Kolter, Z. (2023). Neural Network Preconditioners for Solving the Dirac Equation in Lattice Gauge Theory. Under review at ICLR 2023. Retrieved from [https://arxiv.org/abs/2208.02728](https://arxiv.org/abs/2208.02728).
Feel free to adjust the formatting as needed for your specific README.md file.
