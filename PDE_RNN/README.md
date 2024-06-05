
### Instructions
### J. Sampayan
***
1. rnnObject.py contains the recurring neural network architecture as an object. This script is an object that is called to precondition a linear ill-conditioned PDE by function approximation.
2. cgSolver.py contains the simple CG PDE solver. This is the main PDE solver that accuratately and numerically solves the pde in space and time. 
3. main.py is a batch script that runs the defines ill-conditioned poissons, wave, and heat equations and passes it through the RNN that is then solved in the CG solver. 
4. Run main.py to precondition the PDEs and solve them using the conjugate gradient solver.
