# Python Computational Modelling
This repo contains all the Python computational modelling project that I have done for my master's degree in Theoretical Physics.

### Second year: Computational Physics 201

- Fresnel Diffraction from an Aperture - Simpson's Rule: [[Code]](https://github.com/dazzabaijan/py_comp_model/blob/master/2nd_year/ex_2/dn16018_ex2_code.py) | [[Report]](https://github.com/dazzabaijan/py_comp_model/blob/master/2nd_year/ex_2/dn16018_ex2_report.pdf)

  The Simpson's rule is used to first computationally solve 1-D integrals in order to generate Fresnel and Fraunhofer diffraction patterns. The same rule was then used to solve 2-D integrals in order to generate 2-D diffraction patterns through apertures of different shapes.\
  Library stack: argparse, numpy, math, matplotlib, time

- Free-fall of Felix Baumgartner with Fixed or Varying Drag: [[Code]](https://github.com/dazzabaijan/py_comp_model/blob/master/2nd_year/ex_3/dn16018_ex3_code.py) | [[Report]](https://github.com/dazzabaijan/py_comp_model/blob/master/2nd_year/ex_3/dn16018_ex3_report.pdf)

  Numerical simulation using the modified Euler method to solve ordinary differential equations for a free-fall problem with varying drag at different altitudes and compared against its analytical predictions.\
  Library stack: numpy, matplotlib
  
- Calculation of Basic Rocket Orbits: [[Code]](https://github.com/dazzabaijan/py_comp_model/blob/master/2nd_year/ex_4/dn16018_ex4_code.py) | [[Report]](https://github.com/dazzabaijan/py_comp_model/blob/master/2nd_year/ex_4/dn16018_ex4_report.pdf)

  Simulated the orbit of a moving body bounded by two gravitational potentials by solving general first order ordinary differential equations using the 4-th order Runge-Kutte method.\
  Library stack: math, matplotlib

### Third year: Computational Physics 301

- Solving Partial Differential Equations with Finite Difference Methods: [[Code]](https://github.com/dazzabaijan/py_comp_model/blob/master/3rd_year/ex_2/dn16018_ex2_code.py) | [[Report]](https://github.com/dazzabaijan/py_comp_model/blob/master/3rd_year/ex_2/dn16018_ex2_report.pdf)

  The Gauss-Seidel method is first used to solve for the Laplace equation of a line charge in a partially grounded box. Secondly, both the Gauss-Seidel and Jacobi methods are used to solve for a parallel plate capacitor. A convergence condition is chosen and the discussion of the sensitivity of the solutions to the choice of convergence condition and grid density is also provided. Finally, the heat equation is solved using the backward Euler method for three scenarios including a test case. The solution of the known test case is then compared against its analytical solution.\
  Library stack: numpy, matplotlib, scipy
  
- Random Numbers and Monte-Carlo Methods: [[Code]](https://github.com/dazzabaijan/py_comp_model/blob/master/3rd_year/ex_3/dn16018_ex3_code.txt) | [[Report]](https://github.com/dazzabaijan/py_comp_model/blob/master/3rd_year/ex_3/dn16018_ex4_report.pdf)\
  Library stack: numpy, matplotlib, scipy
  This project has three parts:
  - Random Angles Generation\
    Investigation random angles(number) generation using both the inverse transform sampling and acceptance-rejection methods.
  - Nuclei Decay and Gamma Ray Detection\
    A Monte-Carlo based simulation of gamma ray detection and trajectories due particle injections from the decay of a distant nuclei.
  - Statistical Methods and Hypothesis Testing\
    Pseudo-experiments(Toy Monte-Carlo) are generated with random variables to model the Gaussian uncertainty on the background prediction and the Poisson variation in the background and signal production of a collider experiment.
    
### Final year: Advanced Computational Physics

- Computational Time Optimisation of the Ising Model: [[Code]](https://github.com/dazzabaijan/py_comp_model/blob/master/final_year/ising_mpi_method1.py) | [[Report]](https://github.com/dazzabaijan/py_comp_model/blob/master/final_year/ACP_report.pdf)

  Simulated the 2D Ferromagnetic Ising Model using the Markov Chain Monte-Carlo(MCMC) Metropolis-Hastings algorithm. At the heart of the algorithm it uses the Importance Sampling  method to sample the "important" states from the states of a system according to the equilibrated Boltzmann probability distribution rather than sample them with equal probability. The observables of the simulation - energy, magnetisation, specific heat and susceptibility per spin - have been presented as a function of temperatures. The discussion on ensuring conditions such as detailed balance and ergodicity of the simulation is presented. The Numba Python library is used to optimise and accelerate the simulation. Moreover, two similar methods of domain decomposition are presented and performed using the mpi4py Python library. Finally, the computational speedup of these methods with respect to different parameters e.g. lattice dimension is presented and discuessed.


    
