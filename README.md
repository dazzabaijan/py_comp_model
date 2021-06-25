# Python Computational Modelling
This repo contains all the computational modelling project that I have done for my master's degree in Theoretical Physics.

### Second year: Computational Physics 201

- Fresnel Diffraction from an Aperture - Simpson's Rule: [[Code]](https://github.com/dazzabaijan/py_comp_model/blob/master/2nd_year/ex_2/dn16018_ex2_code.py) | [[Report]](https://github.com/dazzabaijan/py_comp_model/blob/master/2nd_year/ex_2/dn16018_ex2_report.pdf)

  The Simpson's rule is used to first computationally solve 1-D integrals in order to generate Fresnel and Fraunhofer diffraction patterns. The same rule was then used to solve 2-D integrals in order to generate 2-D diffraction patterns through apertures of different shapes.\
  Library stack: argparse, numpy, math, matplotlib, time

- Free-fall of Felix Baumgartner with Fixed or Varying Drag: [[Code]](https://github.com/dazzabaijan/py_comp_model/blob/master/2nd_year/ex_3/dn16018_ex3_code.py) | [[Report]](https://github.com/dazzabaijan/py_comp_model/blob/master/2nd_year/ex_3/dn16018_ex3_report.pdf)

  Numerical simulation using the modified Euler method to solve ordinary differential equations for a free-fall problem with varying drag at different altitudes and compared against its analytical predictions.\
  Library stack: numpy, matplotlib
  
- Calculation of Basic Rocket Orbits: [[Code]](https://github.com/dazzabaijan/py_comp_model/blob/master/2nd_year/ex_4/dn16018_ex4_code.py) | [[Report]](https://github.com/dazzabaijan/py_comp_model/blob/master/2nd_year/ex_4/dn16018_ex4_report.pdf)

  Simulated the orbit of a moving body bounded by a gravitational potential by solving general first order ordinary differential equations using the 4-th order Runge-Kutte method.\
  Library stack: math, matplotlib

### Third year: Computational Physics 301

- Solving Partial Differential Equations with Finite Difference Methods: [[Code]](https://github.com/dazzabaijan/py_comp_model/blob/master/3rd_year/ex_2/dn16018_ex2_code.py) | [[Report]](https://github.com/dazzabaijan/py_comp_model/blob/master/3rd_year/ex_2/dn16018_ex2_report.pdf)

  The Gauss-Seidel method is first used to solve for the Laplace equation of a line charge in a partially grounded box. Secondly, both the Gauss-Seidel and Jacobi methods are used to solve for a parallel plate capacitor. A convergence condition is chosen and the discussion of the sensitivity of the solutions to the choice of convergence condition and grid density is also provided. Finally, the heat equation is solved using the backward Euler method for three scenarios including a test case. The solution of the known test case is then compared against its analytical solution. 
