# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from copy import copy
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from time import clock
import sys

"""Boundary conditions for second Physics problem"""


def hot_cold(m, l, u, F):
    """A boundary condition function for when rod is in a furnace one end and
       in an ice bath for another.

    Parameters:
    m, l, u : ndarray(s) representing the main, lower and upper diagonal
              elements of a tridiagonal matrix
    F     : a constant within a triadiagonal matrix

    Returns:
    m, l, u : The original ndarray with certain elements assigned to specific
              values according to the initial condition.
    """
    m[0], u[0] = 1, 0
    l[-1], m[-1] = 0, 1
    return m, l, u


def adiabatic(m, l, u, F):
    """A boundary condition function for when the poker is adiabatic at either
       end.

    Parameters:
    m, l, u : ndarray(s) representing the main, lower and upper diagonal
              elements of a tridiagonal matrix
    F     : a constant within a triadiagonal matrix

    Returns:
    m, l, u : The original ndarray with certain elements assigned to specific
              values according to the initial condition.
    """
    m[0], u[0] = 1, 0
    l[-1] = -2*F
    return m, l, u


def both_ice(m, l, u, F):
    """A boundary condition function for when the poker is in an ice bath at
       both ends.

    Parameters:
    m, l, u : ndarray(s) representing the main, lower and upper diagonal
              elements of a tridiagonal matrix
    F       : a constant within a triadiagonal matrix

    Returns:
    m, l, u : The original ndarray with certain elements assigned to specific
              values according to the initial condition.
    F       : Not being returned since will be taken in within another function
    """
    m[0], u[0] = 1, 0
    l[-1], m[-1] = 0, 1
    return m, l, u


def potential(x_axis, y_axis, cbar, x_label, y_label, cbar_label, int_pol,
              colorMap, xmin=None, xmax=None, ymin=None, ymax=None, plot=None):
    """A general plotter allowing for the plotting of a heatmap(primarily used
       here for potential function) which takes in relevant data about the
       graph. It also allows for the option of overlaying either a quiver or a
       streamline plot, or not!

    Parameters:
    x_axis, y_axis : The corresponding x and y axis data lists of a plot
    cbar: The heatmap list data which usually corresponds to x and y axis
    x_label, y_label, : The x, y and z label of the graph       dtype = string
    cbar_label : The colourbar label                            dtype = string
    int_pol: The colour interpolation of the heatmap            dtype = integer
    colorMap: The style and colour of heatmap                   dtype = string
    xmin, xmax: The minimum and maximum value of the x-axis
    ymin, ymax: The minimum and maximum value of the y-axis
    plot: "quiver", "stream" allowing for the overlay of quiver or streamline
          plot

    Returns:
    Image : AxesImage
    """
    plt.contourf(x_axis, y_axis, cbar, int_pol, cmap=colorMap)
    cbar_tag = plt.colorbar()
    Z = cbar
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    cbar_tag.set_label(cbar_label, rotation=270)
    cbar_tag.set_clim(-1000.0, 1000.0)
    E = np.gradient(Z)
    E = E/np.sqrt(E[0]**2 + E[1]**2)
    if plot is not None:
        if plot == "quiver":
            print("\nQuiver plot:")
            plt.quiver(x_axis, y_axis, E[1], E[0])
        if plot == "stream":
            print("\nStreamline Plot:")
            plt.streamplot(x_axis, y_axis, -E[1], -E[0], color='black')
    if xmin is not None and xmax is not None:
        plt.xlim(xmin, xmax)
    if ymin is not None and ymax is not None:
        plt.ylim(ymin, ymax)
    plt.show()


def e_field(x_axis, y_axis, cbar, x_label, y_label, cbar_label, int_pol,
            colorMap, xmin=None, xmax=None, ymin=None, ymax=None, plot=None):
    """A general plotter allowing for the plotting of a heatmap(primarily used
       here for electric field) which takes in relevant data about the graph
       It also allows for the option of overlaying either a quiver or a
       streamline plot, or not!

    Parameters:
    x_axis, y_axis : The corresponding x and y axis data lists of a plot
    cbar: The heatmap list data which usually corresponds to x and y axis
    x_label, y_label, : The x, y and z label of the graph       dtype = string
    cbar_label : The colourbar label                            dtype = string
    int_pol: The colour interpolation of the heatmap            dtype = integer
    colorMap: The style and colour of heatmap                   dtype = string
    xmin, xmax: The minimum and maximum value of the x-axis     dtype = int
    ymin, ymax: The minimum and maximum value of the y-axis     dtype = int
    plot: "quiver", "stream" allowing for the overlay of quiver or streamline
          plot

    Returns:
    Image : AxesImage
    """
    a, d = np.gradient(cbar)
    cbar2 = -a
    plt.contourf(x_axis, y_axis, cbar2, int_pol, cmap=colorMap)
    cbar2_tag = plt.colorbar()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    cbar2_tag.set_label(cbar_label, rotation=270)
    E = np.gradient(cbar)
    E = E/np.sqrt(E[0]**2 + E[1]**2)
    if plot is not None:
        if plot == "quiver":
            print("\nQuiver plot:")
            plt.quiver(x_axis, y_axis, E[1], E[0])
        if plot == "stream":
            print("\nStreamline Plot:")
            plt.streamplot(x_axis, y_axis, -E[1], -E[0], color='black')
    if xmin is not None and xmax is not None:
        plt.xlim(xmin, xmax)
    if ymin is not None and ymax is not None:
        plt.ylim(ymin, ymax)
    plt.show()


def multiline(x_axis, y_axis, x_label, y_label, l_names, l_title, y_max,
              loca=None, anchor_x=None, anchor_y=None):
    """Allows for ANY number of line to be plotted on the same graph, with the
       ability to label every line.

    Parameters:
    x_axis, y_axis: Takes in lists of lists of x and y data points
    x_label, y_label: The x and y label of the graph
    l_names: Takes in lists of strings as the corresponding label of each line
    l_title: Title of the legend                                 dtype = string
    y_max: Maximum value of the y axis                           dtype = float
    loca: Location of the legend box
    anchor_x, anchor_y: Coordinates for which the legend box is anchored

    Returns:
    Image : AxesImage
    """
    l_labels = l_names*len(y_axis)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, (y_axis, l_labels) in enumerate(zip(y_axis, l_labels)):
        ax.plot(x_axis, y_axis, label=l_labels)
    ax.legend(title=l_title, ncol=3)
    if loca is not None and anchor_x is not None and anchor_y is not None:
        ax.legend(title=l_title, bbox_to_anchor=(anchor_x, anchor_y), loc=loca,
                  ncol=3)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, y_max)
    plt.show()


def capacitor_bc(v, V):
    """Sets the boundary condition for a parallel plate capacitor

    Parameters:
    V : ndarray of a grid, will only be called within gauss_solver or
        jacobi_solver
    v : Potential difference between plates

    Returns:
    Updated ndarray of the initial grid taken into account of BC
    """
    V[16, 10:31] = -v                  #  here V[y, x] because Python is weird
    V[25, 10:31] = v
    return V


def linecharge_bc(v, V):
    """Sets the boundary condition for a line charge

    Parameters:
    V : ndarray of a grid, will only be called within gauss_solver or
        jacobi_solver
    v : Potential difference of the line charge

    Returns:
    Updated ndarray of the initial grid taken into account of BC
    """
    V[-1, 0:-1] = v
    return V


def gauss_solver(a, d, h, v, BC):
    """An iterative solver for the capacitor, works for both Gauss-Seidel and
       Jacobi methods. It allows for the input of different boundary conditions
       which runs within the function.

    Parameters:
    a  : Length of grid
    d  : Height of grid
    h  : Grid density
    v  : Potential difference
    BC : A boundary condition function

    Returns:
    X, Y, Z : Lists of meshgrid coordinates and potential(V)
    """
    M = np.zeros(shape=(int((a/h)+1), int((d/h)+1)))
    max_iter = 20000
    M1 = copy(M)
    count = 0

    while count < max_iter:
        count += 1
        M2 = copy(M1)
        for i in range(1, (M.shape[0]-1)):
            for j in range(1, (M.shape[1]-1)):
                # top left
                if i == j == 0:
                    M1[i, j] = 0.5*(M1[i+1, j]+M1[i, j+1])
                # top edge no corners
                elif i == 0 and j > 0 and j < (M.shape[1]-1):
                    M1[i, j] = (1/3)*(M1[i, j-1]+M1[i, j+1]+M1[i+1, j])
                # top right
                elif i == 0 and j == (M.shape[1]-1):
                    M1[i, j] = 0.5*(M1[i+1, j]+M1[i, j-1])
                # right edge no corners
                elif j == (M.shape[1]-1) and i > 0 and i < (M.shape[0]-1):
                    M1[i, j] = (1/3)*(M1[i-1, j]+M1[i+1, j]+M1[i, j-1])
                # bot right
                elif i == (M.shape[0]-1) and j == (M.shape[1]-1):
                    M1[i, j] = 0.5*(M1[i-1, j]+M1[i, j-1])
                # bot edge
                elif i == (M.shape[0]-1) and j > 0 and j < (M.shape[1]-1):
                    M1[i, j] = (1/3)*(M1[i, j-1]+M1[i, j+1]+M1[i-1, j])
                # bot left no corners
                elif i == (M.shape[0]-1) and j == 0:
                    M1[i, j] = 0.5*(M1[i-1, j]+M1[i, j+1])
                # left edge
                elif j == 0 and i > 0 and i < (M.shape[0]-1):
                    M1[i, j] = (1/3)*(M1[i-1, j]+M1[i+1, j]+M1[i, j+1])
                else:
                    M1[i, j] = 0.25*(M1[i-1, j]+M1[i+1, j]+M1[i, j-1]+M1[i, j+1])
                    BC(v, M1)
        if np.allclose(M1, M2, rtol=1e-3):
            print("\nConvergence occurs after {} iterations.".format(count))
            break
        else:
            sys.stdout.write("\r"+"Convergence did not happen before {} iterations.".format(count))
    x = np.linspace(0, a, int(a/h)+1)
    y = np.linspace(0, d, int(d/h)+1)
    return x, y, M1


def jacobi_solver(a, d, h, v, BC):
    """An iterative solver for the capacitor, works for both Gauss-Seidel and
       Jacobi methods. It allows for the input of different boundary conditions
       which runs within the function.

    Parameters:
    a  : Length of grid
    d  : Height of grid
    h  : Grid density
    v  : Potential difference
    BC : A boundary condition function

    Returns:
    X, Y, Z : Lists of meshgrid coordinates and potential(V)
    """
    M = np.zeros(shape=(int((a/h)+1), int((d/h)+1)))
    max_iter = 20000
    M1 = copy(M)
    count = 0
    while count < max_iter:
        count += 1
        M2 = copy(M1)
        for i in range(1, (M.shape[0]-1)):
            for j in range(1, (M.shape[1]-1)):         
                # top left
                if i == j == 0:
                    M1[i, j] = 0.5*(M2[i+1, j]+M2[i, j+1])
                # top edge no corners
                elif i == 0 and j > 0 and j < (M.shape[1]-1):
                    M1[i, j] = (1/3)*(M2[i, j-1]+M2[i, j+1]+M2[i+1, j])
                # top right
                elif i == 0 and j == (M.shape[1]-1):
                    M1[i, j] = 0.5*(M2[i+1, j]+M2[i, j-1])
                # right edge no corners
                elif j == (M.shape[1]-1) and i > 0 and i < (M.shape[0]-1):
                    M1[i, j] = (1/3)*(M2[i-1, j]+M2[i+1, j]+M2[i, j-1])
                # bot right
                elif i == (M.shape[0]-1) and j == (M.shape[1]-1):
                    M1[i, j] = 0.5*(M2[i-1, j]+M2[i, j-1])
                # bot edge no corners
                elif i == (M.shape[0]-1) and j > 0 and j < (M.shape[1]-1):
                    M1[i, j] = (1/3)*(M2[i, j-1]+M1[i, j+1]+M2[i-1, j])
                # bot left
                elif i == (M.shape[0]-1) and j == 0:
                    M1[i, j] = 0.5*(M2[i-1, j]+M2[i, j+1])
                # left edge no corners
                elif j == 0 and i > 0 and i < (M.shape[0]-1):
                    M1[i, j] = (1/3)*(M2[i-1, j]+M2[i+1, j]+M2[i, j+1])
                else:
                    M1[i, j] = 0.25*(M2[i-1, j]+M2[i+1, j]+M2[i, j-1]+M2[i, j+1])
                    BC(v, M1)

        if np.allclose(M1, M2, rtol=1e-3):
            print("\nConvergence occurs after {} iterations.".format(count))
            break
        else:
            sys.stdout.write("\r"+"Convergence did not happen before {} iterations.".format(count))
    x = np.linspace(0, a, int(a/h)+1)
    y = np.linspace(0, d, int(d/h)+1)
    return x, y, M1


def heat_eq(T, bc, temp_i, temp_f):
    """Solving the heat equation of a rod for a general boundary condition by
       using the backwards-Euler method. Since the matrix is tridiagonal, a
       sparse matrix is precomputed to save run time by not having to compute
       the elements with 0 value.

    Parameters:
    T      : The maximum run time for which dt is also calculated.
    bc     : A specific boundary condition that's suited for the situation
    temp_i : The initial temperature of the start of the rod
    temp_f : The initial temperature of the tail of the rod

    Returns:
    x : Length of the rod segmented up into points and stored in a list.
    u : The temperature of the rod at time T.
    """
    L, Nx, alpha = 0.5, 99, 59/(450*7900)

    x = np.linspace(0, L, Nx+1)
    t = np.linspace(0, T, Nx+1)
    dx, dt = x[1]-x[0], t[1]-t[0]
    u, u_n = np.zeros(Nx+1), np.zeros(Nx+1)
    K = alpha*dt/(dx**2)

    # Initiate sparse matrix and RHS solution of equation
    main = np.zeros(Nx+1)
    b = np.zeros(Nx+1)
    lower, upper = np.zeros(Nx), np.zeros(Nx)

    # Precompute sparse matrix
    main[:] = 1 + 2*K
    lower[:] = -K
    upper[:] = -K

    # Insert boundary conditions
    main, lower, upper = bc(main, lower, upper, K)

    A = diags(diagonals=[main, lower, upper], offsets=[0, -1, 1], shape=(Nx+1,
              Nx+1), format='csr')

#    print(A.todense())  # Check that A is correct

    # Set initial condition
    for i in range(0, Nx+1):
        u_n[i] = 20

    for n in range(0, T):
        b = u_n
        b[0] = temp_i  # bc start of rod
        b[-1] = temp_f    # bc end of rod
        u[:] = spsolve(A, b)
        u_n[:] = u

    return x, u


def wireframe(x_axis, y_axis, cbar, offset, rs, cs):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(x_axis, y_axis, cbar, color='red', rstride=rs, cstride=cs,linewidth=0.5)
    ax.set_xlabel("x(m)")
    ax.set_ylabel("y(m)")
    ax.set_zlabel("$\phi (x,y)$", rotation = 180)
    plt.show()


def wire_anal(x, y, L, N):
    series = 0
    if N > 114:
        for n in range(1, N):
            series += 4000*np.sin(((2*n)-1)*np.pi*x/L)*np.exp(((2*n)-1)*np.pi*((y/L)-1))/(((2*n)-1)*np.pi)
    else:
        for n in range(1, N):
            series += (4000/(((2*n)-1)*np.pi))*np.sin(((2*n)-1)*np.pi*x/L)*(np.sinh(((2*n)-1)*np.pi*y/L))/(np.sinh(np.pi*((2*n)-1)))
    return series


def heat_anal(x, t, L, N):
    k = 59/(450*7900)
    T_0 = 20   # u(x,0) = 0, u(0,0) = 0
    series = 0
    for n in range(1, N):
        series += (4*T_0/(((2*n)-1)*np.pi))*(np.sin((((2*n)-1)*x*np.pi)/L))*np.exp((-k*t*(np.pi*((2*n)-1)/L)**2))
    return series


def choice_a():
    """Handles choice b in MainMenu()"""
    t1 = clock()
    X, Y, V = gauss_solver(50, 50, 1, 1000, linecharge_bc)
    print("\n\nPotential")
    potential(X, Y, V, "x(cm)", "y(cm)", "Potential(V)", 30, cm.jet, 0, 50, 0,
              50, "stream")
    t1 = clock() - t1
    print("Took {}(s)".format(t1))
    t1 = clock()
    print("\n\nElectric field")
    e_field(X, Y, V, "x(cm)", "y(cm)", "Electric field", 30, cm.jet, 0, 50, 0,
            50, "stream")
    t1 = clock() - t1
    print("Took {}(s)".format(t1))
    Y = 0.25
    X = np.linspace(0, 50, 101)
    u0 = wire_anal(X, Y, 50, 40000)
    X, Y, V = gauss_solver(50, 50, 0.5, 1000, linecharge_bc)
    print("\n\nDifference between analytical Fourier series and GS solution.")
    multiline(X, [u0, V[26,:]], "y(cm)", "$\phi (x, y=0.25)$", ["Analytical",
              "GS"], "Method", 60, 'upper center', 0.5, 1.1)
    X = Y = np.linspace(0, 0.5, 101)
    X, Y = np.meshgrid(X, Y)
    Z = wire_anal(X, Y, 0.5, 21)
    print("\n\nGibbs phenomenon")
    wireframe(X, Y, Z, 0, 3, 3)


def choice_b():
    """Handles choice b in MainMenu()"""
    t1 = clock()
    print("\n\nPotential overlayed with quiver plot using Gauss-Seidel method")
    X, Y, V = gauss_solver(40, 40, 1, 1000, capacitor_bc)
    potential(X, Y, V, "x(cm)", "y(cm)", "Potential(V)", 30, cm.jet, 0, 40, 0,
              40, "quiver")
    t1 = clock() - t1
    print("Took {}(s)".format(t1))
    t2 = clock()
    potential(X, Y, V, "x(cm)", "y(cm)", "Potential(V)", 25, cm.hot, 0, 40, 0,
              40, "stream")
    t2 = clock() - t2
    print("Took {}(s)".format(t2))
    t1 = clock()
    print("\n\nElectric field overlayed with streamline plot using Gauss-Seidel method")
    e_field(X, Y, V, "x(cm)", "y(cm)", "Electric field(V/m)", 30, cm.hot, 0,
            40,0,40, "stream")
    t1 = clock() - t1
    print("Took {}(s)".format(t1))
    t3 = clock()
    print("\n\nPotential overlayed with quiver plot using Jacobi method")
    X, Y, V = jacobi_solver(40, 40, 1, 1000, capacitor_bc)
    potential(X, Y, V, "x(cm)", "y(cm)", "Potential(V)", 30, cm.jet, 0, 40, 0,
              40, "quiver")
    t3 = clock() - t3
    print("Took {}(s)".format(t3))
    t4 = clock()
    print("\n\nElectric field overlayed with streamline plot using Jacobi method")
    e_field(X, Y, V, "x(cm)", "y(cm)", "Electric field(V/m)", 25, cm.hot, 0,
            40, 0, 40, "stream")
    t4 = clock() - t4
    print("Took {}(s)".format(t4))


def choice_d():
    """Handles choice d in MainMenu()"""
    x, u0 = heat_eq(1, both_ice, 0, 0)
    sols0 = heat_anal(x, 1, 0.5, 1000)
    k0 = np.abs(sum(sols0-u0))/len(sols0)

    x, u = heat_eq(50, both_ice, 0, 0)
    sols = heat_anal(x, 50, 0.5, 1000)
    k = np.abs(sum(sols-u))/len(sols)

    x, u2 = heat_eq(150, both_ice, 0, 0)
    sols2 = heat_anal(x, 150, 0.5, 1000)
    k2 = np.abs(sum(sols2-u2))/len(sols2)

    x, u3 = heat_eq(250, both_ice, 0, 0)
    sols3 = heat_anal(x, 250, 0.5, 1000)
    k3 = np.abs(sum(sols3-u3))/len(sols3)

    x, u4 = heat_eq(350, both_ice, 0, 0)
    sols4 = heat_anal(x, 350, 0.5, 1000)
    k4 = np.abs(sum(sols4-u4))/len(sols4)

    x, u5 = heat_eq(450, both_ice, 0, 0)
    sols5 = heat_anal(x, 450, 0.5, 1000)
    k5 = np.abs(sum(sols5-u5))/len(sols5)

    x, u6 = heat_eq(550, both_ice, 0, 0)
    sols6 = heat_anal(x, 550, 0.5, 1000)
    k6 = np.abs(sum(sols6-u6))/len(sols6)

    x, u7 = heat_eq(650, both_ice, 0, 0)
    sols7 = heat_anal(x, 650, 0.5, 1000)
    k7 = np.abs(sum(sols7-u7))/len(sols7)

    x, u8 = heat_eq(750, both_ice, 0, 0)
    sols8 = heat_anal(x, 750, 0.5, 1000)
    k8 = np.abs(sum(sols8-u8))/len(sols8)

    x, u9 = heat_eq(1000, both_ice, 0, 0)
    sols9 = heat_anal(x, 1000, 0.5, 1000)
    k9 = np.abs(sum(sols9-u9))/len(sols9)

    x, u10 = heat_eq(1200, both_ice, 0, 0)
    sols10 = heat_anal(x, 1200, 0.5, 1000)
    k10 = np.abs(sum(sols10-u10))/len(sols10)

    x, u11 = heat_eq(1400, both_ice, 0, 0)
    sols11 = heat_anal(x, 1400, 0.5, 1000)
    k11 = np.abs(sum(sols11-u11))/len(sols11)

    multiline(x, [u0, u, u2, u3, u4, u5, u6, u7, u8], "Length(m)",
              "Temperature($^{\circ}$C)", [1, 50, 150, 250, 350, 450, 550, 650,
                          750], "Time(s)", 21, 'upper center', 0.5, 1.1)    
    print("\n\nAbsolute error between analytical Fourier series solution and GS solution.")
    a = [1, 50, 150, 250, 350, 450, 550, 650, 750, 1000, 1200, 1400]
    b = [k0, k, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11]
    plt.plot(a, b, 'ro-')
    plt.xlabel("Total Time(s)")
    plt.ylabel("Absolute Error")


def choice_e():
    """Handles choice d in MainMenu()"""
    x, u0 = heat_eq(1, adiabatic, 1000, 20)
    x, u = heat_eq(50, adiabatic, 1000, 20)
    x, u2 = heat_eq(150, adiabatic, 1000, 20)
    x, u3 = heat_eq(350, adiabatic, 1000, 20)
    x, u4 = heat_eq(750, adiabatic, 1000, 20)
    x, u5 = heat_eq(4000, adiabatic, 1000, 20)
    x, u6 = heat_eq(10000, adiabatic, 1000, 20)
    x, u7 = heat_eq(20000, adiabatic, 1000, 20)
    x, u8 = heat_eq(50000, adiabatic, 1000, 20)
    multiline(x, [u0, u, u2, u3, u4, u5, u6, u7, u8], "Length(m)",
              "Temperature(Degree Celsius)", [1, 50, 150, 350, 750, 4000,
              "$1x10^{4}$", "$2x10^{4}$", "$5x10^{4}$"], "Time(s)", 1100,
              'upper center', 0.5, 1.1)


def choice_f():
    """Handles choice d in MainMenu()"""
    x, u0 = heat_eq(1, hot_cold, 1000, 0)
    x, u = heat_eq(25, hot_cold, 1000, 0)
    x, u2 = heat_eq(100, hot_cold, 1000, 0)
    x, u3 = heat_eq(200, hot_cold, 1000, 0)
    x, u4 = heat_eq(300, hot_cold, 1000, 0)
    x, u5 = heat_eq(400, hot_cold, 1000, 0)
    x, u6 = heat_eq(500, hot_cold, 1000, 0)
    x, u7 = heat_eq(600, hot_cold, 1000, 0)
    x, u8 = heat_eq(700, hot_cold, 1000, 0)
    multiline(x, [u0, u, u2, u3, u4, u5, u6, u7, u8], "Length(m)",
              "Temperature(Degree Celsius)", [1, 25, 100, 200, 300, 400, 500,
              600, 700], "Time(s)", 1100)


def MainMenu():
    choice = '0'
    while choice != 'q':
        print("\n%s\nData Analysis\n%s" % (13*'=', 13*'='))
        print("(a)Solves Laplace's equation for a line charge.")
        print("(b)Calculate the potential and electric field within and around",
              "a parallel plate capacitor")
        print("(c)Investigate field configuration as a/d becomes large.")
        print("Temperature distribution plotted at different times:")
        print("(d)Starting with ice at both ends of poker, and compared with",
              "its analytical Fourier series solution.")
        print("(e)with no heat loss from the far end of the poker")
        print("(f)with far end of poker immersed in a block of ice at 0*C.")
        print("(g)\n(q)")
        choice = (input("Please enter your choice [a-q] : ").lower())
        if choice == 'a':
            choice_a()
        elif choice == 'b':
            choice_b()
        elif choice == 'd':
            choice_d()
        elif choice == 'e':
            choice_e()
        elif choice == 'f':
            choice_f()
        elif choice != 'q':
            print("Invalid choice. Please try again.")


MainMenu()
