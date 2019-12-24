"""
Created on Fri Feb 23 16:24:33 2018
@author: darren
"""
import numpy as np
import matplotlib.pyplot as plt

"""
------------------------------Glossary------------------------------
(Normal Euler)   (Analytical)               (Modified Euler)
V = Velocity     AV = Analytical Velocity   AV = Modified Velocity
D = Distance     AD = Analytical Distance   AD = Modified Distance

T = Time
DT = Time step 'dt'
VARY = The status of varying air density; either exist or "None"
l_ = Legend i.e. l_title = Legend Title, l_names = Legend Names, l_labels = Legend labels
S_BARR = Sound Barrier
"""
consts = [340.3, 334.4, 328.4, 322.2, 316.0, 309.6, 303.1, 295.4, 294.9]  #Sound barrier values at different interval of heights

############################ EULER ############################
def EULER(Y_0, DT, G, K, M, VARY=None):
    # This EULER function alone works for all scenarios(with or without varying density)
    V_0, T = 0, [0]
    V, MV, AV = [V_0], [V_0], [V_0]
    D, MD, AD = [Y_0], [Y_0], [Y_0]

    i = 0
    while i >= 0:
        if VARY is not None:
            #Analytical doesn't apply here when air density is varying
            #Appending time
            T.append(T[i] + DT)
            #Normal Euler
            V.append(V[i] - DT*(G + ((K/M)*np.exp(-1*(D[i]/7640))*abs(V[i])*V[i])))
            D.append(D[i] + DT*V[i])
            #Modified Euler
            midV = MV[i] - 0.5*DT*(G + ((K/M)*np.exp(-1*(MD[i]/7640))*abs(MV[i])*MV[i]))
            MV.append(MV[i] - DT*(G + ((K/M)*np.exp(-1*(MD[i]/7640))*abs(midV)*midV)))
            MD.append(MD[i] + DT*midV)
            if D[i+1] <= 0 or MD[i+1] <= 0:
               break
            i += 1

        else:
            #Appending time
            T.append(T[i] + DT)
            #Normal Euler
            V.append(V[i] - DT*(G + ((K/M)*abs(V[i])*V[i])))
            D.append(D[i] + DT*V[i])
            #Modified Euler
            midV = MV[i] - 0.5*DT*(G + ((K/M)*abs(MV[i])*MV[i]))
            MV.append(MV[i] - DT*(G + ((K/M)*abs(midV)*midV)))
            MD.append(MD[i] + DT*midV)
            #Analytical
            AD.append(AD[0] - ((1/(2*(K/M)))*np.log((np.cosh(np.sqrt((K/M)*G)*T[i+1]))**2)))
            AV.append(-1*np.sqrt(G/(K/M))*np.tanh(T[i+1]*np.sqrt((K/M)*G)))

            if D[i+1] <= 0 or MD[i+1] <= 0 or AD[i+1] <= 0:
                break
            i += 1
    print(D)
    ERROR_V = abs(np.array(V) - np.array(AV))        #Turning errors into arrays for plotting
    ERROR_D = abs(np.array(D) - np.array(AD))
    ERROR_MV = abs(np.array(MV) - np.array(AV))
    ERROR_MD = abs(np.array(MD) - np.array(AD))
    return V, D, AV, AD, MV, MD, ERROR_V, ERROR_D, ERROR_MV, ERROR_MD, T

def MAXVTIME(V, D, T):   #Finds the distance and time when max velocity is reached
    MAX_D = D[np.argmin(V)]
    MAX_T = T[np.argmin(V)]
    return MAX_D, MAX_T

############################ PARAMETERS ############################
def Parameters():
    G, RHO_0 = 9.81, 1.2
    choice2 = '0'
    print (28 * '=')
    print ("To use: \n(a) Default parameter values \n(b) Custom parameter values")
    print (28 * '=')
    while True:
        choice2 = (input("Please enter your choice [a-b]: ").lower())
        if choice2 == 'a':
            print("\n%s\nDefault Parameters Value"%(57 * "="))
            Cd = 1.2     #drag coefficient
            A = 0.1*0.1*np.pi
            M, DT, Y_0 = 80, 0.01, 1000
            break

        elif choice2 == 'b':
            while True:
                try:
                    Cd = float(input("Please give a value for the drag coefficient: "))
                    break
                except:
                    print("Invalid input. Please try again.\n")
            while True:
                try:
                    A = float(input("Please give a value for the cross sectional area of the projectile: "))
                    break
                except:
                    print("Invalid input. Please try again.\n")
            while True:
                try:
                    M = float(input("Please give a value for the mass of the object: "))
                    break
                except:
                    print("Invalid input. Please try again.\n")
            while True:
                try:
                    Y_0 = float(input("Please give a value for the height at which the object is released: "))
                    break
                except:
                    print("Invalid input. Please try again.\n")
            while True:
                try:
                    DT = float(input("Please give a value for the time step size 'dt' : "))
                    print("\n%s\nCustom Parameters Value"%(57 * "="))
                    break
                except:
                    print("Invalid input. Please try again.\n")
            break
        else:
            print("\nInvalid input. Choices can only be [a/b].\n")

    K  = (Cd*RHO_0*A)/2
    print("%s\nGravitational Constant(g): {}(m/s)".format(G)%(57 * "="))
    print("Air Density(P_0)\t : {}(kgm^(-3))".format(RHO_0))
    print("Drag Coefficient(C_d)\t : {}(N/A)".format(Cd))
    print("Cross Sectional Area(A)\t : {}(m)".format(A))
    print("Mass(m)\t\t\t : {}(kg)".format(M))
    print("Starting Height(y_0)\t : {}(m)".format(Y_0))
    print("Time-step Size(dt)\t : {}(s)".format(DT))
    print("Constant, (k)\t\t : {}(kgm^(-1))".format(K))
    print("Constant, (k/m)\t\t : {}(m^(-1))\n%s".format(K/M)%(57 * "="))
    return G, RHO_0, Cd, A, M, Y_0, DT, K

def S_BARR(y):
    # This function returns the value of sound barrier at certain height
    return consts[-1 if (not (0 < y < 12192) or y % 1524 == 0) else y // 1524]

def SUMMARY(V, D, T, MAX_D, MAX_T, METHOD):    #This function summarises the statistics of the fall
    print("{} method:\nThe projectile reaches the floor after {}(s) with a velocity of {}(m/s) from a height of {}(m).\n".format(METHOD, T[-1], V[-1], D))
    print("{}(s) after it was released, a maximum velocity of {}(m/s) was reached at a height of {}(m). At such altitude the sound barrier was approximately {}(m/s).".format(MAX_T, np.min(V), MAX_D, S_BARR(D)))

    if S_BARR(D) < -np.amin(V):
        print("\nTherefore the projectile broke the sound barrier!\n")
    else:
        print("\nTherefore the projectile did not break the sound barrier.\n")

    return SUMMARY

def GRAPH(x_axis, y_axis, x_label, y_label, title, l_names, l_title): #This function allows any number of lines to be plotted in the same graph
    l_labels = l_names * len(y_axis)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    for i, (y_axis, l_labels) in enumerate(zip(y_axis, l_labels)):
        ax1.plot(x_axis, y_axis, label=l_labels)

    ax1.legend(title=l_title)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(title)
    plt.show()

def MainMenu():
    choice = '0' #Initialising Variables
    while choice != 'q':
        print ("%s\n(a) Euler Method \n(b) Analytical Method \n(c) Modified Euler Method \n(d) Compare all 3 methods \n(e) Varying Air Density \n(q) To quit\n%s"%(28 * '=', 28* '='))
        choice = (input("Please enter your choice [a-q] : ").lower())
        if choice == 'a':
            #Obtain values from user then call them into Euler's function
            G, RHO_0, Cd, A, M, Y_0, DT, K = Parameters()
            V, D, AV, AD, MV, MD, ERROR_V, ERROR_D, ERROR_MV, ERROR_MD, T = EULER(Y_0, DT, G, K, M)
            MAX_D, MAX_T = MAXVTIME(V, D, T)
            #Plot graphs
            GRAPH(T, [D], "Time(s)", "Distance(m)", "Distance vs Time - Constant Air Density", ['Basic Euler'], "Method")
            GRAPH(T, [ERROR_D], "Time(s)", "Distance Error(m)", "Absolute Error in Distance - Basic Euler", ["Basic Euler"], "Method")
            GRAPH(T, [V], "Time(s)", "Velocity(m/s)", "Velocity vs Time - Constant Air Density", ['Basic Euler'], "Method")
            GRAPH(T, [ERROR_V], "Time(s)", "Velocity Error(m/s)", "Absolute Error in Velocity", ['Basic Euler'], "Method")
            #Summarise
            SUMMARY(V, Y_0, T, MAX_D, MAX_T , "Basic Euler")

        elif choice == 'b':
            #Obtain values from user then call them into Euler's function
            G, RHO_0, Cd, A, M, Y_0, DT, K = Parameters()
            V, D, AV, AD, MV, MD, ERROR_V, ERROR_D, ERROR_MV, ERROR_MD, T = EULER(Y_0, DT, G, K, M)
            MAX_D, MAX_T = MAXVTIME(V, D, T)
            #Plot graphs
            GRAPH(T, [AD], "Time(s)", "Distance(m)", "Distance vs Time - Constant Air Density", ['Analytical'], "Method")
            GRAPH(T, [AV], "Time(s)", "Velocity(m/s)", "Velocity vs Time - Constant Air Density", ['Analytical'], "Method")
            #Summarise
            SUMMARY(AV, Y_0, T, MAX_D, MAX_T , "Analytical")

        elif choice == 'c':
            #Obtain values from user then call them into Euler's function
            G, RHO_0, Cd, A, M, Y_0, DT, K = Parameters()
            V, D, AV, AD, MV, MD, ERROR_V, ERROR_D, ERROR_MV, ERROR_MD, T = EULER(Y_0, DT, G, K, M)
            MAX_D, MAX_T = MAXVTIME(V, D, T)
            #Plot graphs
            GRAPH(T, [MD], "Time(s)", "Distance(m)", "Distance vs Time - Constant Air Density", ['Mod Euler'], "Method")
            GRAPH(T, [ERROR_MD], "Time(s)", "Distance Error(m)", "Absolute Error in Distance", ['Mod Euler'], "Method")
            GRAPH(T, [MV], "Time(s)", "Velocity(m/s)", "Velocity vs Time - Constant Air Density", ['Mod Euler'], "Method")
            GRAPH(T, [ERROR_MV], "Time(s)", "Velocity Error(m/s)", "Absolute Error in Velocity", ['Mod Euler'], "Method")
            #Summarise
            SUMMARY(MV, Y_0, T, MAX_D, MAX_T , "Mod Euler")
        elif choice == 'd':
                        #Obtain values from user then call them into Euler's function
            G, RHO_0, Cd, A, M, Y_0, DT, K = Parameters()
            V, D, AV, AD, MV, MD, ERROR_V, ERROR_D, ERROR_MV, ERROR_MD, T = EULER(Y_0, DT, G, K, M)
            MAX_D, MAX_T = MAXVTIME(V, D, T)
            #Plot graphs
            GRAPH(T, [D, MD, AD], "Time(s)", "Distance(m)", "Distance vs Time - Constant Air Density", ['Basic Euler', 'Mod Euler','Analytical'], "Method")
            GRAPH(T, [ERROR_D, ERROR_MD], "Time(s)", "Distance Error(m)", "Absolute Error in Distance", ['Basic Euler', 'Mod Euler'], "Method")
            GRAPH(T, [V, MV, AV], "Time(s)", "Velocity(m/s)", "Velocity vs Time - Constant Air Density", ['Basic Euler', 'Mod Euler','Analytical'], "Method")
            GRAPH(T, [ERROR_V, ERROR_MV], "Time(s)", "Velocity Error(m/s)", "Absolute Error in Velocity", ['Basic Euler', 'Mod Euler'], "Method")
            #Summarise
            SUMMARY(V, Y_0, T, MAX_D, MAX_T , "Basic Euler")
            SUMMARY(MV, Y_0, T, MAX_D, MAX_T , "Mod Euler")
            SUMMARY(AV, Y_0, T, MAX_D, MAX_T , "Analytical")


        elif choice == 'e':
            choice3 = '0'
            while True:
                print (28 * '=')
                print ("(a) Euler Method \n(b) Modified Euler Method")
                print (28 * "=")
                choice3 = (input("Please enter your choice [a-b]: ").lower())
                if choice3 == 'a':
                    #Obtain values from user then call them into Euler's function
                    G, RHO_0, Cd, A, M, Y_0, DT, K = Parameters()
                    V, D, AV, AD, MV, MD, ERROR_V, ERROR_D, ERROR_MV, ERROR_MD, T = EULER(Y_0, DT, G, K, M, 'y')
                    MAX_D,      MAX_T = MAXVTIME(V, D, T)
                    #Plot graphs
                    GRAPH(T, [D], "Time(s)", "Distance(m)", "Distance vs Time - Varying Air Density", ['Basic Euler'], "Method")
                    GRAPH(T, [V], "Time(s)", "Velocity(m/s)", "Velocity vs Time - Varying Air Density", ['Basic Euler'], "Method")
                    SUMMARY(V, Y_0, T, MAX_D, MAX_T , "Basic Euler")
                    break

                elif choice3 == 'b':
                    #Obtain values from user then call them into Euler's function
                    G, RHO_0, Cd, A, M, Y_0, DT, K = Parameters()
                    V, D, AV, AD, MV, MD, ERROR_V, ERROR_D, ERROR_MV, ERROR_MD, T = EULER(Y_0, DT, G, K, M, 'y')
                    MAX_D, MAX_T = MAXVTIME(V, D, T)
                    #Plot graphs
                    GRAPH(T, [MD], "Time(s)", "Distance(m)", "Distance vs Time - Varying Air Density", ['Mod Euler'], "Method")
                    GRAPH(T, [MV], "Time(s)", "Velocity(m/s)", "Velocity vs Time - Varying Air Density", ['Mod Euler'], "Method")
                    #Summarise
                    SUMMARY(MV, Y_0, T, MAX_D, MAX_T , "Mod Euler")

                    break
                else:
                    print("\nInvalid input. Choices can only be [a/b].\n")
        elif choice != 'q':
            print("Invalid choice. Please try again.")

MainMenu()