# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:42:08 2018

@author: dn16018
"""
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.pyplot as plt
import sys

#Default Variables(Global)
MASS_SAT = 549054           #Satelite's Mass
MASS_1 = 5.972E24           #Earth's Mass
RADIUS_1 = 6.3781E6         #Earth's Radius
MASS_2 = 7.34767E22         #Moon's Mass
RADIUS_2 = 1.737E6          #Moon's Radius
G = 6.67E-11                #Gravitational Constant
V_X = 0                     #Starting x-velocity of satelite
V_Y = 1000                  #Starting y-velocity of satelite
X = 3.844E8                 #Starting x-position of satelite
Y = 0                       #Starting y-position of satelite

class Simulation:
    def __init__(self, name_sat, mass_sat, name_1, mass_1, radius_1, h, t, 
                 x = 0, y = 0, v_x = 0, v_y = 0, multi_body = None, name_2 = 
                 None, mass_2 = None, radius_2 = None, x0_2 = None, y0_2 = None):
        """ Here names, masses and radii are defined for the satelite, body 1 and body 2
            by the use of class. x, y, v_x, v_y are the positions and velocities of 
            satelite. If the multi_body is True then extra fields will be turned on.
        """
        self.name_sat = name_sat
        self.mass_sat = mass_sat
        self.name_1 = name_1
        self.mass_1 = mass_1
        self.radius_1 = radius_1
        self.h = h
        self.x = []
        self.y = []
        self.v_x = []
        self.v_y = []
        self.t = []
        self.kinetic_energy = []
        self.potential_energy = []
        self.total_energy = []
        
        self.multi_body = multi_body
        if self.multi_body:               #If multi_body is true
            self.name_2 = name_2
            self.mass_2 = mass_2
            self.radius_2 = radius_2
            self.x0_2 = x0_2
            self.y0_2 = y0_2
        
        self.x.append(x)		
        self.y.append(y)		
        self.v_x.append(v_x)		
        self.v_y.append(v_y)	
        self.t.append(t)
        
        self.kinetic_energy.append(self.get_kinetic_energy(self.v_x[-1], self.v_y[-1]))
        self.potential_energy.append(self.get_potential_energy())
        self.total_energy.append(self.get_total_energy(self.x[-1], self.y[-1], self.v_x[-1], self.v_y[-1]))
        
    def print_info_a(self):
        print("\nName of satelite                : {}     ".format(self.name_sat))
        print("Mass of satelite                : {}kg(s)".format(self.mass_sat))
        print("Name of body 1                  : {}     ".format(self.name_1))
        print("Mass of body 1                  : {}kg(s)".format(self.mass_1))
        print("Radius of body 1                : {}m(s) ".format(self.radius_1))
        print("Time-step h                     : {}s    ".format(self.h))
        print("Starting x-coordinate wrt Earth : {}m(s) ".format(self.x))
        print("Starting y-coordinate wrt Earth : {}m(s) ".format(self.y))
        print("Starting x-velocity wrt Earth   : {}m/s  ".format(self.v_x))
        print("Starting y-velocity wrt Earth   : {}m/s  ".format(self.v_y))

    def print_info_b(self):
        print("Name of body 2                  : {}     ".format(self.name_2))
        print("Mass of body 2                  : {}kg(s)".format(self.mass_2))
        print("Radius of body 2                : {}m(s) ".format(self.radius_2))
        print("X-distance from body 2 to body 1: {}m(s) ".format(self.x0_2))
        print("Y-distance from body 2 to body 1: {}m(s) ".format(self.y0_2))        
        
    def f_1(self, v_x):
        return v_x
      
    def f_2(self, v_y):
    	return v_y
      
    def f_3(self, x, y):
        if self.multi_body:
            return (-G*self.mass_1*x/(x**2 + y**2)**(3/2) - G*self.mass_2*
                    (x - self.x0_2)/((x - self.x0_2)**2 + (y - self.y0_2)**2)**(3/2))
        else:
            return -G*self.mass_1*x/(x**2 + y**2)**(3/2) 

    def f_4(self, x, y):
        if self.multi_body:
            return (-G*self.mass_1*y/(x**2 + y**2)**(3/2) - G*self.mass_2*
                    (y - self.y0_2)/((x-self.x0_2)**2 + (y - self.y0_2)**2)**(3/2))
        else:
            return -G*self.mass_1*y/(x**2 + y**2)**(3/2)    
    
    def runge_kutta(self):
        #Runge-Kutta coefficients calculations
        k1_x = self.f_1(self.v_x[-1])
        k1_y = self.f_2(self.v_y[-1])
        k1_vx = self.f_3(self.x[-1], self.y[-1])
        k1_vy = self.f_4(self.x[-1], self.y[-1])
          
        k2_x = self.f_1(self.v_x[-1] + self.h*k1_vx/2)
        k2_y = self.f_2(self.v_y[-1] + self.h*k1_vy/2)
        k2_vx = self.f_3(self.x[-1] + self.h*k1_x/2, self.y[-1] + self.h*k1_y/2)
        k2_vy = self.f_4(self.x[-1] + self.h*k1_x/2, self.y[-1] + self.h*k1_y/2)
          
        k3_x = self.f_1(self.v_x[-1] + self.h*k2_vx/2)
        k3_y = self.f_2(self.v_y[-1] + self.h*k2_vy/2)
        k3_vx = self.f_3(self.x[-1] + self.h*k2_x/2, self.y[-1] + self.h*k2_y/2)
        k3_vy = self.f_4(self.x[-1] + self.h*k2_x/2, self.y[-1] + self.h*k2_y/2)
          
        k4_x = self.f_1(self.v_x[-1] + self.h*k3_vx)
        k4_y = self.f_2(self.v_y[-1] + self.h*k3_vy)
        k4_vx = self.f_3(self.x[-1] + self.h*k3_x, self.y[-1] + self.h*k3_y)
        k4_vy = self.f_4(self.x[-1] + self.h*k3_x, self.y[-1] + self.h*k3_y)
        
        #Appending the distances and speed onto a list
        self.x.append(self.x[-1] + self.h/6*(k1_x + 2*k2_x + 2*k3_x + k4_x))
        self.y.append(self.y[-1] + self.h/6*(k1_y + 2*k2_y + 2*k3_y + k4_y))
        self.v_x.append(self.v_x[-1] + self.h/6*(k1_vx + 2*k2_vx + 2*k3_vx + k4_vx))
        self.v_y.append(self.v_y[-1] + self.h/6*(k1_vy + 2*k2_vy + 2*k3_vy + k4_vy))
         
        #Appending the energies onto a list
        self.kinetic_energy.append(self.get_kinetic_energy(self.v_x[-1], self.v_y[-1]))
        self.potential_energy.append(self.get_potential_energy())
        self.total_energy.append(self.get_total_energy(self.x[-1], self.y[-1], self.v_x[-1], self.v_y[-1]))
          
        self.t.append(self.t[-1] + self.h)
    
    def get_kinetic_energy(self, v_x, v_y):          #Returns kinetic energy
        return 0.5*self.mass_sat*((v_x**2) + (v_y**2)) 
    
    def get_potential_energy(self):                  #Returns potential energy
        return -G*self.mass_sat*self.mass_1/self.get_radius()
    
    def get_total_energy(self, x, y, v_x, v_y):      #Returns total energy
        return self.get_kinetic_energy(v_x, v_y) + self.get_potential_energy()
      
    def get_radius(self):               #Returns radius of satelite orbit wrt origin
        return math.sqrt(self.x[-1]**2 + self.y[-1]**2)
    
    def has_crashed(self):
        #boolean crash status - it compares the satelite orbit's radius with earth's radius
        has_crashed = (math.sqrt((self.x[-1])**2 + (self.y[-1])**2) < self.radius_1)
        
        if self.multi_body:      #compares the satelite's radius orbit with moon's radius
            distance_to_body_2 = math.sqrt((self.x[-1] - self.x0_2)**2 + (self.y[-1] - self.y0_2)**2)
            has_crashed |= (distance_to_body_2 < self.radius_2)                                       
            # a |= b     has_crashed boolean is true or false depending on the comparison     
            return has_crashed
        
    def update(self):
        self.runge_kutta()
        return self.has_crashed()     #Returns the boolean of the crash check 
                                      #status for every iteration            
def plot_distance(body):              #Distance plot function
    fig, ax = plt.subplots()
    ax.plot(body.x, body.y, color = 'white', label = body.name_sat)
    ax.set_xlabel("X distance (m)")
    ax.set_ylabel("Y distance (m)")
    plt.suptitle("X against Y")
    ax.set_aspect("equal")
    ax.set_xlim((1.2*min(body.x), 1.2*max(body.x)))
    ax.set_ylim((1.2*min(body.y), 1.2*max(body.y)))
    ax.add_artist(plt.Circle((0,0), 6.371E6, color = "blue"))       #Plots Earth
    ax.add_artist(plt.Circle((3.844E8,0), 1.737E6, color = 'grey')) #Plots Moon
    ax.set_facecolor('black')
    plt.legend()
    plt.show()

def plot_energies(body):                            #Energy plot function
    plt.plot(body.t, body.kinetic_energy, label = "KE")
    plt.plot(body.t, body.potential_energy, label = "PE")
    plt.plot(body.t, body.total_energy, label = "TE")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J")
    plt.title("Energies against Time")
    plt.legend(title = body.name_sat)
    plt.show()

def plot_threeD(body, part_b = False):              #3D plot function
    """ the x and y coordinates are predetermined and are not lists
        so their lists are made by multiplying the length of the lists of the
        x and y coordinates of the satelite
    """
    fig = plt.figure()                              
    ax = fig.gca(projection='3d')
    ax.plot(body.x, body.y, body.t, color = 'red', label = body.name_sat)
    ax.plot([0]*len(body.x), [0]*len(body.y), body.t, color = 'blue', label = body.name_1)
    if part_b:
        ax.plot([body.x0_2]*len(body.x), [body.y0_2]*len(body.y), body.t, color = 'grey', label= body.name_2)
    ax.legend()               
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Time (s)")
    plt.show()

def plot_distanceD(body):
    """Graphing function for the attempted three-body problem"""
    fig, ax = plt.subplots()
    ax.plot(body.x_1, body.y_1, label = body.name_1)
    ax.plot(body.x_2, body.y_2, label = body.name_2)
    ax.plot(body.x_3, body.y_3, label = body.name_3)
    ax.set_xlabel("X distance (m)")
    ax.set_ylabel("Y distance (m)")
    plt.suptitle("X against Y")
    ax.add_artist(plt.Circle((0,0), 6.371E6))        #Plots Earth
    ax.add_artist(plt.Circle((3.844E8,0), 1.737E6))  #Plots Moon
    plt.legend()
    plt.show()

"""Menu"""

def handle_choice_aa(N):
    moon = Simulation("Moon", MASS_2, "Earth", MASS_1, RADIUS_1, 100, 0, 3.844E8, 0, 0, 1000)
    moon.print_info_a()
    for i in range(N):
        crash_status = moon.update()
        sys.stdout.write("{}You're {}% way through.".format("\r", (int(i*100/N) + 1)))
        
        if crash_status == True:
            print("Crashed!")
            break
    
    plot_distance(moon)
    plot_energies(moon)

def handle_choice_ab(N):
    iss = Simulation("ISS", 417289, "Earth", MASS_1, RADIUS_1, 100, 0, 6.371E6+4.08E5, 0, 0, 7670)
    iss.print_info_a()
    for i in range(N):
        crash_status = iss.update()
        sys.stdout.write("{}You're {}% way through.".format("\r", (int(i*100/N) + 1)))
        
        if crash_status == True:
            print("It crashed!")
            break
    plot_distance(iss)
    plot_energies(iss)
        
def handle_choice_b(N):
    rocket = Simulation("Rocket", MASS_SAT, "Earth", MASS_1, RADIUS_1, 75, 0, 
                        -1.3E7, 0, 0, 7680.405, True, "Moon", MASS_2, RADIUS_2,
                        3.844E8, 0)   
    rocket.print_info_a()
    rocket.print_info_b()
    for i in range(N):
        crash_status = rocket.update()
        sys.stdout.write("{}You're {}% way through.".format("\r", (int(i*100/N) + 1)))
    
        if crash_status == True:
            print("Crashed!")
            break
    plot_distance(rocket)
    plot_energies(rocket)
    plot_threeD(rocket, True)

def handle_choice_c(N):
    wow = Three_Body("Earth", 5.972E24, 6.371E6, "Earth", 5.972E24, 6.371E6, 
                     "Earth", 5.972E24, 6.371E6, 100, 0, 10E8 + 10E12, 10E12, 
                     2500, -2500, -1*10E8 + 10E2, 10E12, 2500, -2500, 10E12, 
                     1.732*10E8 + 10E12, -2500, 0)
    for i in range(N):
        wow.update_2()
        sys.stdout.write("{}You're {}% way through.".format("\r", (int(i*100/N) + 1)))
    plot_distanceD(wow)

def MainMenu():
    choice = '0' #Initialising Variables
    while choice != 'q':
        print("\n" + 28*'=' + "\n(a)Orbits simulation \n(b)Orbit around 2 stationary")
        print("   bodies \n(c)3-Body Problem attempt  \n(q)To quit\n" + 28*"=")
        choice = (input("Please enter your choice [a-q]: ").lower())
        if choice == 'a':
            choice_a = '0'
            while choice_a != 'q':
                print (28*'=' + "\n(a)Lunar orbit simulations")
                print ("(b)ISS orbit simulation \n(q)To quit\n" + 28*"=")
                choice_a = (input("Please enter your choice [a-q]: ").lower())
                if choice_a == 'a':
                    handle_choice_aa(100000)

                elif choice_a == 'b':
                    handle_choice_ab(100)
                                        
                elif choice_a != 'q':
                    print("\nInvalid input. Choices can only be [a-q].\n")
            
        elif choice == "b":
            handle_choice_b(15000)
            
        elif choice == 'c':
            handle_choice_c(100000)
        
        elif choice != 'q':
            print("Invalid choice. Please try again.")
    
class Three_Body:
    def __init__(self, name_1, mass_1, radius_1, name_2, mass_2, radius_2, name_3,
                 mass_3, radius_3, h, t, x_1 = 0, y_1 = 0, vx_1 = 0, vy_1 = 0,
                  x_2 = 0, y_2 = 0, vx_2 = 0, vy_2 = 0,  x_3 = 0, y_3 = 0, vx_3 = 0, 
                  vy_3 = 0):
        self.name_1 = name_1
        self.mass_1 = mass_1
        self.radius_1 = radius_1
        self.name_2 = name_2
        self.mass_2 = mass_2
        self.radius_2 = radius_2
        self.name_3 = name_3
        self.mass_3 = mass_3
        self.radius_3 = radius_3
        self.h = h
        self.t = []
        self.x_1 = []
        self.y_1 = []
        self.vx_1 = []
        self.vy_1 = []
        self.x_2 = []
        self.y_2 = []
        self.vy_2 = []
        self.vx_2 = []
        self.x_3 = []
        self.y_3 = []
        self.vx_3 = []
        self.vy_3 = []
        
        self.x_1.append(x_1)		# self.x[0] == the initial x
        self.y_1.append(y_1)		# self.y[0] == the initial y
        self.vx_1.append(vx_1)	# self.v_x[0] == the initial v_x	
        self.vy_1.append(vy_1)	# self.v_y[0] == the initial v_y
        
        self.x_2.append(x_2)		# self.x[0] == the initial x
        self.y_2.append(y_2)		# self.y[0] == the initial y
        self.vx_2.append(vx_2)	# self.v_x[0] == the initial v_x	
        self.vy_2.append(vy_2)	# self.v_y[0] == the initial v_y

        self.x_3.append(x_3)		# self.x[0] == the initial x
        self.y_3.append(y_3)		# self.y[0] == the initial y
        self.vx_3.append(vx_3)	# self.v_x[0] == the initial v_x	
        self.vy_3.append(vy_3)	# self.v_y[0] == the initial v_y
        
        self.t.append(t)
        
    def f_11(self, vx_1):
        return vx_1
    
    def f_12(self, vy_1):
        return vy_1
    
    def f_13(self, x_1, y_1):
        return -G*self.mass_2*(x_1 - self.x_2[-1])/((x_1 - self.x_2[-1])**2 + 
                (y_1 - self.y_2[-1])**2)**(3/2) - G*self.mass_3*(x_1 - 
                self.x_3[-1])/((x_1 - self.x_3[-1])**2 + (y_1 - self.y_3[-1])**2)**(3/2)
    
    def f_14(self, x_1, y_1):
        return -G*self.mass_2*(y_1 - self.y_2[-1])/((x_1 - self.x_2[-1])**2 + 
                (y_1 - self.y_2[-1])**2)**(3/2) - G*self.mass_3*(y_1 - 
                self.x_3[-1])/((x_1 - self.x_3[-1])**2 + (y_1 - self.y_3[-1])**2)**(3/2)
    
    def f_21(self, vx_2):
        return vx_2
    
    def f_22(self, vy_2):
        return vy_2
    
    def f_23(self, x_2, y_2):
        return -G*self.mass_1*(x_2 - self.x_1[-1])/((x_2 - self.x_1[-1])**2 + 
                (y_2 - self.y_1[-1])**2)**(3/2) - G*self.mass_3*(x_2 - 
                self.x_3[-1])/((x_2 - self.x_3[-1])**2 + (y_2 - self.y_3[-1])**2)**(3/2)
    
    def f_24(self, x_2, y_2):
        return -G*self.mass_1*(y_2 - self.y_1[-1])/((x_2 - self.x_1[-1])**2 + 
                (y_2 - self.y_2[-1])**2)**(3/2) - G*self.mass_3*(y_2 - 
                self.y_3[-1])/((x_2 - self.x_3[-1])**2 + (y_2 - self.y_3[-1])**2)**(3/2)

    def f_31(self, vx_3):
        return vx_3
    
    def f_32(self, vy_3):
        return vy_3
    
    def f_33(self, x_3, y_3):
        return -G*self.mass_1*(x_3 - self.x_1[-1])/((x_3 - self.x_1[-1])**2 + 
                (y_3 - self.y_1[-1])**2)**(3/2) - G*self.mass_3*(x_3 - 
                self.x_2[-1])/((x_3 - self.x_2[-1])**2 + (y_3 - self.y_2[-1])**2)**(3/2)

    def f_34(self, x_3, y_3):
        return -G*self.mass_1*(y_3 - self.y_1[-1])/((x_3 - self.x_1[-1])**2 + 
               (y_3 - self.y_2[-1])**2)**(3/2) - G*self.mass_2*(y_3 - 
               self.y_2[-1])/((x_3 - self.x_2[-1])**2 + (y_3 - self.y_2[-1])**2)**(3/2)

    def improved_runge_kutta(self):
        
        k11_x = self.f_11(self.vx_1[-1])
        k11_y = self.f_12(self.vy_1[-1])
        k11_vx = self.f_13(self.x_1[-1], self.y_1[-1])
        k11_vy = self.f_14(self.y_1[-1], self.y_1[-1])
        
        k12_x = self.f_11(self.vx_1[-1] + self.h*k11_vx/2)
        k12_y = self.f_12(self.vy_1[-1] + self.h*k11_vy/2)
        k12_vx = self.f_13(self.x_1[-1] + self.h*k11_x/2, self.y_1[-1] + self.h*k11_y/2)
        k12_vy = self.f_14(self.x_1[-1] + self.h*k11_x/2, self.y_1[-1] + self.h*k11_y/2)
        
        k13_x = self.f_11(self.vx_1[-1] + self.h*k12_vx/2)
        k13_y = self.f_12(self.vy_1[-1] + self.h*k12_vy/2)
        k13_vx = self.f_13(self.x_1[-1] + self.h*k12_x/2, self.y_1[-1] + self.h*k12_y/2)
        k13_vy = self.f_14(self.x_1[-1] + self.h*k12_x/2, self.y_1[-1] + self.h*k12_y/2)
        
        k14_x = self.f_11(self.vx_1[-1] + self.h*k13_vx/2)
        k14_y = self.f_12(self.vy_1[-1] + self.h*k13_vy/2)
        k14_vx = self.f_13(self.x_1[-1] + self.h*k13_x/2, self.y_1[-1] + self.h*k13_y/2)
        k14_vy = self.f_14(self.x_1[-1] + self.h*k13_x/2, self.y_1[-1] + self.h*k13_y/2)
        
        self.x_1.append(self.x_1[-1] + self.h/6*(k11_x + 2*k12_x + 2*k13_x + k14_x))
        self.y_1.append(self.y_1[-1] + self.h/6*(k11_y + 2*k12_y + 2*k13_y + k14_y))
        self.vx_1.append(self.vx_1[-1] + self.h/6*(k11_vx + 2*k12_vx + 2*k13_vx + k14_vx))
        self.vy_1.append(self.vy_1[-1] + self.h/6*(k11_vy + 2*k12_vy + 2*k13_vy + k14_vy))
        
        
        k21_x = self.f_21(self.vx_2[-1])
        k21_y = self.f_22(self.vy_2[-1])
        k21_vx = self.f_23(self.x_2[-1], self.y_2[-1])
        k21_vy = self.f_24(self.y_2[-1], self.y_2[-1])
        
        k22_x = self.f_21(self.vx_2[-1] + self.h*k21_vx/2)
        k22_y = self.f_22(self.vy_2[-1] + self.h*k21_vy/2)
        k22_vx = self.f_23(self.x_2[-1] + self.h*k21_x/2, self.y_2[-1] + self.h*k21_y/2)
        k22_vy = self.f_24(self.x_2[-1] + self.h*k21_x/2, self.y_2[-1] + self.h*k21_y/2)
        
        k23_x = self.f_21(self.vx_2[-1] + self.h*k22_vx/2)
        k23_y = self.f_22(self.vy_2[-1] + self.h*k22_vy/2)
        k23_vx = self.f_23(self.x_2[-1] + self.h*k22_x/2, self.y_2[-1] + self.h*k22_y/2)
        k23_vy = self.f_24(self.x_2[-1] + self.h*k22_x/2, self.y_2[-1] + self.h*k22_y/2)

        k24_x = self.f_21(self.vx_2[-1] + self.h*k23_vx/2)
        k24_y = self.f_22(self.vy_2[-1] + self.h*k23_vy/2)
        k24_vx = self.f_23(self.x_2[-1] + self.h*k23_x/2, self.y_2[-1] + self.h*k23_y/2)
        k24_vy = self.f_24(self.x_2[-1] + self.h*k23_x/2, self.y_2[-1] + self.h*k23_y/2)

        self.x_2.append(self.x_2[-1] + self.h/6*(k21_x + 2*k22_x + 2*k23_x + k24_x))
        self.y_2.append(self.y_2[-1] + self.h/6*(k21_y + 2*k22_y + 2*k23_y + k24_y))
        self.vx_2.append(self.vx_2[-1] + self.h/6*(k21_vx + 2*k22_vx + 2*k23_vx + k24_vx))
        self.vy_2.append(self.vy_2[-1] + self.h/6*(k21_vy + 2*k22_vy + 2*k23_vy + k24_vy))
        
        
        k31_x = self.f_31(self.vx_3[-1])
        k31_y = self.f_32(self.vy_3[-1])
        k31_vx = self.f_33(self.x_3[-1], self.y_3[-1])
        k31_vy = self.f_34(self.y_3[-1], self.y_3[-1])

        k32_x = self.f_31(self.vx_3[-1] + self.h*k31_vx/2)
        k32_y = self.f_32(self.vy_3[-1] + self.h*k31_vy/2)
        k32_vx = self.f_33(self.x_3[-1] + self.h*k31_x/2, self.y_3[-1] + self.h*k31_y/2)
        k32_vy = self.f_34(self.x_3[-1] + self.h*k31_x/2, self.y_3[-1] + self.h*k31_y/2)
        
        k33_x = self.f_31(self.vx_3[-1] + self.h*k32_vx/2)
        k33_y = self.f_32(self.vy_3[-1] + self.h*k32_vy/2)
        k33_vx = self.f_33(self.x_3[-1] + self.h*k32_x/2, self.y_3[-1] + self.h*k32_y/2)
        k33_vy = self.f_34(self.x_3[-1] + self.h*k32_x/2, self.y_3[-1] + self.h*k32_y/2)
        
        k34_x = self.f_31(self.vx_3[-1] + self.h*k33_vx/2)
        k34_y = self.f_32(self.vy_3[-1] + self.h*k33_vy/2)
        k34_vx = self.f_33(self.x_3[-1] + self.h*k33_x/2, self.y_3[-1] + self.h*k33_y/2)
        k34_vy = self.f_34(self.x_3[-1] + self.h*k33_x/2, self.y_3[-1] + self.h*k33_y/2)
        
        self.x_3.append(self.x_3[-1] + self.h/6*(k31_x + 2*k32_x + 2*k33_x + k34_x))
        self.y_3.append(self.y_3[-1] + self.h/6*(k31_y + 2*k32_y + 2*k33_y + k34_y))
        self.vx_3.append(self.vx_3[-1] + self.h/6*(k31_vx + 2*k32_vx + 2*k33_vx + k34_vx))
        self.vy_3.append(self.vy_3[-1] + self.h/6*(k31_vy + 2*k32_vy + 2*k33_vy + k34_vy))
        
        self.t.append(self.t[-1] + self.h)
        
    def update_2(self):
        self.improved_runge_kutta()
MainMenu()