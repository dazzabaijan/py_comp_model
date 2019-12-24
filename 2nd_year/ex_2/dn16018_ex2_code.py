"""
Name: Darren Ng - Email: dn16018@bristol.ac.uk
Student No: 1631747
Assignment 2: Fresnel diffraction from an aperture - Simpson's rule
"""

import argparse
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import time

EPSILON = 0.000000000008854
C = 300000000 

class Integration():
	def __init__(self, type, lowerLimit = None, upperLimit = None, n = None):
		self.type = type 
		self.lowerLimit = lowerLimit
		self.upperLimit = upperLimit
		self.n = n
		self.function = None
		self.function_str = None

	def print_info(self):
		print("")
		print("[INFO] Integration type      : {}".format(self.type))
		print("[INFO] Function to integrate : {}".format(self.function_str))
		print("[INFO] Lower Limit           : {}".format(self.lowerLimit))
		print("[INFO] Upper Limit           : {}".format(self.upperLimit))
		print("[INFO] Value of N            : {}".format(self.n))
		print("")


class LimitException(Exception):
	pass


class NValueException(Exception):
	pass


class FunctionException(Exception):
	pass


def partA(args):
	print("Solving Part A:")

	io = getIntegrationObj(args)
	io.print_info()
	integrationResult = simpsonsRule(io.function, io.lowerLimit, io.upperLimit, io.n)

	print("[RESULT] The value of integration is: {}".format(integrationResult))
	print("")
	print("Finished Part A.")
	return


def partB(args):
	print("Solving Part B:")

	WAVELENGTH = float(args.wavelength)						# get the wavelength from argparse
	Z = float(args.z)										# get the z value from argparse
	NUM_OF_POINTS = int(args.number_of_points)				# get the number of points from argparse
	NUM_OF_ITERATIONS = int(args.number_of_iterations)		# get the number of iterations (N) from argparse
	K = (2*np.pi) / WAVELENGTH
	CONST = (1.0j*K) / (2*Z)

	# aperture size
	X_MIN = -0.1E-4       
	X_MAX = 0.1E-4

	# screen size
	X_SCREEN_MIN = -0.5E-2     
	X_SCREEN_MAX = 0.5E-2

	dx = (X_SCREEN_MAX - X_SCREEN_MIN) / (NUM_OF_POINTS - 1)
	xVals = np.zeros(NUM_OF_POINTS, dtype = np.float64)
	yVals = np.zeros(NUM_OF_POINTS, dtype = np.float64)

	print("[INFO] Wavelength\t\t: {}".format(WAVELENGTH))
	print("[INFO] k\t\t\t: {}".format(K))
	print("[INFO] z\t\t\t: {}".format(Z))
	print("[INFO] Const\t\t\t: {}".format(CONST))
	print("[INFO] Iterations (N)\t\t: {}".format(NUM_OF_ITERATIONS))
	print("")
	print("[INFO] Lower Limit\t\t: {}".format(X_MIN))
	print("[INFO] Upper Limit\t\t: {}".format(X_MAX))
	print("[INFO] Min Screen Size\t\t: {}".format(X_SCREEN_MIN))
	print("[INFO] Max Screen Size\t\t: {}".format(X_SCREEN_MAX))
	print("[INFO] Delta (dx)\t\t: {}".format(dx))
	print("")

	for i in range(NUM_OF_POINTS):
		xVals[i] = X_SCREEN_MIN + (i * dx)
		c = simpsonsRule(expFunction, X_MIN, X_MAX, NUM_OF_ITERATIONS, xVals[i], CONST)
		yVals[i] = (np.absolute(c))**2

	fig, ax1 = plt.subplots()
	ax1.plot(xVals,yVals)
	ax1.set_ylabel("Intensity")
	ax1.set_xlabel("x")
	plt.suptitle("Graph of Intensity vs x-coordinates")
	plt.savefig("fresnel1D.png",dpi=300)
	plt.show()
	print("Finished Part B.")
	return 


def partC(args):
	print("Solving Part C:")
	
	SHAPES = {
		"square": SquareAperture,
		"circular": CircularAperture,
		"triangular": TriangularAperture
	}

	WAVELENGTH = float(args.wavelength)					# get the wavelength from argparse
	Z = float(args.z)									# get the z value from argparse
	NUM_OF_POINTS = int(args.number_of_points)			# get the number of points from argparse
	NUM_OF_ITERATIONS = int(args.number_of_iterations)	# get the number of iterations (N) from argparse
	RADIUS = float(args.radius)							# get the radius from argparse
	HEIGHT = float(args.height)							# get the height from argparse

	K = (2*np.pi) / WAVELENGTH
	CONST = (1.0j*K) / (2*Z)
	CONST_2 = (K)/(2*np.pi*Z)

	problem = SHAPES[args.shape.strip().lower()](WAVELENGTH, Z, NUM_OF_POINTS, NUM_OF_ITERATIONS, K, CONST, CONST_2, RADIUS, HEIGHT)
	print("Finished Part C.")
	return

# end of Part C

def SquareAperture(WAVELENGTH, Z, NUM_OF_POINTS, NUM_OF_ITERATIONS, K, CONST, CONST_2, RADIUS = None, HEIGHT = None):
	startTime = time.time()  

	X_MIN = -10E-5        #aperture size
	X_MAX = 10E-5        
	Y_MIN = -10E-5
	Y_MAX = 10E-5
	SCREEN_MIN = -9E-5    #screen size in x and y direction
	SCREEN_MAX = 9E-5     

	dxy = (SCREEN_MAX - SCREEN_MIN) / (NUM_OF_POINTS)
	xVals = np.zeros(NUM_OF_POINTS, dtype = np.float64)
	yVals = np.zeros(NUM_OF_POINTS, dtype = np.float64)

	print("[INFO] Wavelength\t\t: {}".format(WAVELENGTH))
	print("[INFO] k\t\t\t: {}".format(K))
	print("[INFO] z\t\t\t: {}".format(Z))
	print("[INFO] Const 1\t\t\t: {}".format(CONST))
	print("[INFO] Const 2\t\t\t: {}".format(CONST_2))
	print("[INFO] n\t\t\t: {}".format(NUM_OF_ITERATIONS))
	print("")
	print("[INFO] x Lower Limit\t\t: {}".format(X_MIN))
	print("[INFO] x Upper Limit\t\t: {}".format(X_MAX))
	print("[INFO] y Lower Limit\t\t: {}".format(Y_MIN))
	print("[INFO] y Upper Limit\t\t: {}".format(Y_MAX))
	print("[INFO] Min Screen Size\t\t: {}".format(SCREEN_MIN))
	print("[INFO] Max Screen Size\t\t: {}".format(SCREEN_MAX))
	print("[INFO] Screen delta (dx | dy)\t: {}".format(dxy))
	print("[INFO] Aperture Shape\t\t: {}".format("Square"))
	print("")

	intensityArray = np.zeros((NUM_OF_POINTS, NUM_OF_POINTS))
	counter = 0

	for j in range(NUM_OF_POINTS):
		yVals[j] = SCREEN_MIN + (j * dxy)

		for i in range(NUM_OF_POINTS):
			xVals[i] = SCREEN_MIN + (i * dxy)

			eValue = CONST_2 * simpsonsRule(expFunction, X_MAX, X_MIN, NUM_OF_ITERATIONS, xVals[i], CONST) * simpsonsRule(expFunction, Y_MAX, Y_MIN, NUM_OF_ITERATIONS, yVals[j], CONST)
			intensityArray[i,j] = C * EPSILON * np.real((eValue * eValue.conjugate()))


			counter += 1
			progress = round((counter/NUM_OF_POINTS))
			timeElapsed = time.time() - startTime
			sys.stdout.write("\r" + "You are " + str(progress) + "% of the way through the calculation. Time elapsed: {} second(s)".format(timeElapsed))
	
	fig, ax1 = plt.subplots()
	ax1.imshow(intensityArray)
	ax1.set_xlabel("x")
	ax1.set_ylabel("y")
	plt.suptitle("Graph of Intensity vs x and y coordinates")
	cbar = plt.colorbar(ax1.imshow(intensityArray))
	cbar.set_label('Intensity', rotation=270)
	plt.savefig("SquareDiffPattern.png",dpi=300)
	plt.show()

	return


def CircularAperture(WAVELENGTH, Z, NUM_OF_POINTS, NUM_OF_ITERATIONS, K, CONST, CONST_2, RADIUS, HEIGHT = None):
	startTime = time.time()

	Y_MIN = -RADIUS      #aperture size   
	SCREEN_MIN = -9E-5 
	SCREEN_MAX = 9E-5    #screen size in x and y direction

	dxy = (SCREEN_MAX - SCREEN_MIN) / NUM_OF_POINTS
	dyInc = (2*RADIUS) / NUM_OF_POINTS
	xVals = np.zeros(NUM_OF_POINTS, dtype = np.float64)
	yVals = np.zeros(NUM_OF_POINTS, dtype = np.float64)

	intensityArray = np.zeros((NUM_OF_POINTS, NUM_OF_POINTS))

	eValue = 0
	counter = 1

	for j in range(NUM_OF_POINTS):
		progress = round(counter/NUM_OF_POINTS)
		sys.stdout.write("\r" + "You are " + str(progress) + "% of the way through the calculation.")
		yVals[j] = SCREEN_MIN + (j * dxy)

		for i in range(NUM_OF_POINTS):
			xVals[i] = SCREEN_MIN + (i * dxy)

			for k in range(NUM_OF_POINTS):
				yMinStrip = Y_MIN + (k * dyInc)
				yMaxStrip = yMinStrip + dyInc
				xMin = -1*np.sqrt(RADIUS**2-yMinStrip**2)
				xMax = -1*xMin
				
				eValue += CONST_2 * simpsonsRule(expFunction, xMax, xMin, NUM_OF_ITERATIONS, xVals[i], CONST) * simpsonsRule(expFunction, yMaxStrip, yMinStrip, NUM_OF_ITERATIONS, yVals[j], CONST)
	
			intensityArray[i,j] = C * EPSILON * np.real((eValue * eValue.conjugate()))
			eValue = 0

			counter += 1
			timeElapsed = time.time() - startTime
			sys.stdout.write("\r" + "You are " + str(progress) + "% of the way through the calculation. Time elapsed: {} second(s)".format(timeElapsed))

	fig, ax1 = plt.subplots()
	ax1.imshow(intensityArray)
	ax1.set_xlabel("x")
	ax1.set_ylabel("y")
	plt.suptitle("Graph of Intensity vs x and y coordinates")
	cbar = plt.colorbar(ax1.imshow(intensityArray))
	cbar.set_label('Intensity', rotation=270)
	plt.savefig("CircularDiffPattern.png",dpi=300)
	plt.show()
	return 


def TriangularAperture(WAVELENGTH, Z, NUM_OF_POINTS, NUM_OF_ITERATIONS, K, CONST, CONST_2, RADIUS, HEIGHT):
	startTime = time.time()

	Y_MIN = 0
	Y_MAX = HEIGHT                         #aperture size   

	SCREEN_MIN = -17E-5 
	SCREEN_MAX = 17E-5    #screen size in x and y direction

	dyInc = (Y_MAX) / (NUM_OF_POINTS - 1)
	dxy = (SCREEN_MAX - SCREEN_MIN) / (NUM_OF_POINTS)

	xVals = np.zeros(NUM_OF_POINTS, dtype = np.float64)
	yVals = np.zeros(NUM_OF_POINTS, dtype = np.float64)
	intensityArray = np.zeros((NUM_OF_POINTS, NUM_OF_POINTS))

	eValue = 0
	counter = 1

	for j in range(NUM_OF_POINTS):
		progress = round((counter/NUM_OF_POINTS))
		yVals[j] = SCREEN_MIN + (j * dxy)

		for i in range(NUM_OF_POINTS):
			xVals[i] = SCREEN_MIN + (i * dxy)

			for k in range(NUM_OF_POINTS):               
				yMinStrip = Y_MIN + (k * dyInc)
				yMaxStrip = yMinStrip + dyInc
				xMin = -(Y_MAX - yMinStrip) / (np.sqrt(3))             
				xMax = -xMin

				eValue += CONST_2 * simpsonsRule(expFunction, yMaxStrip, yMinStrip, NUM_OF_ITERATIONS, xVals[i], CONST) * simpsonsRule(expFunction, xMax, xMin, NUM_OF_ITERATIONS, yVals[j], CONST)

			intensityArray[i,j] = C * EPSILON * np.real((eValue * eValue.conjugate()))

			eValue = 0
			counter += 1       
			timeElapsed = time.time() - startTime
			sys.stdout.write("\r" + "You are " + str(progress) + "% of the way through the calculation. Time elapsed: {} second(s)".format(timeElapsed))

	fig, ax1 = plt.subplots()
	ax1.imshow(intensityArray)
	ax1.set_xlabel("x")
	ax1.set_ylabel("y")
	plt.suptitle("Graph of Intensity vs x and y coordinates")
	cbar = plt.colorbar(ax1.imshow(intensityArray))
	cbar.set_label('Intensity', rotation=270)
	plt.savefig("TriangularDiffPattern.png",dpi=300)
	plt.show()
	return


def simpsonsRule(func, a, b, N, xValue = None, CONST = None):
	h = (b - a) / N

	if xValue is not None and CONST is not None:
		c = func(xValue, a, CONST) + func(xValue, b, CONST)		#Adding first & last term

	else:
		c = func(a) + func(b)          									#Adding first & last term

	# For each loop, adds each consecutive strip with odd heights
	for i in np.arange(1, N, 2):
		if xValue is not None and CONST is not None:
			c += 4*func(xValue, a + i * h, CONST)
			continue

		c += 4*func(a + i * h)

	# For each loop, adds each consecutive strip with odd heights
	for i in np.arange(2, N-1, 2):
		if xValue is not None and CONST is not None:
			c += 2*func(xValue, a + i * h, CONST)
			continue

		c += 2*func(a + i * h)

	d = h * (c/3)
	return d


def expFunction(xValue, xPrime, CONST):
	return np.exp(CONST*(xValue-xPrime)**2)


def checkNValue(n):
	if (n <= 0):
		raise NValueException("The minimum number of N is 2.")

	if (n % 2 != 0):
		raise NValueException("N needs to be an even number for the integral to compute.")

	return n


def checkLimits(lowerLimit, upperLimit):
	lowerLimitFloat = 0.0
	upperLimitFloat = 0.0

	if lowerLimit.upper() == "PI": lowerLimit = math.pi 
	if upperLimit.upper() == "PI": upperLimit = math.pi

	lowerLimitFloat = float(lowerLimit)
	upperLimitFloat = float(upperLimit)

	if lowerLimitFloat >= upperLimitFloat:
		raise LimitException("The upper limit must be bigger than the lower limit.")

	return lowerLimitFloat, upperLimitFloat


def checkFunction(function):
	if "SIN" in function.upper():
		return lambda x:np.sin(x)

	if "COS" in function.upper():
		return lambda x:np.cos(x)

	if "EXP" in function.upper():
		return lambda x:np.exp(x)

	raise FunctionException("Only sin(x), cos(x) and exp(x) is supported.")


def getIntegrationObj(args):
	integration = Integration(args.which)

	if args.which == "b" or args.which == "c":
		return integration

	# check and validate lower_limit and upper_limit passed through command line argument
	lowerLimit, upperLimit = checkLimits(args.lower_limit, args.upper_limit)
	integration.lowerLimit = lowerLimit
	integration.upperLimit = upperLimit

	# check and validate N passed through command line argument
	integration.n = checkNValue(args.number_of_iterations)

	# check and validate type of function passed through command line argument
	integration.function = checkFunction(args.function)
	integration.function_str = "{}(x)".format(args.function.strip().lower())

	return integration


def buildParser():
	global parser

	# initialize argparse object
	parser = argparse.ArgumentParser("python fresnel.py")

	# creating subparser for each part of the questions
	subparsers = parser.add_subparsers(title = "Choose the question to solve.")

	# initializing subparser for each part of question
	a_parser = subparsers.add_parser("a", help = "Part A")
	a_parser.set_defaults(which="a")
	b_parser = subparsers.add_parser("b", help = "Part B")
	b_parser.set_defaults(which="b")
	c_parser = subparsers.add_parser("c", help = "Part C")
	c_parser.set_defaults(which="c")

	# adding arguments/options relevant to Part A of question
	a_parser.add_argument("-l", dest = "lower_limit" ,help= "Lower limit of integration", required = True)
	a_parser.add_argument("-u", dest = "upper_limit", help= "Upper limit of integration", required = True)
	a_parser.add_argument("-n", dest = "number_of_iterations", help= "Number of iterations (N)", required = True, type=int)
	a_parser.add_argument("-f", dest = "function", help= "Type of function: sin, cos, exp", required = True)
	
	# adding arguments/options relevant to Part B of question
	b_parser.add_argument("-z", dest = "z" ,help= "Value of z", default = float(1e-1), type = float)
	b_parser.add_argument("-n", dest = "number_of_iterations", help= "Number of iterations (N)", default = 50, type = int)
	b_parser.add_argument("-p", dest = "number_of_points", help= "Number of points", default = 100, type=int)
	b_parser.add_argument("-w", dest = "wavelength", help= "Wavelength", default = float(500e-9), type=float)

	# adding arguments/options relevant to Part C of question
	c_parser.add_argument("-z", dest = "z" ,help= "Value of z", default = float(1e-1), type = float)
	c_parser.add_argument("-n", dest = "number_of_iterations", help= "Number of iterations (N)", default = 50, type = int)
	c_parser.add_argument("-p", dest = "number_of_points", help= "Number of points", default = 100, type=int)
	c_parser.add_argument("-w", dest = "wavelength", help= "Wavelength", default = float(500e-9), type=float)
	c_parser.add_argument("-s", dest = "shape", help= "Shape of aperture: square, circular, triangular", default = "square", type=str)
	c_parser.add_argument("-y", dest = "height", help= "Height", default = float(2e-4), type=float)
	c_parser.add_argument("-r", dest = "radius", help= "Radius", default = float(10e-5), type=float)

	print("")


def main():

	# define function pointers to function to handle different questions
	QUESTIONS = {
		"a": partA,
		"b": partB,
		"c": partC
	}

	# helper function to build parser object using argparse to process user input
	buildParser()

	# try to process user input using argparse object. If any error is encountered,
	# it will throw an exception, print out the help messages and exit the program.
	try:
		args = parser.parse_args()

		# target the appropriate function to handle the part of question
		question = QUESTIONS[args.which]

		# call function to handle part of question
		# args object is passed in and will be further utilized in function
		question(args)

		print("")
		print("[DONE] Exiting program.")
		print("")
		
	except LimitException as e:
		print("[ERROR] {}".format(e))
		print("")

	except NValueException as e:
		print("[ERROR] {}".format(e))
		print("")

	except:
		parser.print_help()
		sys.exit(0)


if __name__== "__main__":

	# This is the program entry point
	parser = None	# initialize a global parser object
	main()			# calls main() function which does the heavy work.