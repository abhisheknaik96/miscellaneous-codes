###########################################################
# Author	: Abhishek Naik
# 
# What 		: Implements a multi-dimensional kalman filter
# Why		: For CS373, Udacity's MOOC on 'AI for Robotics'
############################################################

import numpy as np 

numDims = 2

######################## 1-D data ###########################

if numDims == 1:

	measurements = [1, 2, 3]
	dT = 1 							# assuming $\delta t$ to be 1 

	x = np.array([[0],[0]])				# initial state (position and velocity)
	P = np.matrix([[100,0],[0,100]])	# initial covariance matrix (uncorrelated uncertainty)
	u = np.array([[0],[0]])				# initial motion
	F = np.matrix([[1,dT],[0,1]])		# next state function
	H = np.matrix([1,0])				# measurement function
	R = np.matrix([[1]])				# measurement uncertainty
	I = np.identity(2)

######################## 2-D data ###########################

elif numDims == 2:

	measurements = [[5., 10.], [6., 8.], [7., 6.], [8., 4.], [9., 2.], [10., 0.]]
	initial_xy = [4., 12.]
	dT = 0.1 											

	x = np.array([[initial_xy[0]], [initial_xy[1]], [0], [0]])		# initial state (position and velocity)
	u = np.array([[0],[0],[0],[0]])									# initial motion

	P = np.matrix([[0,0,0,0],[0,0,0,0],[0,0,100,0],[0,0,0,100]])	# initial covariance matrix (uncorrelated uncertainty)
	F = np.matrix([[1,0,dT,0],[0,1,0,dT],[0,0,1,0],[0,0,0,1]])		# next state function
	H = np.matrix([[1,0,0,0],[0,1,0,0]])							# measurement function
	R = np.matrix([[0.1,0],[0,0.1]])								# measurement uncertainty
	I = np.identity(4)


def kalman_filter(x, P):
    
    for z in measurements:
        
        z = np.matrix(z)
        # prediction
        x = F * x + u 								# x = F.x + u
        P = F * P * np.transpose(F)					# P = F.P.F'
        
        # measurement update
        y = np.transpose(z) - H * x					# y = z - H.x
        S = H * P * np.transpose(H) + R				# S = H.P.H' + R
        K = P * np.transpose(H) * np.linalg.inv(S)	# K = P.H.S^{-1}

        x = x + K * y								# x = x + K.y
        P = (I - K * H) * P 						# P = (I-K.H).P

        # print('x = ', x)
        # print('P = ', P)

    return x,P

print(kalman_filter(x, P))

