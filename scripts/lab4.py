import rosbag
import tf
import math
import numpy as np
from scipy.stats import mvn
from scipy.stats import norm

degreeQuantum = 90
pos = [12, 28]
heading = 200.52/degreeQuantum
noise = 0.1
gridQuantum = 0.2

prevGridProb = [[0 for i in range(35)] for j in range(35)]
path = []
tags = {0:{'x': 1.25, 'y': 5.25}, 1:{'x': 1.25,'y':3.25}, 2:{'x': 1.25,'y':1.25}, 3:{'x': 4.25,'y':1.25}, 4:{'x': 4.25,'y':3.25}, 5:{'x': 4.25,'y':5.25}}

def quaternionToEuler(quaternion):
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w
    (roll,pitch,yaw) = tf.transformations.euler_from_quaternion((x, y, z, w))

    return yaw

def toRad(angle):
    return math.radians(angle*degreeQuantum)

def calculateNeighbourProb(x, y, rot, prior, gridProb):
    cur = (int(x/gridQuantum), int(y/gridQuantum))
    neighbourCell = np.array([[(0, 0) for i in range(3)] for j in range(3)])
    prob = np.array([[0.0 for i in range(3)] for j in range(3)])

    for i in range(3):
        for j in range(3):
            neighbourCell[i][j] = np.array(((cur[0] + (i-1))*gridQuantum + (gridQuantum/2.0), (cur[1] + (j-1))*gridQuantum + (gridQuantum/2.0)))
    neighbourCell[1][1] = (x, y)

    corners = np.array([[(cur[0]*gridQuantum, cur[1]*gridQuantum + gridQuantum),
							 (cur[0]*gridQuantum + gridQuantum, cur[1]*gridQuantum + gridQuantum)],
						 [(cur[0]*gridQuantum, cur[1]*gridQuantum),
						 	 (cur[0]*gridQuantum + gridQuantum, cur[1]*gridQuantum)]])

    rotMat = np.array([[math.cos(-rot), math.sin(-rot)], [-math.sin(-rot), math.cos(-rot)]])

    for i in range(2):
		for j in range(2):
			corners[i][j] = np.dot(rotMat, corners[i][j]) - (x, y)

    for i in range(3):
        for j in range(3):
            neighbourCell[i][j] = np.dot(rotMat, neighbourCell[i][j]) - (x, y)

    
    sigma = np.array([0.1, 0.1])
    cov = np.diag(sigma**2)

    mu = np.array([0.0, 0.0])
    '''
    print neighbourCell[0][0]
    print corners[0][0]
    print mu
    print cov
    '''
    prob[0][0], t = mvn.mvnun(neighbourCell[0][0], corners[0][0], mu, cov)
    prob[0][1], t = mvn.mvnun(neighbourCell[0][1], (corners[0][0] + corners[0][1])/2.0, mu, cov)
    prob[0][2], t = mvn.mvnun(corners[0][1], neighbourCell[0][2], mu, cov)

    prob[1][0], t = mvn.mvnun(neighbourCell[1][0], (corners[0][0] + corners[1][0])/2.0, mu, cov)
    prob[1][1], t = mvn.mvnun(corners[1][0], corners[0][1], mu, cov)
    prob[1][0], t = mvn.mvnun((corners[0][1] + corners[1][1])/2.0, neighbourCell[1][0], mu, cov)

    prob[2][0], t = mvn.mvnun(neighbourCell[2][0], corners[1][0], mu, cov)
    prob[2][1], t = mvn.mvnun(neighbourCell[2][1], (corners[1][0] + corners[1][1])/2.0, mu, cov)
    prob[2][2], t = mvn.mvnun(corners[1][1], neighbourCell[2][2], mu, cov)

    print prob
    try:
        prob /= sum(prob)
    except:
        print prob

    for i in range(3):
        for j in range(3):
            if (cur[0]-i-1 >= 0) and (cur[0]-i-1 < 35) and (cur[1]-j-1 >= 0) and (cur[1]-j-1 < 35):
                gridProb[cur[0] - i - 1][cur[1] - j - 1] += prior*prob[i][j]

    
def handle_movement(message):
	pAngle = dict(zip(np.linspace(-2.5, 2.5, 6), norm.cdf(np.linspace(-2.5, 2.5, 6), 0, 0.5)))
	gridProb = np.array([[0 for j in range(35)] for i in range(35)])

	for row in range(35):
		for col in range(35):
			pos = (row, col)
			
			for i in range(-2, 3):
				p = pAngle[i+0.5] - pAngle[i-0.5]
				rot1 = quaternionToEuler(message.rotation1)
				rot2 = quaternionToEuler(message.rotation2)

				rot1 = i*degreeQuantum + rot1
				x = ((pos[0]*gridQuantum) + message.translation*math.cos(heading + rot1))
				y = ((pos[1]*gridQuantum) + message.translation*math.sin(heading + rot1))

				calculateNeighbourProb(x, y, rot1, prevGridProb[row][col]*p, gridProb)
	return (gridProb, heading+rot1+rot2)

def handle_observation(message, gridProb, heading):
	bearing = quaternionToEuler(message.bearing)
	sample = np.random.normal(0.0, 0.1)
	mean = message.range - sample
	theta = bearing+heading

	pObs = dict(zip(np.array([-0.3, -0.1, 0.1, 0.3]), norm.cdf(np.array([-0.3, -0.1, 0.1, 0.3]), 0, 0.5)))
	
	y = tags[message.tagNum]['y'] - mean*math.sin(theta)
	x = tags[message.tagNum]['x'] - mean*math.cos(theta)
	gridProb[int(x/gridQuantum)][int(y/gridQuantum)] *= pObs[2] - pObs[1]

	y1 = tags[message.tagNum]['y'] - mean*math.sin(theta) - 0.1
	x1 = tags[message.tagNum]['x'] - mean*math.cos(theta) - 0.1
	gridProb[int(x1/gridQuantum)][int(y1/gridQuantum)] *= pObs[1] - pObs[0]

	y2 = tags[message.tagNum]['y'] - mean*math.sin(theta) + 0.1
	x2 = tags[message.tagNum]['x'] - mean*math.cos(theta) + 0.1
	gridProb[int(x2/gridQuantum)][int(y2/gridQuantum)] *= pObs[3] - pObs[2]



def readMessages(filename):
    bag = rosbag.Bag(filename)
    gridProb = np.array([])

    for message in bag.read_messages():
        if message[0] == 'Movements':
            gridProb, heading = handle_movement(message[1])
        else:
            handle_observation(message[1], gridProb, heading)


if __name__ == '__main__':
    readMessages('grid.bag')

