import rosbag
import tf
import math
import numpy as np
from scipy.stats import mvn
from scipy.stats import norm
import matplotlib.mlab as mlab

degreeQuantum = math.pi/2.0
pos = [12, 28]
heading = math.radians(200.52)
noise = 0.1
gridQuantum = 0.2

numCells = 35

prevGridProb = [[0.0 for i in range(numCells)] for j in range(numCells)]
prevGridProb[11][27]  = 1.0
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
    cordsX = np.array([[0.0 for j in range(3)] for i in range(3)])
    cordsY = np.array([[0.0 for j in range(3)] for i in range(3)])
    p = np.array([[0.0 for j in range(3)] for i in range(3)])

    X = np.array(map(lambda temp: temp*gridQuantum + gridQuantum/2.0, [cur[0]-1, cur[0], cur[0]+1]))
    Y = np.array(map(lambda temp: temp*gridQuantum + gridQuantum/2.0, [cur[1]-1, cur[1], cur[1]+1]))

    for i in range(3):
        for j in range(3):
            cordsX[i][j] = -(X[i]-x)*math.sin(-rot) + (Y[j]-y)*math.cos(-rot)
            cordsY[i][j] = -1*((X[i]-x)*math.cos(-rot) + (Y[j]-y)*math.sin(-rot))

    Z = mlab.bivariate_normal(cordsX, cordsY, 0.1, 0.1, 0.0, 0.0)
    Z /= Z.sum()

    for i in range(3):
        for j in range(3):
            try:
                if X[i] < 7.0 and Y[j] < 7.0:
                    gridProb[int(X[i]/gridQuantum)][int(Y[j]/gridQuantum)] += prior*Z[i][j]
            except:
                print X[i]
                print Y[j]
            if prior > 0:
                print 'prior:', prior, Z[i][j]
                print gridProb[int(X[i]/gridQuantum)][int(Y[j]/gridQuantum)]

    
def handle_movement(message):
    pAngle = dict(zip(np.linspace(-2.5, 2.5, 6), norm.cdf(np.linspace(-2.5, 2.5, 6), 0, 0.5)))
    gridProb = np.array([[0.0 for j in range(numCells)] for i in range(numCells)])

    for row in range(numCells):
        for col in range(numCells):
            pos = (row, col)
            
            for angle in range(-2, 3):
                p = pAngle[angle+0.5] - pAngle[angle-0.5]
                rot1 = quaternionToEuler(message.rotation1)
                rot2 = quaternionToEuler(message.rotation2)


                rot1 = angle*degreeQuantum + rot1
                x = ((pos[0]*gridQuantum) + message.translation*math.cos(heading + rot1))
                y = ((pos[1]*gridQuantum) + message.translation*math.sin(heading + rot1))

                if angle == 0:
                    print x,y
                    print int(x/gridQuantum), int(y/gridQuantum)

                calculateNeighbourProb(x, y, heading+rot1, prevGridProb[row][col]*p, gridProb)

    return (gridProb, heading+rot1+rot2)

def handle_observation(message, gridProb, heading):
    bearing = quaternionToEuler(message.bearing)
    sample = np.random.normal(0.0, 0.1)
    mean = message.range - sample
    theta = bearing+heading

    pObs = dict(zip([-0.3, -0.1, 0.1, 0.3], norm.cdf([-0.3, -0.1, 0.1, 0.3], 0, 0.5)))

    print pObs
    y = tags[message.tagNum]['y'] - mean*math.sin(theta)
    x = tags[message.tagNum]['x'] - mean*math.cos(theta)
    gridProb[int(x/gridQuantum)][int(y/gridQuantum)] *= pObs[0.1] - pObs[-0.1]

    y1 = tags[message.tagNum]['y'] - mean*math.sin(theta) - 0.1
    x1 = tags[message.tagNum]['x'] - mean*math.cos(theta) - 0.1
    gridProb[int(x1/gridQuantum)][int(y1/gridQuantum)] *= pObs[-0.1] - pObs[-0.3]

    y2 = tags[message.tagNum]['y'] - mean*math.sin(theta) + 0.1
    x2 = tags[message.tagNum]['x'] - mean*math.cos(theta) + 0.1
    gridProb[int(x2/gridQuantum)][int(y2/gridQuantum)] *= pObs[0.3] - pObs[0.1]

    print gridProb

def readMessages(filename):
    bag = rosbag.Bag(filename)
    gridProb = np.array([])
    c = 0
    for message in bag.read_messages():
        c += 1
        if message[0] == 'Movements':
            gridProb, heading = handle_movement(message[1])
        else:
            handle_observation(message[1], gridProb, heading)
        if c == 3:
            break;

if __name__ == '__main__':
    np.set_printoptions(threshold=np.nan)
    readMessages('grid.bag')

