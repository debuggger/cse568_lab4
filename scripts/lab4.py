import rosbag
import tf
import math
import numpy as np
from scipy.stats import mvn
from scipy.stats import norm
import matplotlib.mlab as mlab

degreeQuantum = math.pi/2.0
pos = [12, 28]
heading = 200.52
noise = 0.1
gridQuantum = 0.2

numCells = 35

prevGridProb = np.zeros((4,35,35))
path = []
landmarks = {0:{'x': 1.25, 'y': 5.25}, 1:{'x': 1.25,'y':3.25}, 2:{'x': 1.25,'y':1.25}, 3:{'x': 4.25,'y':1.25}, 4:{'x': 4.25,'y':3.25}, 5:{'x': 4.25,'y':5.25}}
angleProb = {0: 0.68268949213708585, 1: 0.15730535589982697, 2: 0.0013496113800581799, 3: 0.15730535589982697}

def quaternionToEuler(quaternion):
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w
    (roll,pitch,yaw) = tf.transformations.euler_from_quaternion((x, y, z, w))

    return math.degrees(yaw)

def cos(deg):
    return math.cos(math.radians(deg))

def sin(deg):
    return math.sin(math.radians(deg))

def rad(angle):
    return math.radians(angle*degreeQuantum)

def deg(rad):
    return math.degrees(rad)

def print_nonzero(gridProb):
    nz = np.nonzero(gridProb)
    ind = zip(nz[0], nz[1], nz[2])
    for i in ind:
        print i,'-->',gridProb[i[0]][i[1]][i[2]]

def mvnpdf(x, y, mux, muy, sigma):
    X = np.copy(x)
    Y = np.copy(y)
    X -= mux
    Y -= muy
    X = x**2
    Y = y**2
    XY = (X+Y)/(2*(sigma**2))

    z = np.exp(-XY)
    z /= z.sum()

    return z

def updateHeadingBelief(rot, gridProb):
    print 'rot:',quaternionToEuler(rot)
    for x in range(35):
        for y in range(35):
            for h in range(4):
                p = gridProb[h][x][y]
                gridProb[h][x][y] = 0.0
                newHeadingMean = ((90*h) + quaternionToEuler(rot))%360
                for posterior in range(4):
                    angleBin = int(((newHeadingMean + (posterior*90))%360)/90)
                    gridProb[angleBin][x][y] += p*angleProb[posterior]
    #print_nonzero(gridProb)

def updatePositionBelief(trans, gridProb):
    print 't:',trans
    tempGrid = np.zeros((4,35,35))
    Y,X = np.meshgrid(range(35), range(35))
    X = X*0.2 + 0.1
    Y = Y*0.2 + 0.1
    
    for h in range(4):
        for x in np.array(range(35))*0.2 + 0.1:
            for y in np.array(range(35))*0.2 + 0.1:
                if h == 0:
                    mean = (x + trans, y)
                elif h == 1:
                    mean = (x, y + trans)
                elif h == 2:
                    mean = (x - trans, y)
                else:
                    mean = (x, y - trans)

                p = gridProb[h][x][y]
                gridProb[h][x][y] = 0.0

                Z = mvnpdf(X, Y, mean[0], mean[1], 0.1)
                Z *= p
                tempGrid[h] += Z
        print '*****************'

    for direction in range(4):
        gridProb[direction] = tempGrid[direction]
                 
       
def handle_movement(message, gridProb):
    updateHeadingBelief(message.rotation1, gridProb)
    updatePositionBelief(message.translation, gridProb)
    updateHeadingBelief(message.rotation2, gridProb)

def handle_observation(message, gridProb):
    Y,X = np.meshgrid(range(35), range(35))
    X = X*0.2 + 0.1
    Y = Y*0.2 + 0.1
    mean = landmarks[message.tagNum]
    Z = mvnpdf(X, Y, mean['x'], mean['y'], 0.1)
    Z /= Z.sum()
    #print gridProb
    for h in range(4):
        theta = h*90+quaternionToEuler(message.bearing)
        deltaLandmark = (message.range*cos(theta), message.range*sin(theta))
        for x in range(35):
            for y in range(35):
                expectedLandmark = (np.floor((np.array((x*0.2, y*0.2))+deltaLandmark)/0.2))
                if max(expectedLandmark) < 35:
                    gridProb[h][x][y] *= Z[expectedLandmark[0]][expectedLandmark[1]]
                else:
                    gridProb[h][x][y] *= 0

def readMessages(filename):
    bag = rosbag.Bag(filename)
    gridProb = np.zeros((4, 35, 35))
    c = 0
    gridProb[2][11][27] = 1.0
    for message in bag.read_messages():
        c += 1
        if message[0] == 'Movements':
            handle_movement(message[1], gridProb)
        else:
            handle_observation(message[1], gridProb)
            print np.max(gridProb)
            print np.where(gridProb == gridProb.max())

if __name__ == '__main__':
    #np.set_printoptions(threshold=np.nan)
    readMessages('grid.bag')

