# Brings back the angle to -pi to pi range.

from numpy import pi

def normalise_angle(phi):
    if(phi > pi):
        phi = phi - 2*pi
    elif(phi < -pi):
        phi = phi + 2*pi
    
    return phi

def normalise_bearings(arr):
    for i in range(1,arr.shape[0],2):
        # print(f"i:{i}")
        normalise_angle(arr[i,0])
    return arr

