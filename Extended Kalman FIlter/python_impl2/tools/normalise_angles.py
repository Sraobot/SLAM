# Brings back the angle to -pi to pi range.

from numpy import pi

def normalise_angle(phi):
    while(phi > pi):
        phi = phi - 2*pi
    while(phi < -pi):
        phi = phi + 2*pi
    
    return phi

def normalise_bearings(arr):
    # print("arr.shape[0]:",arr.shape[0])
    for i in range(1,arr.shape[0],2):
        # print(f"i:{i}")
        arr[i,0] = normalise_angle(arr[i,0])
    return arr

