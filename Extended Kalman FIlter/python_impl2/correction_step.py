import numpy as np
from tools.normalise_angles import *

def low_H_t_i(q, sqr_q, del_x, del_y):
    return (1/q)*np.array([[-sqr_q*del_x, -sqr_q*del_y,  0, sqr_q*del_x, sqr_q*del_y],
                           [       del_y,       -del_x, -q,      -del_y,       del_x]],dtype=float)

def correction_step(mu, sigma, z, observedLandmarks):
    """
    Updates the belief, i.e., mu and sigma after observing landmarks, according to the sensor model
    The employed  sensor model measures the range and bearing of a landmark
    mu: 2N+3 x 1 vector representing the state mean.
    The first 3 components of mu correspond to the current estimate of the robot pose [x; y; theta]
    The current pose estimate of the landmark with id = j is: [mu(2*j+2); mu(2*j+3)]
    
    sigma: 2N+3 x 2N+3 is the covariance matrix
    
    z: sliced dataframe of sensor data
       z -> ('SENSOR', observation_number, range, bearing(angle), )
    
    observedLandmarks(j) is false if the landmark with id = j has never been observed before.
    """
    # Total number of landmarks in total
    # print(f"z:{z}")
    N = np.shape(observedLandmarks)[0] - 1

    # Number of measurements in the step 
    m = z.iloc[0,4]

    # expectedZ and Z used in calculating mu
    expectedZ = np.zeros((2*m, 1))
    Z = np.zeros((2*m, 1))

    delta = np.zeros((2,1))
    
    Fxj = np.zeros((5, 2*N+3))
    Fxj_part1 = np.zeros((5,3))
    Fxj_part1[0:3,0:3] = np.eye(3)
    
    # Jacobian
    H = np.zeros((2*m, 2*N+3))

    for i in range(0,m):
        landmarkId = int(z.iloc[i,1])

        if(observedLandmarks[landmarkId]== False):
            mu[3 + 2*(landmarkId-1) : 3 + 2*(landmarkId)]  = mu[0:2] + \
                        np.array([z.iloc[i,2] * np.cos(z.iloc[i,3]+mu[2]), z.iloc[i,2]*np.sin(z.iloc[i,3]+mu[2])])
            observedLandmarks[landmarkId] = 1
        
        Z[2*i: 2*(i+1)] = np.array([[z.iloc[i,2]],
                                    [z.iloc[i,3]]],dtype=float)
        
        delta = mu[3+2*(landmarkId-1):3+2*landmarkId] - mu[0:2]
        q = np.squeeze(delta.T@delta)
        
        expectedZ[2*i: 2*(i+1)] = np.array([[np.sqrt(q)],
                                            [normalise_angle(np.arctan2(delta[1], delta[0]) - mu[2])]],dtype=float)
        

        Fxj_part2 = np.zeros((5, 2*N))
        Fxj_part2[3:5, 2*(landmarkId-1):2*(landmarkId-1)+2] = np.eye(2)
        Fxj = np.concatenate((Fxj_part1, Fxj_part2), axis=1)
        H[2*i:2*(i+1),:] = low_H_t_i(q, np.sqrt(q), delta[0], delta[1])@Fxj
    
    Q = np.eye(2*m)
    Q = Q*0.01
    K = sigma@H.T@np.linalg.inv(H@sigma@H.T + Q)

    difference = normalise_bearings(np.subtract(Z, expectedZ))

    mu = mu + K@difference
    sigma = (np.eye(2*N +3)-K@H)@sigma

    mu[2] = normalise_angle(mu[2])

    return mu, sigma, observedLandmarks