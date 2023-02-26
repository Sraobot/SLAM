import numpy as np
import copy
def compute_sigma_points_weights(mu, sigma, Lambda, n, alpha, beta):
    sigma_points = np.zeros((n,2*n+1))

    sigma_points[:, [0]] = mu

    # const_term is same throughout all iterations. No point of computing it again
    const_term = np.sqrt(n+Lambda)*np.linalg.cholesky(sigma)
    
    # print("const_term shape:", const_term.shape)
    
    for i in range(1,n+1):
        sigma_points[:,[i]] = mu + const_term[:,[i-1]] 
    
    for i in range(n+1,2*n+1):
        sigma_points[:,[i]] = mu - const_term[:,[i-n-1]]

    weights_m = np.zeros((2*n+1,1))
    weights_c = np.zeros((2*n+1,1))
    
    weights_m[0] = Lambda/(n+Lambda)
    weights_c[0] = weights_m[0] + (1-alpha**2 + beta)

    weights_m[1:] = 0.5/(n+Lambda)
    weights_c[1:] = 0.5/(n+Lambda)
    # print("weights_m:",weights_m)
    return sigma_points, weights_m, weights_c

def transform(sigma_points,transform_type):
    points = copy.deepcopy(sigma_points)
    # Linear Transform
    if transform_type == 1:
        points[0,:] = points[0,:] + 1
        points[1,:] = points[1,:] + 2
        return points
    
    elif transform_type == 2:
        r = np.sqrt(np.sum(np.square(points),axis=0,keepdims=True))
        theta = np.arctan2(points[0,:],points[1,:]).reshape((1,5))
        points = np.concatenate((r,theta),axis=0)
        return points
    
    elif transform_type == 3:
        points[0,:] = points[0,:]*np.cos(points[0,:])*np.sin(points[0,:])
        points[1,:] = points[1,:]*np.cos(points[1,:])*np.sin(points[1,:])
        return points
    
    else:
        points[0,:] = points[0,:]*np.cos(points[0,:])*np.exp(-0.2*points[0,:])
        points[1,:] = points[1,:]*np.sin(points[1,:])*np.exp(-0.2*points[1,:])
        return points

def recover_gaussian(trans_sigma_points, weights_m, weights_c,n):
    mu = np.zeros((n,1),dtype=float)
    sigma = np.zeros((n,n),dtype=float)
    
    for i in range(0,2*n+1):
        mu = mu + trans_sigma_points[:,[i]]*weights_m[i]
    
    for i in range(0,2*n+1):
        sigma = sigma + weights_c[i]*((trans_sigma_points[:,[i]]-mu)@((trans_sigma_points[:,[i]]-mu).T))
    
    return mu, sigma