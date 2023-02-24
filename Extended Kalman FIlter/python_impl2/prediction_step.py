import numpy as np

from tools.normalise_angles import normalise_angle

def prediction_step(mu, sigma, u, N):
    """
    mu is NumPy vector of dim 2N+3
    sigma is NumPy matrix of dim: (2N+3) x (2N+3)
    u is a sliced dataframe with a single row
    u = ('ODOMETRY',rot1, trans, rot2, gap)
    """
    rot1 = u[1];    trans = u[2];   rot2 = u[3]
    last_theta = mu[2]
    # print(f"u:{u}")
    update = np.array([[trans*np.cos(normalise_angle(mu[2]+rot1))], 
                       [trans*np.sin(normalise_angle(mu[2] + rot1))], 
                       [normalise_angle(rot1+rot2)]],dtype=float)
    
    mu[:3] = mu[:3] + update
    # Normalising the angle
    mu[2] = normalise_angle(mu[2])

    F_x = np.concatenate((np.eye(3), np.zeros((3, 2*N))),axis=1)
    low_G_x_t = np.eye(3) + np.array([[0, 0, -trans*np.sin(last_theta+rot1)], # See this
                                      [0, 0, trans*np.cos(last_theta+rot1)],
                                      [0, 0, 0]],dtype=float)
    G_t = np.eye(3+2*N) + F_x.T@low_G_x_t@F_x

    R_x_t = np.array([[0.1,  0  ,    0], 
                      [0  ,  0.1,    0],
                      [0  ,  0  , 0.01]])

    sigma = G_t@sigma@G_t.T + F_x.T@R_x_t@F_x

    return mu, sigma

