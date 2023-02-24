from prediction_step import *
from correction_step import *
from tools.reading_data import read_sensor_data, read_world
from tools.plot_state import drawprobellipse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.markers import MarkerStyle
import numpy as np
import time
frames_num = 330
class EKF_SLAM:
    def __init__(self):

        # self.ax = ax
        # self.fig = fig
        # True landmark coordiantes, not given to the robot
        self.landmarks = read_world('../data/world.dat')
        # self.scat = ax.scatter(np.zeros((2,1)),np.zeros((2,1)))        
        # Reading sessor data, i.e., odometry and range bearing sensor
        self.df_data = read_sensor_data('../data/sensor_data.dat')
        
        # Get number of landmarks in the map
        self.N = self.landmarks.shape[0]
        
        self.observedLandmarks = np.zeros((self.N+1, 1), dtype=int) 
        # + 1 helps in computation

        # Initialize belief:
        # mu: 2N+3x1 vector representing the mean of the normal distribution
        # The first 3 components of mu correspond to the pose of the robot,
        # and the landmark poses (xi, yi) are stacked in ascending id order.
        # sigma: (2N+3)x(2N+3) covariance matrix of the normal distribution
        
        self.mu = np.zeros((2*self.N+3, 1),dtype=float)
        
        self.mu_arr = np.zeros((3,frames_num))

        self.robot_sigma = np.zeros((3,3))
        self.robot_map_sigma = np.zeros((3, 2*self.N))
        self.map_sigma = 100*np.eye(2*self.N)
        self.sigma = np.concatenate((np.concatenate((self.robot_sigma, self.robot_map_sigma), axis=1),
                       np.concatenate((self.robot_map_sigma.T, self.map_sigma), axis=1)), axis=0)
        self.gaps = 0         
        self.curr_gap = 0
        self.plus = MarkerStyle("+")
        self.circle = MarkerStyle("o")

    def update(self,i):
        print()
        print(f"i:{i}")
        self.curr_gap = self.df_data.iloc[i+self.gaps][4]
        
        self.mu, self.sigma = prediction_step(self.mu, self.sigma, self.df_data.iloc[i+self.gaps,:], self.N)

        # +1 in required to move from odometry to sensor observation for that timestep.
        self.mu, self.sigma, self.observedLandmarks = correction_step(self.mu, self.sigma, 
                                self.df_data.iloc[i+self.gaps+1:i+self.gaps+1+self.curr_gap,:], self.observedLandmarks)
        # print(f"mu:{self.mu[:3]}")
        self.gaps = self.gaps + self.curr_gap # Keeps a track of number of sensor observations in the data
        # It helps in slicing the dataframe 
        self.mu_arr[:3,i] = self.mu[:3,0]
        # plotting the state of the robot
        self.update_plot()

    
    def update_plot(self):
        """
            Plots the robot's state. 
            
            It plots the robot position and it's ellipse.
            It plots all the landmarks.
            It plots the ellipse of the observed landmarks.
        """
        plt.cla()
        plt.xlim(-2,12)
        plt.ylim(-2,12)
        # sum_observerland = np.sum(self.observedLandmarks)
        # print(f"sum_of_observed_landmarks:{sum_observerland}")
        # Plot the robot and it's ellipse
        robot_data = self.plot_robot()
        # print(robot_data)

        plt.scatter(self.mu[0], self.mu[1], c='r', marker=self.circle, s=10)
        plt.plot(robot_data[0,:],robot_data[1,:])        
        
        # Plot the landmarks coordinates from the world.dat as 'k+'
        landmarks_data = self.plot_landmarks()
        plt.scatter(landmarks_data[0,:],landmarks_data[1,:],c='k',marker=self.plus,s=100)
        
        # Plot the ellipses and location(from mu) of only those landmarks which are in observedLandmarks vector
        j = 0 # It is used to pick the sigma associated with the landmark.
        for landmark in range(0, self.N):
            if self.observedLandmarks[landmark+1]:
                # print(f"landmark:{landmark}")
                # scat_data[:,101+self.N+ j*100: 101+self.N+(j+1)*100] = drawprobellipse(self.mu[3+2*landmark:3+2*(landmark+1)], 
                        # self.sigma[3+2*landmark:3+2*(landmark+1),3+2*landmark:3+2*(landmark+1)],0.6 ,'red')
                data = drawprobellipse(self.mu[3+2*landmark:3+2*(landmark+1)], self.sigma[3+2*landmark:3+2*(landmark+1),3+2*landmark:3+2*(landmark+1)],0.6 ,'red')
                plt.plot(data[0,:],data[1,:], c='b')
                j = j + 1

    def plot_robot(self):
        return drawprobellipse(self.mu[:3], self.sigma[:3,:3], 0.6, 'k')
    
    def plot_landmarks(self):
        return self.landmarks[:,1:].T
    
    def init_plot(self):
        
        # plt.cla()
        ######## We don't need to plot ellipses in the init plot. 
        # We plot ellipses as we observe the landmarks ###################
        
        print("init_plot")
        robot_data = self.plot_robot()
        
        landmark_data = self.plot_landmarks() # We already have it just slices out and transposes the required data
        
        plt.scatter(self.mu[0], self.mu[1], c='r', marker=self.circle, s=20)
        plt.plot(robot_data[0,:],robot_data[1,:])

        plt.scatter(landmark_data[0,:],landmark_data[1,:],c='k',marker=self.plus,s=100)

ekf = EKF_SLAM()
fig= plt.figure(figsize=(10,10))
# Total max frames can be 330.
anim = FuncAnimation(fig, func = ekf.update,frames=frames_num,init_func=ekf.init_plot,interval=10,repeat=False)
# anim.save("plotting.mp4")
plt.show()
print("time.sleep()")
time.sleep(3)
plt.cla()
# print(ekf.mu_arr)
abc = np.diff(ekf.mu_arr,axis=1)
# print(abc)
abc = np.linalg.norm(abc, axis=0)
# print(abc)
plt.plot(np.arange(frames_num-1),abc)
plt.show()