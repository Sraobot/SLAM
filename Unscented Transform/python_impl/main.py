import numpy as np
import matplotlib.pyplot as plt
from unscented_transform import *
from tools.plot_helper_functions import *

sigma = 0.1*np.eye(2)
mu = np.zeros((2,1))
# print(mu.shape)
#mu = np.array([[1],[2]], dtype=float)
mu[0] = 1
mu[1] = 2
n = np.size(mu)

# Compute lambda
alpha = 0.9
beta = 2
kappa  = 1
Lambda = (alpha**2)*(n+kappa)-n

# Compute sigma points and weights
sigma_points, weights_m, weights_c = compute_sigma_points_weights(mu, sigma, Lambda, n, alpha, beta)

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(10,10))

# Plotting original distribution and the sigma points
ax1.scatter(mu[0], mu[1], marker="+",c='b', s=150)
ax1.scatter(sigma_points[0,:], sigma_points[1,:], c='b', marker="x",s=150)
ax2.scatter(mu[0], mu[1], marker="+",c='b', s=150)
ax2.scatter(sigma_points[0,:], sigma_points[1,:], c='b', marker="x",s=150)
ax3.scatter(mu[0], mu[1], marker="+",c='b', s=150)
ax3.scatter(sigma_points[0,:], sigma_points[1,:], c='b', marker="x",s=150)
ax4.scatter(mu[0], mu[1], marker="+",c='b', s=150)
ax4.scatter(sigma_points[0,:], sigma_points[1,:], c='b', marker="x",s=150)


points = drawprobellipse(mu, sigma, 0.9, 'red')
line1, = ax1.plot(points[0,:], points[1,:], c='b')
ax2.plot(points[0,:], points[1,:], c='b')
ax3.plot(points[0,:], points[1,:], c='b')
ax4.plot(points[0,:], points[1,:], c='b')

# Transform sigma points amd recover the gaussian
trans_sigma_points = transform(sigma_points,1)
mu_trans, sigma_trans = recover_gaussian(trans_sigma_points,weights_m, weights_c, n)
points = drawprobellipse(mu_trans, sigma_trans, 0.9, 'red')
line2, = ax1.plot(points[0,:], points[1,:], c='r')
ax1.scatter(trans_sigma_points[0,:], trans_sigma_points[1,:], c='r', marker="x", s=150)
ax1.set_title("Linear")

trans_sigma_points = transform(sigma_points,2)
mu_trans, sigma_trans = recover_gaussian(trans_sigma_points,weights_m, weights_c, n)
points = drawprobellipse(mu_trans, sigma_trans, 0.9, 'red')
ax2.plot(points[0,:], points[1,:], c='r')
ax2.scatter(trans_sigma_points[0,:], trans_sigma_points[1,:], c='r', marker="x", s=150)
ax2.set_title("Polar form")

trans_sigma_points = transform(sigma_points,3)
mu_trans, sigma_trans = recover_gaussian(trans_sigma_points,weights_m, weights_c, n)
points = drawprobellipse(mu_trans, sigma_trans, 0.9, 'red')
ax3.plot(points[0,:], points[1,:], c='r')
ax3.scatter(trans_sigma_points[0,:], trans_sigma_points[1,:], c='r', marker="x", s=150)
ax3.set_title("Nonlinear case")

trans_sigma_points = transform(sigma_points,4)
mu_trans, sigma_trans = recover_gaussian(trans_sigma_points,weights_m, weights_c, n)
points = drawprobellipse(mu_trans, sigma_trans, 0.9, 'red')
ax4.plot(points[0,:], points[1,:], c='r')
ax4.scatter(trans_sigma_points[0,:], trans_sigma_points[1,:], c='r', marker="x", s=150)
ax4.set_title("Nonlinear case")

plt.figlegend([line1,line2],['Original','Transformed'],fontsize=14)
fig.suptitle("Unscented Transform", fontsize=16)
plt.savefig("plots/Unscented_transform.png")
plt.show()