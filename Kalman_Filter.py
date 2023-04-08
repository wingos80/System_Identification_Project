import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time, sys, os, control.matlab
from KF_Functions import *
from Kalman_Filter_Class import *
from Plotter import *

# Set random seed for reproducibility
np.random.seed(7)


########################################################################
## Data I/O managing
########################################################################

filename = 'data/F16traindata_CMabV_2023.csv'
train_data = genfromtxt(filename, delimiter=',').T

C_m = train_data[0]
Z = train_data[1:4]
U = train_data[4:]


result_file = open(f"kf_results.csv", "w")
result_file.write(f"C_a_up\n")


########################################################################
## Set simulation parameters
########################################################################

n               = 4                          # state dimension (not used)
nm              = 3                          # measurement dimension
m               = 3                          # input dimension (not used)
dt              = 0.01                       # time step (s)
N               = len(U[0])                  # number of samples
epsilon         = 10**(-10)                  # IEKF threshold
maxIterations   = 200                        # maximum amount of iterations per sample

printfigs       = False                      # enable saving figures
figpath         = 'figs/'                    # direction for printed figures

########################################################################
## Set initial values for states and statistics
########################################################################

E_x_0       = np.array([[150],[0],[0],[+0.6]])     # initial estimate of optimal value of x_k1_k1

B           = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0]])               # input matrix
# Initial estimate for covariance matrix
std_x_0   = 1                                     # initial standard deviation of state prediction error
P_stds    = [std_x_0, std_x_0, std_x_0, 3]

# System noises, all noise are white (unbiased and uncorrelated in time)
std_w_u = 1*10**(-3)                              # standard deviation of u noise
std_w_v = 1*10**(-3)                              # standard deviation of v noise
std_w_w = 1*10**(-3)                              # standard deviation of w noise
std_w_C = 0                                       # standard deviation of Caup noise
Q_stds  = [std_w_u, std_w_v, std_w_w, std_w_C]

G       = np.eye(4)                           # system noise matrix

# Measurement noise statistics, all noise are white (unbiased and uncorrelated in time)
std_nu_a = 0.035             # standard deviation of alpha noise
std_nu_b = 0.010             # standard deviation of beta noise
std_nu_V = 0.110             # standard deviation of velocity noise
R_stds   = [std_nu_a, std_nu_b, std_nu_V]

########################################################################
## Run the Kalman filter
########################################################################

tic           = time.time()

# Initialize the Kalman filter object
kalman_filter = IEKF(N, nm, dt, epsilon, maxIterations)

# Set up the system in the Kalman filter
kalman_filter.setup_system(E_x_0, kf_calc_f, kf_calc_h, kf_calc_Fx, kf_calc_Hx, B, G, rk4)

# Set up the noise in the Kalman filter
kalman_filter.setup_covariances(P_stds, Q_stds, R_stds)

# Run the filter through all N samples
for k in range(0, N):
    if k % 100 == 0:
        tonc = time.time()
        print(f'Sample {k} of {N} ({k/N*100:.3f} %), time elapsed: {tonc-tic:.2f} s')
        print(f'    Current estimate of C_a_up: {kalman_filter.x_k1_k1[-1,0]:.4f}\n')
    
    # Picking out the k-th entry in the input and measurement vectors
    U_k = U[:,k]  
    Z_k = Z[:,k]

    # Predict and discretize the system
    kalman_filter.predict_and_discretize(U_k)
    
    # Running iterations of the IEKF
    while kalman_filter.not_converged():
        kalman_filter.run_iteration(U_k, Z_k)

    # Once converged, update the state and state covariances estimates
    kalman_filter.update(k)
    
    
    result_file.write(f"{kalman_filter.x_k1_k1[-1,0]}\n")

result_file.close()
toc = time.time()

print(f'Elapsed time: {toc-tic:.5f} s')

########################################################################
## Plotting results
########################################################################

x      = dt*np.arange(0, N, 1)
y1, a1 = kalman_filter.XX_k1_k1[3,:], 1.0         # Estimated C_a_up

ys  = {'Estimated C_a_up' : [y1, a1]}

plotter(x, ys, 'C_a_up estimate evolution', 'Time [s]', 'C_a_up [-]', printfigs)

# Plot variance of all states
y1, a1  = kalman_filter.PP_k1_k1[0,:], 1.0         # Estimated variance of u
y2, a2  = kalman_filter.PP_k1_k1[1,:], 1.0         # Estimated variance of v
y3, a3  = kalman_filter.PP_k1_k1[2,:], 1.0         # Estimated variance of w
y4, a4  = kalman_filter.PP_k1_k1[3,:], 1.0         # Estimated variance of C_a_up

ys  = { 'Estimated variance of u'      : [y1, a1],
        'Estimated variance of v'      : [y2, a2],
        'Estimated variance of w'      : [y3, a3],
        'Estimated variance of C_a_up' : [y4, a4]}

plotter(x, ys, 'Variance of states', 'Time [s]', 'Variance [-]', printfigs)

# Plot number of IEKF iterations at each IEKF step
y1  = kalman_filter.IEKFitcount           # Number of IEKF iterations

ys  = {'Number of IEKF iterations' : [y1, a1]}

plotter(x, ys, 'Number of IEKF iterations', 'Time [s]', 'Number of iterations [-]', printfigs)


########################################################################
## Reconstructing true alpha
########################################################################

alpha_m    = Z[0]                             # Measured alpha
alpha_m_kf = kalman_filter.ZZ_pred[0]         # Predicted alpha from KF

C_a_up     = kalman_filter.XX_k1_k1[3,-1]     # Taking the last estimate of C_a_up
alpha_t    = alpha_m/(1+C_a_up)               # Reconstructing true alpha, noise is 
                                              #  assumed unbiased thus this estimation of 
                                              #  alpha is unbiased as well
alpha_t_kf = alpha_m_kf/(1+C_a_up)            # Reconstructing true alpha from KF

# experimenting with using an exponentially weighted moving average to filter alpha_m

rho = 0.95 # Rho value for smoothing

s_prev = 0 # Initial value ewma value

# Empty arrays to hold the smoothed data
ewma, ewma_bias_corr = np.empty(0), np.empty(0) 

for i in range(len(alpha_m)):
    y = alpha_m[i]
    # Variables to store smoothed data point
    s_cur = 0
    s_cur_bc = 0

    s_cur = rho* s_prev + (1-rho)*y
    s_cur_bc = s_cur/(1-(rho**(i+1)))
    # Append new smoothed value to array
    ewma = np.append(ewma,s_cur)
    ewma_bias_corr = np.append(ewma_bias_corr,s_cur_bc)

    s_prev = s_cur

# Note to self, EWMA is much easier to implement, does not require any knowledge of the system, has a lot less places where bugs can occur, and also 
# computes much faster than the IEKF (perhaps for other variants of the KF as well). However, if we inspect the output of EWMA, we can see that in some 
# occasions the EWMA fails to predict the correct signs of the signal, which might be extremely costly in some applications. Moreover, there is a time 
# delay associated with the signal computed with EWMA, and EWMA can only smooth out signals, it won't be able to reconstruct some unmeasurable states 
# such as the true alpha or upwash coefficient: but KF's can. Moreover, KF's can be used to estimate the variance of the states, which is not possible 
# with EWMA (but there is a way to build confidence intervals for EWMA, in page 8 of: http://www.wiley.com.tudelft.idm.oclc.org/legacy/wileychi/marketmodels/chapter5.pdf). 

# In conclusion, KF's are more powerful than EWMA, but EWMA is much easier to implement and faster.


# Plotting the true alpha and the reconstructed alpha
y1, a1  = alpha_t    , 0.4                    # True alpha
y2, a2  = alpha_m    , 0.4                    # Measured alpha
y3, a3  = alpha_t_kf , 1.0                    # True alpha from KF
y4, a4  = alpha_m_kf , 1.0                    # Predicted alpha from KF
y5, a5  = ewma       , 1.0                    # ewma alpha

ys  = { 'True alpha'              : [y1, a1],
        'Measured alpha'          : [y2, a2],
        'True alpha from KF'      : [y3, a3],
        'Predicted alpha from KF' : [y4, a4],
        'ewma alpha'              : [y5, a5]}

plotter(x, ys, 'Reconstructed alpha', 'Time [s]', 'alpha [-]', printfigs)

plt.show()
