import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time, sys, os, control.matlab
from kf_calc_f import kf_calc_f
from kf_calc_h import kf_calc_h
from rk4 import rk4
from kf_calc_Fx import kf_calc_Fx
from kf_calc_Hx import kf_calc_Hx
plt.close('all')
np.random.seed(7)
pathname = os.path.basename(sys.argv[0])
filename = os.path.splitext(pathname)[0]


########################################################################
## Importing data
########################################################################

filename = 'F16traindata_CMabV_2023.csv'
train_data = genfromtxt(filename, delimiter=',').T

C_m = train_data[0]
Z_k = train_data[1:4]
U_k = train_data[4:]

########################################################################
## Set simulation parameters
########################################################################

n               = 4                          # state dimension
nm              = 3                          # measurement dimension
m               = 3                          # input dimension
dt              = 0.01                       # time step (s)
N               = len(U_k[0])                # number of samples
epsilon         = 10**(-10)                  # IEKF threshold
doIEKF          = True                       # If false, EKF without iterations is used
maxIterations   = 100                        # maximum amount of iterations per sample

printfigs       = False                      # enable saving figures
figpath         = 'figs/'                    # direction for printed figures

########################################################################
## Set initial values for states and statistics
########################################################################

E_x_0       = np.array([[150],[0],[0],[-0.6]])     # initial estimate of optimal value of x_k1_k1

B           = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0]])               # input matrix
G           = np.zeros((4,4))                     # input noise matrix, why is this 4x4 and not 4x3?

# Initial estimate for covariance matrix
std_x_0   = 1                                     # initial standard deviation of state prediction error
P_0       = np.array([[std_x_0**2, 0, 0, 0],
                      [0, std_x_0**2, 0, 0],
                      [0, 0, std_x_0**2, 0], 
                      [0, 0, 0, std_x_0**2]])     # initial covariance of state prediction error

# System noise statistics, all noise are white (unbiased and uncorrelated in time)
E_w_u = 0                                         # bias of u noise
std_w_u = 1*10**(-3)                              # standard deviation of u noise
std_w_v = 1*10**(-3)                              # standard deviation of v noise
std_w_w = 1*10**(-3)                              # standard deviation of w noise
std_w_C = 0                                       # standard deviation of Caup noise

Q         = np.array([[std_w_u**2, 0, 0, 0],
                        [0, std_w_v**2, 0, 0],
                        [0, 0, std_w_w**2, 0], 
                        [0, 0, 0, std_w_C**2]])   # variance of system noise

# Measurement noise statistics, all noise are white (unbiased and uncorrelated in time)
std_nu_a = 0.035             # standard deviation of alpha noise
std_nu_b = 0.010             # standard deviation of beta noise
std_nu_V = 0.110             # standard deviation of velocity noise

R      = np.array([[std_nu_a**2, 0, 0],
                    [0, std_nu_b**2, 0],
                    [0, 0, std_nu_V**2]])     # variance of system noise

########################################################################
## Initialize Extended Kalman filter
########################################################################

t_k         = 0
t_k1        = dt

# allocate space to store traces
XX_k1_k1    = np.zeros([n, N])
PP_k1_k1    = np.zeros([n, N])
STD_x_cor   = np.zeros([n, N])
STD_z       = np.zeros([nm, N])
ZZ_pred     = np.zeros([nm, N])
IEKFitcount = np.zeros([N, 1])

# initialize state estimation and error covariance matrix
x_k1_k1     = np.array(E_x_0)   # x(0|0) = E(x_0)
P_k1_k1     = np.array(P_0)     # P(0|0) = P(0)

########################################################################
## Run the Kalman filter
########################################################################

t0          = time.time()

# Run the filter through all N samples
for k in range(0, 1):
    
    # x(k+1|k) (prediction)
    t, x_k1_k   = rk4(kf_calc_f, x_k1_k1, U_k[:,k], [t_k, t_k1])   # add in U_k vector

    # Calc Jacobians Phi(k+1, k) and Gamma(k+1, k)
    Fx          = kf_calc_Fx(0, x_k1_k, U_k[:,k])
    Hx          = kf_calc_Hx(0, x_k1_k, U_k[:,k])
    # Continuous to discrete time transformation of Df(x,u,t)
    ss_B        = control.matlab.ss(Fx, B, Hx, np.zeros((nm, m)))
    ss_G        = control.matlab.ss(Fx, G, np.zeros((nm, n)), np.zeros((nm, n)))
    Psi         = control.matlab.c2d(ss_B, dt).B
    Phi         = control.matlab.c2d(ss_G, dt).A
    Gamma       = control.matlab.c2d(ss_G, dt).B

    # P(k+1|k) (prediction covariance matrix)
    P_k1_k      = Phi@P_k1_k1@Phi.transpose() + Gamma@Q@Gamma.transpose()
    
    print(f'P_k1_k = {P_k1_k}')