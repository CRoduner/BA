import math
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as p

pi=math.pi

#   function for j
def jp(j, n_phi):
    if j==n_phi-1:
        return 0
    else:
        return j+1
def jm(j, n_phi):
    if j==0:
        return n_phi-1
    else:
        return j-1
def sin(j):
    return math.sin(j*2*pi/n_phi)


#   Dimensions and fixed stuff
##  Tank size, roation, temperature
r_t = 0.6               # in m
h_t = 0.3               # in m
omega = 2*pi/6          # in 1/s
T0 = 273                # in K
TR = 293                # in K

##  Number of grid points
n_r = 5
n_phi = 4
n_z = 5

##  Stepsizes           Dimensions:
dr = r_t/n_r            # 1/m
dphi = 2*pi/n_phi       # 1/m
dz = h_t/n_z            # 1/m

dT = (TR-T0)/(n_r+1)    # K/m

dri = 1/dr
dphii = 1/dphi


##  Nondimensionalization
U = 0.01                # typical size velocity in m/s
L = dr                  # typical size coordinates in m

a = 69*10**(-6)         # linear thermal expansion coefficient [a]=1/K

gn = 9.81
gdim = L/U**2
g = gn/gdim             # undimensionalized version of gravity constant

omegadim = L/U
om = omega/omegadim     # undimensionalized version of omega

#   Matrix initialization
u = np.zeros((n_r,n_phi,n_z))      
unew = np.zeros((n_r,n_phi,n_z))

r = np.arange(dr/2,r_t+dr/2, dr)

v = np.zeros((n_r,n_phi,n_z))
vnew = np.zeros((n_r,n_phi,n_z))

T = np.ones((n_r+1,n_phi,n_z))


#   Calculations:
for i in range(n_r):
    for j in range(n_phi):
        for k in range(n_z):
            T[i,j,k] = (i+1)*dT
            T[n_r,j,k] = (n_r+1)*dT
            
            unew[i,j,k] = u[i,j,k] - dz*(0.5*a*g/(om*sin(j+0.5)*r[i])*dphii*(T[i,jp(j,n_phi),k]-T[i,j,k]))
            vnew[i,j,k] = v[i,j,k] + dz*(0.5*a*g/(om*(sin(j)+0.0001))*dri*(T[i+1,j,k]-T[i,j,k]))                # Problem mit sin(j) -> div by 0
            
print("u:", unew)
print("v:", vnew)
print("T:", T)
