import math
import numpy as np
from mayavi.mlab import *

pi=math.pi

#   Dimensions and fixed stuff
##  Tank size, roation, temperature
r_t = 0.6               # in m
r_i = 0.1               # in m (Innenrand)    
h_t = 0.3               # in m
omega = 2*pi/6          # in 1/s
T0 = 273                # in K
TR = 300   

##  Number of grid points
n_r = 10
n_phi = 8
n_z = 10

##  Stepsizes           Dimensions:
dr = (r_t-r_i)/n_r            # 1/m
dphi = 2*pi/n_phi       # 1/m
dz = h_t/n_z            # 1/m

dT = (TR-T0)/(n_r)    # K/m

dri = 1/dr
dphii = 1/dphi

#   functions for j and phi, h
def jp(j):
    if j==n_phi-1:
        return 0
    else:
        return j+1
def phi(j):
    return j*2*pi/n_phi
def h(k):
    return k*dz


##  Nondimensionalization
U = 0.01                # typical size velocity in m/s
L = dr                  # typical size coordinates in m

a = 69*10**(-6)         # linear thermal expansion coefficient [a]=1/K

gn = -9.81
gdim = L/U**2
g = gn/gdim             # undimensionalized version of gravity constant

omegadim = L/U
om = omega/omegadim     # undimensionalized version of omega

#   Matrix initialization
u = np.zeros((n_r,n_phi,n_z))      
u_cart = np.zeros((n_r,n_phi,n_z))

r = np.arange(r_i,r_t+dr/2, dr)

v = np.zeros((n_r,n_phi,n_z))
v_cart = np.zeros((n_r,n_phi,n_z))

w = np.zeros((n_r,n_phi,n_z))

T = np.ones((n_r+1,n_phi,n_z))

x = np.zeros((n_r,n_phi,n_z))
y = np.zeros((n_r,n_phi,n_z))
z = np.zeros((n_r,n_phi,n_z))


#   Calculations:
for i in range(n_r):
    for j in range(n_phi):
        for k in range(n_z-1):
            T[i,j,k] = T0 + (i)*dT
            T[i+1,j,k] = T0 + (i+1)*dT              # Habe T(r) immer mit 1 verglichen
                                                    # => radialer Unterschied in v
            v[i,j,k+1] = v[i,j,k] + dz*(0.5*a*g/om*dri*(T[i+1,j,k]-T[i,j,k]))

            
#print("u:", unew[4,4,:])
#print("v:", vnew[:,1,n_z-1])
print("T(r):", T[:,1,1])

for i in range(n_r):
    for j in range(n_phi):
        for k in range(n_z):
            u_cart[i,j,k] = u[i,j,k]*math.cos(phi(j)) - v[i,j,k]*math.sin(phi(j))
            v_cart[i,j,k] = u[i,j,k]*math.sin(phi(j)) + v[i,j,k]*math.cos(phi(j))

            x[i,j,k] = r[i]*math.cos(phi(j))
            y[i,j,k] = r[i]*math.sin(phi(j))
            z[i,j,k] = h(k)


quiver3d(x,y,z,u_cart,v_cart,w)
