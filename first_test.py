import math
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt


pi=math.pi
# u Radialkomponente der Geschwindigkeit, v Azimutalkomponente der Geschwindigkeit, w vertikale Komponente der Geschwindigkeit, p Druck
# tanksize & angular velocity
r_t=10
h_t=10
omega=10      # in rpm

# fixed stuff
gn = 9.81
#gdim = 
#omegadim =

# time steps
dt = 0.02

# number of grid points
n_r,n_phi,n_z = 50,50,25

# stepsizes
dr = r_t/n_r
dphi = 2*pi/n_phi
dz = h_t/n_z

# r-vector                             brauchts verschiedene r für die versch. Gitter?? r auf u-Gitter definieren?
r = np.linspace(dr-dr/2,r_t-dr/2, n_r)

# u-Gitter u(r,phi,z):
r_u = np.linspace(dr-dr/2,r_t-dr/2, n_r)
phi_u = np.linspace(0,2*pi, n_phi)
z_u = np.linspace(0,h_t, n_z)

#u_grid = [r_u,phi_u,z_u]

# v-Gitter v(r,phi,z):
r_v = np.linspace(dr,r_t, n_r)
phi_v = np.linspace(pi/2,2*pi+pi/2, n_phi)
z_v = np.linspace(0,h_t, n_z)

#v_grid = [r_v,phi_v,z_v]

# w-Gitter w(r,phi,z)           evtl. erste w-Ebene unter erster u,v,p-Ebene ?
r_w = np.linspace(dr,r_t, n_r)
phi_w =  np.linspace(0,2*pi, n_phi)
z_w = np.linspace(dz+dz/2,h_t-dz/2, n_z-1)

# p-Gitter p(r,phi,z):
r_p = np.linspace(dr,r_t, n_r)
phi_p = np.linspace(0,2*pi, n_phi)
z_p = np.linspace(0,h_t, n_z)

#p_grid = [r_p,phi_p,z_p]


# matrix initialization
u = np.zeros((n_r,n_phi,n_z))
unew = np.zeros((n_r,n_phi,n_z))
v = np.zeros((n_r,n_phi,n_z))
vnew = np.zeros((n_r,n_phi,n_z))
w = np.zeros((n_r,n_phi,n_z-1))
wnew = np.zeros((n_r,n_phi,n_z-1))
p = np.ones((n_r,n_phi,n_z))    # Nachregelung?

# discretization

dri=1/dr
dphii=1/dphi
dzi=1/dz

#om=omega/omegadim # undimensionalized version of omega
#g=gn/gdim         # undimensionalized version of gravity constant gn
om=10
g=9.81

for i in range(1,n_r):
    ri=1/r[i]           # r noch definieren
    for j in range(1,n_phi):
        for k in range(1,n_z):

            unew[i,j,k] = u[i,j,k] + dt*(dri**2*(u[i+1,j,k]-2.*u[i,j,k]+u[i-1,j,k]) + ri**2*dphii**2*(u[i,j+1,k]-2.*u[i,j,k]+u[i,j-1,k]) +
                                         dzi**2*(u[i,j,k+1]-2.*u[i,j,k]+u[i,j,k-1]) + (ri-u[i,j,k])*0.5*dri*(u[i+1,j,k]-u[i-1,j,k]) -
                                         ri*0.25*(v[i,j,k]+v[i,j-1,k]+v[i+1,j-1,k]+v[i+1,j,k])*0.5*dphii*(u[i,j+1,k]-u[i,j-1,k]) -
                                         0.25*(w[i,j,k]+w[i+1,j,k]+w[i,j,k-1]+w[i+1,j,k-1])*0.5*dzi*(u[i,j,k+1]-u[u,j,k-1]) -
                                         ri**2*dphii*(v[i+1,j,k]+v[i,j,k]-v[i+1,j-1,k]-v[i,j-1,k]) - ri**2*u[i,j,k] +
                                         0.5*om*(v[i,j,k]+v[i,j-1,k]+v[i+1,j-1,k]+v[i+1,j,k]) + (om**2)*r[i] - dri*(p[i+1,j,k]-p[i,j,k]) )

            vnew[i,j,k] = v[i,j,k] + dt*(dri**2*(v[i+1,j,k]-2.*v[i,j,k]+v[i-1,j,k]) + ri**2*dphii**2*(v[i,j+1,k]-2.*v[i,j,k]+u[i,j-1,k]) +
                                         dzi**2*(v[i,j,k+1]-2.*v[i,j,k]+v[i,j,k-1]) +
                                         (ri-0.25*(u[i,j,k]+u[i-1,j,k]+u[i,j+1,k]+u[i-1,j+1,k]))*0.5*dri*(v[i+1,j,k]-v[i-1,j,k]) -
                                         ri*v[i,j,k]*0.5*dphii*(v[i,j+1,k]-v[i,j-1,k]) -
                                         0.25*(w[i,j,k]+w[i,j+1,k]+w[i,j,k-1]+w[i,j+1,k-1])*0.5*dzi*(v[i,j,k+1]-v[i,j,k-1]) +
                                         ri**2*dphii*(u[i,j,k]+u[i-1,j,k]-u[i,j+1,k]-u[i-1,j+1,k]) - ri**2*v[i,j,k] -
                                         0.5*om*(u[i,j,k]+u[i-1,j,k]+u[i,j+1,k]+u[i-1,j+1,k]) - ri*dphii*(p[i,j+1,k]-p[i,j,k]) )

            wnew[i,j,k] = w[i,j,k] + dt*(dri**2*(w[i+1,j,k]-2.*w[i,j,k]+w[i-1,j,k]) + ri**2*dphii**2*(w[i,j+1,k]-2.*w[i,j,k]+w[i,j-1,k]) +
                                         dzi**2*(w[i,j,k+1]-2.*w[i,j,k]+w[i,j,k-1]) +
                                         (ri-0.25*(u[i,j,k]+u[i-1,j,k]+u[i,j,k+1]+u[i-1,j,k+1]))*0.5*dri*(w[i+1,j,k]-w[i-1,j,k]) -
                                         ri*0.25(v[i,j,k]+v[i,j-1,k]+v[i,j,k+1]+v[i,j-1,k+1])*0.5*dphii*(w[i,j+1,k]-w[i,j-1,k]) -
                                         w[i,j,k]*0.5*dzi*(w[i,j,k+1]-w[i,j,k-1]) - dzi*(p[i,j,k+1]-p[i,j,k]) - g )


print(unew)
print(vnew)
print(wnew)
print(p)


