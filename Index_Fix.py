import math
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

# u Radialkomponente der Geschwindigkeit, v Azimutalkomponente der Geschwindigkeit, w vertikale Komponente der Geschwindigkeit, p Druck

pi=math.pi

# tanksize & angular velocity
r_t=10
h_t=10
omega=10      # in rpm

# fixed stuff
gn = 9.81
#gdim = 
#omegadim =
#om=omega/omegadim # undimensionalized version of omega
#g=gn/gdim         # undimensionalized version of gravity constant gn
om=10
g=9.81

# time steps
dt = 0.02

# number of grid points
n_r = 20
n_phi = 20
n_z = 10

# stepsizes
dr = r_t/n_r
dphi = 2*pi/n_phi
dz = h_t/n_z


## Abkürzungen, Dimensionen
dri=1/dr
dphii=1/dphi
dzi=1/dz


# r-vector                             brauchts verschiedene r für die versch. Gitter?? r auf u-Gitter definieren?
r = np.linspace(dr-dr/2,r_t-dr/2, n_r)

# matrix initialization
u = np.zeros((n_r-1,n_phi,n_z))      
unew = np.zeros((n_r-1,n_phi,n_z))

v = np.zeros((n_r,n_phi,n_z))
vnew = np.zeros((n_r,n_phi,n_z))

w = np.zeros((n_r,n_phi,n_z-1))
wnew = np.zeros((n_r,n_phi,n_z-1))

p = np.ones((n_r,n_phi,n_z))    # Nachregelung?


# Zeitschlaufe

## Intitialisierung der Randwerte / Spezialfälle:


### Randwerte i=0, i=n_r-1 für v,p und i=n_r-1 für w(Werte noch nicht bestimmt)
for j in range(n_phi):
    for k in range (1,n_z):
        v[n_r-1,j,k] = om
        p[n_r-1,j,k] = 1
        v[0,j,k] = 0
        p[0,j,k] = 1
    for k in range (1,n_z-1):
        w[n_r-1,j,k] = 0

### Spezialfälle i=0 für u,w und i=n_r-2 für u
for j in range(n_phi):
    if j==0:
        jm = n_phi-1
        jp = j+1
    elif j==n_phi-1:
        jm = j-1
        jp = 0
    else:
        jm = j-1
        jp = j+1

    for k in range(1,n_z-1):
        ri=1/r[0]
        unew[0,j,k] = u[0,j,k] + dt*(dri**2*(u[2,j,k]-2.*u[1,j,k]+u[0,j,k]) + ri**2*dphii**2*(u[0,jp,k]-2.*u[0,j,k]+u[0,jm,k]) +
                                     dzi**2*(u[0,j,k+1]-2.*u[0,j,k]+u[0,j,k-1]) + (ri-u[0,j,k])*dri*(u[1,j,k]-u[0,j,k]) -
                                     ri/3.*(v[0,j,k]+v[1,jm,k]+v[1,j,k])*0.5*dphii*(u[0,jp,k]-u[0,jm,k]) -
                                     0.25*(w[0,j,k]+w[1,j,k]+w[0,j,k-1]+w[1,j,k-1])*0.5*dzi*(u[0,j,k+1]-u[0,j,k-1]) -
                                     ri**2*dphii*(v[1,j,k]+v[0,j,k]-v[1,jm,k]-v[0,jm,k]) - ri**2*u[0,j,k] +
                                     0.5*om*(v[0,j,k]+v[0,jm,k]+v[1,jm,k]+v[1,j,k]) + (om**2)*r[0] - dri*(p[1,j,k]-p[0,j,k]) )
        ri=1/r[n_r-2]
        unew[n_r-2,j,k] = u[n_r-2,j,k] + dt*(dri**2*(u[n_r-2,j,k]-2.*u[n_r-3,j,k]+u[n_r-4,j,k]) + ri**2*dphii**2*(u[n_r-2,jp,k]-2.*u[n_r-2,j,k]+u[n_r-2,jm,k]) +
                                     dzi**2*(u[n_r-2,j,k+1]-2.*u[n_r-2,j,k]+u[n_r-2,j,k-1]) + (ri-u[n_r-2,j,k])*dri*(u[n_r-2,j,k]-u[n_r-3,j,k]) -
                                     ri*0.25*(v[n_z-2,j,k]+v[n_z-2,jm,k]+v[n_z-3,jm,k]+v[n_z-3,j,k])*0.5*dphii*(u[n_z-2,jp,k]-u[n_z-2,jm,k]) -
                                     0.25*(w[n_z-2,j,k]+w[n_z-3,j,k]+w[n_z-2,j,k-1]+w[n_z-3,j,k-1])*0.5*dzi*(u[n_z-2,j,k+1]-u[n_z-2,j,k-1]) -
                                     ri**2*dphii*(v[n_z-3,j,k]+v[n_z-2,j,k]-v[n_z-3,jm,k]-v[n_z-2,jm,k]) - ri**2*u[n_z-2,j,k] +
                                     0.5*om*(v[n_z-2,j,k]+v[n_z-2,jm,k]+v[n_z-3,jm,k]+v[n_z-3,j,k]) + (om**2)*r[n_z-2] - dri*(p[n_z-2,j,k]-p[n_z-3,j,k]) )
    for k in range(1,n_z-2):
        ri=1/r[0]
        wnew[0,j,k] = w[0,j,k] + dt*(dri**2*(w[2,j,k]-2.*w[1,j,k]+w[0,j,k]) + ri**2*dphii**2*(w[0,jp,k]-2.*w[0,j,k]+w[0,jm,k]) +
                                     dzi**2*(w[0,j,k+1]-2.*w[0,j,k]+w[0,j,k-1]) +
                                     (ri-0.25*(u[1,j,k]+u[0,j,k]+u[1,j,k+1]+u[0,j,k+1]))*dri*(w[1,j,k]-w[0,j,k]) -
                                     ri/3.*(v[0,j,k]+v[0,j,k+1]+v[0,jm,k+1])*0.5*dphii*(w[0,jp,k]-w[0,jm,k]) -
                                     w[0,j,k]*0.5*dzi*(w[0,j,k+1]-w[0,j,k-1]) - dzi*(p[0,j,k+1]-p[0,j,k]) - g )


### Randwerte k=0 für u,v,p (Werte noch nicht bestimmt)
for j in range(n_phi):
    for i in range(n_r-1):
        u[i,j,0] = 0
    for i in range(n_r):
        v[i,j,0] = 0
        p[i,j,0] = 1
### Spezialfall k=0, k=n_z-2 für w und k=n_z-1 für u,v
for i in range(1,n_r-1):
    ri=1/r[i]
    for j in range(n_phi):
        if j==0:
            jm = n_phi-1
            jp = j+1
        elif j==n_phi-1:
            jm = j-1
            jp = 0
        else:
            jm = j-1
            jp = j+1

        wnew[i,j,0] = w[i,j,0] + dt*(dri**2*(w[i+1,j,0]-2.*w[i,j,0]+w[i-1,j,0]) + ri**2*dphii**2*(w[i,jp,0]-2.*w[i,j,0]+w[i,jm,0]) +
                                     dzi**2*(w[i,j,2]-2.*w[i,j,1]+w[i,j,0]) +
                                     (ri-0.25*(u[i,j,0]+u[i-1,j,0]+u[i,j,1]+u[i-1,j,1]))*0.5*dri*(w[i+1,j,0]-w[i-1,j,0]) -
                                     ri*0.25*(v[i,j,0]+v[i,jm,0]+v[i,j,1]+v[i,jm,1])*0.5*dphii*(w[i,jp,0]-w[i,jm,0]) -
                                     w[i,j,0]*dzi*(w[i,j,1]-w[i,j,0]) - dzi*(p[i,j,1]-p[i,j,0]) - g )
        
        wnew[i,j,n_z-2] = w[i,j,n_z-2] + dt*(dri**2*(w[i+1,j,n_z-2]-2.*w[i,j,n_z-2]+w[i-1,j,n_z-2]) +
                                             ri**2*dphii**2*(w[i,jp,n_z-2]-2.*w[i,j,n_z-2]+w[i,jm,n_z-2]) +
                                             dzi**2*(w[i,j,n_z-4]-2.*w[i,j,n_z-3]+w[i,j,n_z-2]) +
                                             (ri-0.25*(u[i,j,n_z-2]+u[i-1,j,n_z-2]+u[i,j,n_z-3]+u[i-1,j,n_z-3]))*0.5*dri*(w[i+1,j,n_z-2]-w[i-1,j,n_z-2]) -
                                             ri*0.25*(v[i,j,n_z-2]+v[i,jm,n_z-2]+v[i,j,n_z-3]+v[i,jm,n_z-3])*0.5*dphii*(w[i,jp,n_z-2]-w[i,jm,n_z-2]) -
                                             w[i,j,n_z-2]*dzi*(w[i,j,n_z-2]-w[i,j,n_z-3]) - dzi*(p[i,j,n_z-2]-p[i,j,n_z-3]) - g )

        vnew[i,j,n_z-1] = v[i,j,n_z-1] + dt*(dri**2*(v[i+1,j,n_z-1]-2.*v[i,j,n_z-1]+v[i-1,j,n_z-1]) +
                                             ri**2*dphii**2*(v[i,jp,n_z-1]-2.*v[i,j,n_z-1]+u[i,jm,n_z-1]) +
                                             dzi**2*(v[i,j,n_z-1]-2.*v[i,j,n_z-2]+v[i,j,n_z-3]) +
                                             (ri-0.25*(u[i,j,n_z-1]+u[i-1,j,n_z-1]+u[i,jp,n_z-1]+u[i-1,jp,n_z-1]))*0.5*dri*(v[i+1,j,n_z-1]-v[i-1,j,n_z-1]) -
                                             ri*v[i,j,n_z-1]*0.5*dphii*(v[i,jp,n_z-1]-v[i,jm,n_z-1]) -
                                             0.25*(0+0+w[i,j,n_z-2]+w[i,jp,n_z-2])*0.5*dzi*(v[i,j,n_z-1]-v[i,j,n_z-2]) + # Vorfaktor 0.5 oder 0.25 / w ganz aussen nul setzen bzw. vernachlässigen
                                             ri**2*dphii*(u[i,j,n_z-1]+u[i-1,j,n_z-1]-u[i,jp,n_z-1]-u[i-1,jp,n_z-1]) - ri**2*v[i,j,n_z-1] -
                                             0.5*om*(u[i,j,n_z-1]+u[i-1,j,n_z-1]+u[i,jp,n_z-1]+u[i-1,jp,n_z-1]) - ri*dphii*(p[i,jp,n_z-1]-p[i,j,n_z-1]) )

for i in range(1,n_r-2):
    ri=1/r[i]
    for j in range(n_phi):
        if j==0:
            jm = n_phi-1
            jp = j+1
        elif j==n_phi-1:
            jm = j-1
            jp = 0
        else:
            jm = j-1
            jp = j+1
            
        unew[i,j,n_z-1] = u[i,j,n_z-1] + dt*(dri**2*(u[i+1,j,n_z-1]-2.*u[i,j,n_z-1]+u[i-1,j,n_z-1]) +
                                             ri**2*dphii**2*(u[i,jp,n_z-1]-2.*u[i,j,n_z-1]+u[i,jm,n_z-1]) +
                                             dzi**2*(u[i,j,n_z-1]-2.*u[i,j,n_z-2]+u[i,j,n_z-3]) + (ri-u[i,j,n_z-1])*0.5*dri*(u[i+1,j,n_z-1]-u[i-1,j,n_z-1]) -
                                             ri*0.25*(v[i,j,n_z-1]+v[i,jm,n_z-1]+v[i+1,jm,n_z-1]+v[i+1,j,n_z-1])*0.5*dphii*(u[i,jp,n_z-1]-u[i,jm,n_z-1]) -
                                             0.25*(0+0+w[i,j,n_z-2]+w[i+1,j,n_z-2])*0.5*dzi*(u[i,j,n_z-1]-u[i,j,n_z-2]) - # Vorfaktor 0.5 oder 0.25 -> wie v
                                             ri**2*dphii*(v[i+1,j,n_z-1]+v[i,j,n_z-1]-v[i+1,jm,n_z-1]-v[i,jm,n_z-1]) - ri**2*u[i,j,n_z-1] +
                                             0.5*om*(v[i,j,n_z-1]+v[i,jm,n_z-1]+v[i+1,jm,n_z-1]+v[i+1,j,n_z-1]) +
                                             (om**2)*r[i] - dri*(p[i+1,j,n_z-1]-p[i,j,n_z-1]) )


### Speziafall k=n_z-1 für u,v
        
## allgemeiner Fall
for i in range(1,n_r-2):
    ri=1/r[i]
    for j in range(n_phi):
        if j==0:
            jm = n_phi-1
            jp = j+1
        elif j==n_phi-1:
            jm = j-1
            jp = 0
        else:
            jm = j-1
            jp = j+1
            
        for k in range(1,n_z-2):
            
            unew[i,j,k] = u[i,j,k] + dt*(dri**2*(u[i+1,j,k]-2.*u[i,j,k]+u[i-1,j,k]) + ri**2*dphii**2*(u[i,jp,k]-2.*u[i,j,k]+u[i,jm,k]) +
                                         dzi**2*(u[i,j,k+1]-2.*u[i,j,k]+u[i,j,k-1]) + (ri-u[i,j,k])*0.5*dri*(u[i+1,j,k]-u[i-1,j,k]) -
                                         ri*0.25*(v[i,j,k]+v[i,jm,k]+v[i+1,jm,k]+v[i+1,j,k])*0.5*dphii*(u[i,jp,k]-u[i,jm,k]) -
                                         0.25*(w[i,j,k]+w[i+1,j,k]+w[i,j,k-1]+w[i+1,j,k-1])*0.5*dzi*(u[i,j,k+1]-u[i,j,k-1]) -
                                         ri**2*dphii*(v[i+1,j,k]+v[i,j,k]-v[i+1,jm,k]-v[i,jm,k]) - ri**2*u[i,j,k] +
                                         0.5*om*(v[i,j,k]+v[i,jm,k]+v[i+1,jm,k]+v[i+1,j,k]) + (om**2)*r[i] - dri*(p[i+1,j,k]-p[i,j,k]) )

            vnew[i,j,k] = v[i,j,k] + dt*(dri**2*(v[i+1,j,k]-2.*v[i,j,k]+v[i-1,j,k]) + ri**2*dphii**2*(v[i,jp,k]-2.*v[i,j,k]+u[i,jm,k]) +
                                         dzi**2*(v[i,j,k+1]-2.*v[i,j,k]+v[i,j,k-1]) +
                                         (ri-0.25*(u[i,j,k]+u[i-1,j,k]+u[i,jp,k]+u[i-1,jp,k]))*0.5*dri*(v[i+1,j,k]-v[i-1,j,k]) -
                                         ri*v[i,j,k]*0.5*dphii*(v[i,jp,k]-v[i,jm,k]) -
                                         0.25*(w[i,j,k]+w[i,jp,k]+w[i,j,k-1]+w[i,jp,k-1])*0.5*dzi*(v[i,j,k+1]-v[i,j,k-1]) +
                                         ri**2*dphii*(u[i,j,k]+u[i-1,j,k]-u[i,jp,k]-u[i-1,jp,k]) - ri**2*v[i,j,k] -
                                         0.5*om*(u[i,j,k]+u[i-1,j,k]+u[i,jp,k]+u[i-1,jp,k]) - ri*dphii*(p[i,jp,k]-p[i,j,k]) )

            wnew[i,j,k] = w[i,j,k] + dt*(dri**2*(w[i+1,j,k]-2.*w[i,j,k]+w[i-1,j,k]) + ri**2*dphii**2*(w[i,jp,k]-2.*w[i,j,k]+w[i,jm,k]) +
                                         dzi**2*(w[i,j,k+1]-2.*w[i,j,k]+w[i,j,k-1]) +
                                         (ri-0.25*(u[i,j,k]+u[i-1,j,k]+u[i,j,k+1]+u[i-1,j,k+1]))*0.5*dri*(w[i+1,j,k]-w[i-1,j,k]) -
                                         ri*0.25*(v[i,j,k]+v[i,jm,k]+v[i,j,k+1]+v[i,jm,k+1])*0.5*dphii*(w[i,jp,k]-w[i,jm,k]) -
                                         w[i,j,k]*0.5*dzi*(w[i,j,k+1]-w[i,j,k-1]) - dzi*(p[i,j,k+1]-p[i,j,k]) - g )


print("u:", unew[1,:,1])
##print("v:", vnew)
##print("w:", wnew)
##print("p", p)

