#! /usr/bin/env python                                                                                              
# -*- coding: latin-1 -*-   
import math
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

## To do:
##          - Wie berechne ich w aus u und v? Wie berechne ich w_c?
##          - 3D CFL

##          - w(r)=0 für r=0 und r=n_r-1 --> wieso? ; w(phi,k) überall gleich 
##          - u(r) wächst nach aussen an, äusserster Wert ist kleiner; u(k) überall gleich ausser zu unterst Null ; u(phi) überall gleich
##          - v überall 0



# u Radialkomponente der Geschwindigkeit, v Azimutalkomponente der Geschwindigkeit, w vertikale Komponente der Geschwindigkeit, p Druck

pi=math.pi
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


# tanksize & angular velocity
r_t=0.5
h_t=0.3
omega=1./6      # in s^(-1)

# number of grid points
n_r = 20
n_phi = 15
n_z = 10

# stepsizes
dr = r_t/n_r
dphi = 2*pi/n_phi
dz = h_t/n_z

precision=10**(-5)  # Gewünschte Präzision der Divergenzfreiheit


# fixed stuff & Entdimensionalisierung
U = 0.01                # typische Grösse für (u,v,w) ? -> müsste das grösser sein?
L = dr                  # typische Grösse für (r,phi,z) -> Gittergrösse?

nu_W = 10**(-6)         # kinematische Viskosität
Re = U*L/nu_W           # Reynolds Number
Reyi = 1/Re             # inverse Reynoldszahl

gn = 9.81
gdim = L/U**2
g = gn/gdim             # undimensionalized version of gravity constant gn

omegadim = L/U
om = omega/omegadim     # undimensionalized version of omega

lamb=1.5                # Koeffizient für Drucknachregelung lambda

rho0=998
p0Pa=101325 #Pascal
p0=p0Pa/(U**2*rho0) #dimensionless

# time steps
dt = dphi/om*0.01            # entdimensionalisiertes als max. Geschwindigkeit Omega => entdimensionalisierter Zeitschritt dt

## Abkürzungen, Dimensionen
dri=1/dr
dphii=1/dphi
dzi=1/dz


# r-vector, phi-vector
ru = np.arange(dr/2,r_t+dr/2, dr)
rp = np.arange(dr,r_t+dr, dr)  # muss evtl. angepasst werden auf r_t+2dr oder so
phiv = np.linspace(dphi/2, 2*pi-dphi/2, n_phi)


# matrix initialization
#   u: i E [0,n_r), j E [0,n_phi), k E [0,n_z]
#   v: i E [0,n_r), j E [0,n_phi), k E [0,n_z]
#   w: i E [0,n_r), j E [0,n_phi), k E [0,n_z-1]  -> es existiert ein w_c in der Mitte
#   p: i E [0,n_r), j E [0,n_phi), k E [0,n_z]    -> es existiert ein p_c in der Mitte
u = np.zeros((n_r,n_phi,n_z))      
unew = np.zeros((n_r,n_phi,n_z))

v = np.zeros((n_r,n_phi,n_z))
vnew = np.zeros((n_r,n_phi,n_z))

w = np.zeros((n_r,n_phi,n_z-1))
wnew = np.zeros((n_r,n_phi,n_z-1))
w_c = np.zeros((n_z-1))
w_c_new = np.zeros((n_z-1))

p = p0*np.ones((n_r,n_phi,n_z))
pnew = np.zeros((n_r,n_phi,n_z))
p_c = p0*np.ones((n_z))

div_u = np.zeros((n_r+1,n_phi,n_z))


#   Zeitschlaufe (sollte dann mal hier beginnen...)
  
##  allgemeiner Fall
for i in range(1,n_r-1):
    rui=1/ru[i]
    rpi=1/rp[i]
    for j in range(n_phi):
        for k in range(1,n_z-1):
            unew[i,j,k] = u[i,j,k] + dt*(Reyi*(dri**2*(u[i+1,j,k]-2.*u[i,j,k]+u[i-1,j,k]) + rui**2*dphii**2*(u[i,jp(j,n_phi),k]-2.*u[i,j,k]+u[i,jm(j,n_phi),k]) +
                                               dzi**2*(u[i,j,k+1]-2.*u[i,j,k]+u[i,j,k-1]) -
                                               rui**2*0.5*dphii*(v[i-1,j,k]+v[i,j,k]-v[i-1,jm(j,n_phi),k]-v[i,jm(j,n_phi),k]) - rui**2*u[i,j,k] ) +
                                         (Reyi*rui-u[i,j,k])*0.5*dri*(u[i+1,j,k]-u[i-1,j,k]) +
                                         rui*(0.25*(v[i,j,k]+v[i,jm(j,n_phi),k]+v[i-1,jm(j,n_phi),k]+v[i-1,j,k]))**2 -
                                         rui*0.25*(v[i,j,k]+v[i,jm(j,n_phi),k]+v[i-1,jm(j,n_phi),k]+v[i-1,j,k])*0.5*dphii*(u[i,jp(j,n_phi),k]-u[i,jm(j,n_phi),k]) -
                                         0.25*(w[i,j,k]+w[i-1,j,k]+w[i,j,k-1]+w[i-1,j,k-1])*0.5*dzi*(u[i,j,k+1]-u[i,j,k-1]) +
                                         0.5*om*(v[i,j,k]+v[i,jm(j,n_phi),k]+v[i-1,jm(j,n_phi),k]+v[i-1,j,k]) + (om**2)*ru[i] - dri*(p[i,j,k]-p[i-1,j,k]) )

            vnew[i,j,k] = v[i,j,k] + dt*(Reyi*(dri**2*(v[i+1,j,k]-2.*v[i,j,k]+v[i-1,j,k]) + rpi**2*dphii**2*(v[i,jp(j,n_phi),k]-2.*v[i,j,k]+v[i,jm(j,n_phi),k]) +
                                               dzi**2*(v[i,j,k+1]-2.*v[i,j,k]+v[i,j,k-1]) +
                                               rpi**2*0.5*dphii*(u[i,j,k]+u[i+1,j,k]-u[i,jp(j,n_phi),k]-u[i+1,jp(j,n_phi),k]) - rpi**2*v[i,j,k]) +
                                         (Reyi*rpi-0.25*(u[i,j,k]+u[i+1,j,k]+u[i,jp(j,n_phi),k]+u[i+1,jp(j,n_phi),k]))*0.5*dri*(v[i+1,j,k]-v[i-1,j,k]) -
                                         rpi*v[i,j,k]*0.25*(u[i,j,k]+u[i+1,j,k]+u[i,jp(j,n_phi),k]+u[i+1,jp(j,n_phi),k]) - 
                                         rpi*v[i,j,k]*0.5*dphii*(v[i,jp(j,n_phi),k]-v[i,jm(j,n_phi),k]) -
                                         0.25*(w[i,j,k]+w[i,jp(j,n_phi),k]+w[i,j,k-1]+w[i,jp(j,n_phi),k-1])*0.5*dzi*(v[i,j,k+1]-v[i,j,k-1]) -
                                         0.5*om*(u[i,j,k]+u[i+1,j,k]+u[i,jp(j,n_phi),k]+u[i+1,jp(j,n_phi),k]) - #
                                         rpi*dphii*(p[i,jp(j,n_phi),k]-p[i,j,k]) )

##  Randwerte / Spezialfälle
### i=0 & i=n_r-1
for j in range(n_phi):
    for k in range (1,n_z-1):
        # i=0
        i=0
        rui=1/ru[i]
        rpi=1/rp[i]         # rp startet noch in der Mitte (rp[0]=0)
        unew[i,j,k] = u[i,j,k] + dt*(Reyi*(dri**2*(u[i+2,j,k]-2.*u[i+1,j,k]+u[i,j,k]) + rui**2*dphii**2*(u[i,jp(j,n_phi),k]-2.*u[i,j,k]+u[i,jm(j,n_phi),k]) +
                                           dzi**2*(u[i,j,k+1]-2.*u[i,j,k]+u[i,j,k-1]) -
                                           rui**2*dphii*0.5*(0+v[i,j,k]-0-v[i,jm(j,n_phi),k]) - rui**2*u[i,j,k] ) +
                                     (Reyi*rui-u[i,j,k])*dri*(u[i+1,j,k]-u[i,j,k]) +
                                     rui*(1/3.*(v[i,j,k]+v[i,jm(j,n_phi),k]+0))**2 -
                                     rui*1/3.*(v[i,j,k]+v[i,jm(j,n_phi),k]+0)*0.5*dphii*(u[i,jp(j,n_phi),k]-u[i,jm(j,n_phi),k]) -
                                     0.25*(w[i,j,k]+w_c[k]+w_c[k-1]+w[i,j,k-1])*0.5*dzi*(u[i,j,k+1]-u[i,j,k-1]) +
                                     2*om*(1/3.*(v[i,j,k]+v[i,jm(j,n_phi),k]+0)) + (om**2)*ru[i] - dri*(p[i,j,k]-p_c[k]) )
        
        vnew[i,j,k] = v[i,j,k] + dt*(Reyi*(dri**2*(v[i+2,j,k]-2.*v[i+1,j,k]+v[i,j,k]) + rpi**2*dphii**2*(v[i,jp(j,n_phi),k]-2.*v[i,j,k]+v[i,jm(j,n_phi),k]) +
                                           dzi**2*(v[i,j,k+1]-2.*v[i,j,k]+v[i,j,k-1]) +
                                           rpi**2*0.5*dphii*(u[i,j,k]+u[i+1,j,k]-u[i,jp(j,n_phi),k]-u[i+1,jp(j,n_phi),k]) - rpi**2*v[i,j,k]) +
                                     (Reyi*rpi-0.25*(u[i,j,k]+u[i+1,j,k]+u[i,jp(j,n_phi),k]+u[i+1,jp(j,n_phi),k]))*dri*(v[i+1,j,k]-v[i,j,k]) -
                                     rpi*v[i,j,k]*0.25*(u[i,j,k]+u[i+1,j,k]+u[i,jp(j,n_phi),k]+u[i+1,jp(j,n_phi),k]) - 

                                     i*v[i,j,k]*0.5*dphii*(v[i,jp(j,n_phi),k]-v[i,jm(j,n_phi),k]) -
                                     0.25*(w[i,j,k]+w[i,jp(j,n_phi),k]+w[i,j,k-1]+w[i,jp(j,n_phi),k-1])*0.5*dzi*(v[i,j,k+1]-v[i,j,k-1]) -
                                     0.5*om*(u[i,j,k]+u[i+1,j,k]+u[i,jp(j,n_phi),k]+u[i+1,jp(j,n_phi),k]) -
                                     rpi*dphii*(p[i,jp(j,n_phi),k]-p[i,j,k]) )
        # i=n_r-1
        i=n_r-1
        rui=1/ru[i]
        rpi=1/rp[i]
        unew[i,j,k] = u[i,j,k] + dt*(Reyi*(dri**2*(u[i,j,k]-2.*u[i-1,j,k]+u[i-2,j,k]) + rui**2*dphii**2*(u[i,jp(j,n_phi),k]-2.*u[i,j,k]+u[i,jm(j,n_phi),k]) +
                                           dzi**2*(u[i,j,k+1]-2.*u[i,j,k]+u[i,j,k-1]) -
                                           rui**2*dphii*0.5*(0+v[i,j,k]-0-v[i,jm(j,n_phi),k]) - rui**2*u[i,j,k] ) +
                                     (Reyi*rui-u[i,j,k])*dri*(u[i,j,k]-u[i-1,j,k]) +
                                     rui*(0.25*(v[i,j,k]+v[i,jm(j,n_phi),k]+v[i-1,jm(j,n_phi),k]+v[i-1,j,k]))**2 -
                                     rui*(0.25*(v[i,j,k]+v[i,jm(j,n_phi),k]+v[i-1,jm(j,n_phi),k]+v[i-1,j,k]))*0.5*dphii*(u[i,jp(j,n_phi),k]-u[i,jm(j,n_phi),k]) -
                                     0.25*(w[i,j,k]+w[i-1,j,k]+w[i,j,k-1]+w[i-1,j,k-1])*0.5*dzi*(u[i,j,k+1]-u[i,j,k-1]) +
                                     2*om*(0.25*(v[i,j,k]+v[i,jm(j,n_phi),k]+v[i-1,jm(j,n_phi),k]+v[i-1,j,k])) + (om**2)*ru[i] - dri*(p[i,j,k]-p[i-1,j,k]) )
    for k in range(n_z):                                 
        vnew[n_r-1,j,k] = 0         # am Tankrand muss die Azimutalgeschwindigkeit der des Tanks entsprechen, auf allen Höhen

### k=0 und k=n_z-1 (für u,v) und k=n_z-2 (für w)
for j in range(n_phi):                                   
    for i in range(1,n_r-1):
        ui=1/ru[i]
        rpi=1/rp[i]
        # k=0                       Wasser kann nicht raus, Wasser kann nicht rein => w=0 ganz unten am Rand und "aussen"
        k=0
        unew[i,j,k] = u[i,j,k] + dt*(Reyi*(dri**2*(u[i+1,j,k]-2.*u[i,j,k]+u[i-1,j,k]) + rui**2*dphii**2*(u[i,jp(j,n_phi),k]-2.*u[i,j,k]+u[i,jm(j,n_phi),k]) +
                                           dzi**2*(u[i,j,k+2]-2.*u[i,j,k+1]+u[i,j,k]) -
                                           rui**2*0.5*dphii*(v[i-1,j,k]+v[i,j,k]-v[i-1,jm(j,n_phi),k]-v[i,jm(j,n_phi),k]) - rui**2*u[i,j,k] ) +
                                     (Reyi*rui-u[i,j,k])*0.5*dri*(u[i+1,j,k]-u[i-1,j,k]) +
                                     rui*(0.25*(v[i,j,k]+v[i,jm(j,n_phi),k]+v[i-1,jm(j,n_phi),k]+v[i-1,j,k]))**2 -
                                     rui*0.25*(v[i,j,k]+v[i,jm(j,n_phi),k]+v[i-1,jm(j,n_phi),k]+v[i-1,j,k])*0.5*dphii*(u[i,jp(j,n_phi),k]-u[i,jm(j,n_phi),k]) -
                                     0.25*(w[i,j,k]+w[i-1,j,k]+0+0)*dzi*(u[i,j,k+1]-u[i,j,k]) +
                                     0.5*om*(v[i,j,k]+v[i,jm(j,n_phi),k]+v[i-1,jm(j,n_phi),k]+v[i-1,j,k]) + (om**2)*ru[i] - dri*(p[i,j,k]-p[i-1,j,k]) )

        vnew[i,j,k] = v[i,j,k] + dt*(Reyi*(dri**2*(v[i+1,j,k]-2.*v[i,j,k]+v[i-1,j,k]) + rpi**2*dphii**2*(v[i,jp(j,n_phi),k]-2.*v[i,j,k]+v[i,jm(j,n_phi),k]) +
                                           dzi**2*(v[i,j,k+2]-2.*v[i,j,k+1]+v[i,j,k]) +
                                           rpi**2*0.5*dphii*(u[i,j,k]+u[i+1,j,k]-u[i,jp(j,n_phi),k]-u[i+1,jp(j,n_phi),k]) - rpi**2*v[i,j,k]) +
                                     (Reyi*rpi-0.25*(u[i,j,k]+u[i+1,j,k]+u[i,jp(j,n_phi),k]+u[i+1,jp(j,n_phi),k]))*0.5*dri*(v[i+1,j,k]-v[i-1,j,k]) -
                                     rpi*v[i,j,k]*0.25*(u[i,j,k]+u[i+1,j,k]+u[i,jp(j,n_phi),k]+u[i+1,jp(j,n_phi),k]) - 
                                     rpi*v[i,j,k]*0.5*dphii*(v[i,jp(j,n_phi),k]-v[i,jm(j,n_phi),k]) -
                                     0.25*(w[i,j,k]+w[i,jp(j,n_phi),k]+0+0)*dzi*(v[i,j,k+1]-v[i,j,k]) -
                                     0.5*om*(u[i,j,k]+u[i+1,j,k]+u[i,jp(j,n_phi),k]+u[i+1,jp(j,n_phi),k]) - #
                                     rpi*dphii*(p[i,jp(j,n_phi),k]-p[i,j,k]) )

        # k=k_max                   Wasser kann nicht raus, Wasser kann nicht rein => w=0 ganz oben am Rand und "aussen"? -> Nein
        k=n_z-1
        unew[i,j,k] = u[i,j,k] + dt*(Reyi*(dri**2*(u[i+1,j,k]-2.*u[i,j,k]+u[i-1,j,k]) + rui**2*dphii**2*(u[i,jp(j,n_phi),k]-2.*u[i,j,k]+u[i,jm(j,n_phi),k]) +
                                           dzi**2*(u[i,j,k]-2.*u[i,j,k-1]+u[i,j,k-2]) -
                                           rui**2*0.5*dphii*(v[i-1,j,k]+v[i,j,k]-v[i-1,jm(j,n_phi),k]-v[i,jm(j,n_phi),k]) - rui**2*u[i,j,k] ) +
                                     (Reyi*rui-u[i,j,k])*0.5*dri*(u[i+1,j,k]-u[i-1,j,k]) +
                                     rui*(0.25*(v[i,j,k]+v[i,jm(j,n_phi),k]+v[i-1,jm(j,n_phi),k]+v[i-1,j,k]))**2 -
                                     rui*0.25*(v[i,j,k]+v[i,jm(j,n_phi),k]+v[i-1,jm(j,n_phi),k]+v[i-1,j,k])*0.5*dphii*(u[i,jp(j,n_phi),k]-u[i,jm(j,n_phi),k]) -
                                     0.25*(w[i,j,k-1]+w[i-1,j,k-1]+0+0)*dzi*(u[i,j,k]-u[i,j,k-1]) + # was mache ich mit w?
                                     0.5*om*(v[i,j,k]+v[i,jm(j,n_phi),k]+v[i-1,jm(j,n_phi),k]+v[i-1,j,k]) + (om**2)*ru[i] - dri*(p[i,j,k]-p[i-1,j,k]) )

        vnew[i,j,k] = v[i,j,k] + dt*(Reyi*(dri**2*(v[i+1,j,k]-2.*v[i,j,k]+v[i-1,j,k]) + rpi**2*dphii**2*(v[i,jp(j,n_phi),k]-2.*v[i,j,k]+v[i,jm(j,n_phi),k]) +
                                           dzi**2*(v[i,j,k]-2.*v[i,j,k-1]+v[i,j,k-2]) +
                                           rpi**2*0.5*dphii*(u[i,j,k]+u[i+1,j,k]-u[i,jp(j,n_phi),k]-u[i+1,jp(j,n_phi),k]) - rpi**2*v[i,j,k]) +
                                     (Reyi*rpi-0.25*(u[i,j,k]+u[i+1,j,k]+u[i,jp(j,n_phi),k]+u[i+1,jp(j,n_phi),k]))*0.5*dri*(v[i+1,j,k]-v[i-1,j,k]) -
                                     rpi*v[i,j,k]*0.25*(u[i,j,k]+u[i+1,j,k]+u[i,jp(j,n_phi),k]+u[i+1,jp(j,n_phi),k]) - 
                                     rpi*v[i,j,k]*0.5*dphii*(v[i,jp(j,n_phi),k]-v[i,jm(j,n_phi),k]) -
                                     0.25*(w[i,j,k-1]+w[i,jp(j,n_phi),k-1]+0+0)*dzi*(v[i,j,k]-v[i,j,k-1]) -
                                     0.5*om*(u[i,j,k]+u[i+1,j,k]+u[i,jp(j,n_phi),k]+u[i+1,jp(j,n_phi),k]) - #
                                     rpi*dphii*(p[i,jp(j,n_phi),k]-p[i,j,k]) )
### doppelte Randfälle
for j in range(n_phi):
    # i=0, k=0
    i=0
    k=0
    rui=1/ru[i]
    rpi=1/rp[i]
    unew[i,j,k] = u[i,j,k] + dt*(Reyi*(dri**2*(u[i+2,j,0]-2.*u[i+1,j,k]+u[i,j,k]) + rui**2*dphii**2*(u[i,jp(j,n_phi),k]-2.*u[i,j,k]+u[i,jm(j,n_phi),k]) +
                                       dzi**2*(u[i,j,k+2]-2.*u[i,j,k+1]+u[i,j,k]) -
                                       rui**2*dphii*0.5*(0+v[i,j,k]-0-v[i,jm(j,n_phi),k]) - rui**2*u[i,j,k] ) +
                                 (Reyi*rui-u[i,j,k])*dri*(u[i+1,j,k]-u[i,j,k]) +
                                 rui*(1/3.*(v[i,j,k]+v[i,jm(j,n_phi),k]+0))**2 -
                                 rui*1/3.*(v[i,j,k]+v[i,jm(j,n_phi),k]+0)*0.5*dphii*(u[i,jp(j,n_phi),k]-u[i,jm(j,n_phi),k]) -
                                 0.25*(w[i,j,k]+w_c[k]+0)*dzi*(u[i,j,k+1]-u[i,j,k]) +
                                 2*om*(1/3.*(v[i,j,k]+v[i,jm(j,n_phi),k]+0)) + (om**2)*ru[i] - dri*(p[i,j,k]-p_c[k]) )

    vnew[i,j,k] = v[i,j,k] + dt*(Reyi*(dri**2*(v[i+2,j,k]-2.*v[i+1,j,k]+v[i,j,k]) + rpi**2*dphii**2*(v[i,jp(j,n_phi),k]-2.*v[i,j,k]+v[i,jm(j,n_phi),k]) +
                                       dzi**2*(v[i,j,k+2]-2.*v[i,j,k+1]+v[i,j,k]) +
                                       rpi**2*0.5*dphii*(u[i,j,k]+u[i+1,j,k]-u[i,jp(j,n_phi),k]-u[i+1,jp(j,n_phi),k]) - rpi**2*v[i,j,k]) +
                                 (Reyi*rpi-0.25*(u[i,j,k]+u[i+1,j,k]+u[i,jp(j,n_phi),k]+u[i+1,jp(j,n_phi),k]))*dri*(v[i+1,j,k]-v[i,j,k]) -
                                 rpi*v[i,j,k]*0.25*(u[i,j,k]+u[i+1,j,k]+u[i,jp(j,n_phi),k]+u[i+1,jp(j,n_phi),k]) - 
                                 rpi*v[i,j,k]*0.5*dphii*(v[i,jp(j,n_phi),k]-v[i,jm(j,n_phi),k]) -
                                 0.25*(w[i,j,k]+w[i,jp(j,n_phi),k]+0+0)*dzi*(v[i,j,k+1]-v[i,j,k]) -
                                 0.5*om*(u[i,j,k]+u[i+1,j,k]+u[i,jp(j,n_phi),k]+u[i+1,jp(j,n_phi),k]) - #
                                 rpi*dphii*(p[i,jp(j,n_phi),k]-p[i,j,k]) )

    # i=i_max, k=0
    i=n_r-1
    k=0
    rui=1/ru[i]
    rpi=1/rp[i]
    unew[i,j,k] = u[i,j,k] + dt*(Reyi*(dri**2*(u[i,j,k]-2.*u[i-1,j,k]+u[i-2,j,k]) + rui**2*dphii**2*(u[i,jp(j,n_phi),k]-2.*u[i,j,k]+u[i,jm(j,n_phi),k]) +
                                       dzi**2*(u[i,j,k+2]-2.*u[i,j,k+1]+u[i,j,k]) -
                                       rui**2*dphii*0.5*(v[i-1,j,k]+v[i,j,k]-v[i-1,jm(j,n_phi),k]-v[i,jm(j,n_phi),k]) - rui**2*u[i,j,k] ) +
                                 (Reyi*rui-u[i,j,k])*dri*(u[i,j,k]-u[i-1,j,k]) +
                                 rui*(0.25*(v[i,j,k]+v[i,jm(j,n_phi),k]+v[i-1,jm(j,n_phi),k]+v[i-1,j,k]))**2 -
                                 rui*(0.25*(v[i,j,k]+v[i,jm(j,n_phi),k]+v[i-1,jm(j,n_phi),k]+v[i-1,j,k]))*0.5*dphii*(u[i,jp(j,n_phi),k]-u[i,jm(j,n_phi),k]) -
                                 0.25*(w[i,j,k]+w[i-1,j,k]+0+0)*dzi*(u[i,j,k]-u[i,j,k-1]) +
                                 2*om*(0.25*(v[i,j,k]+v[i,jm(j,n_phi),k]+v[i-1,jm(j,n_phi),k]+v[i-1,j,k])) + (om**2)*ru[i] - dri*(p[i,j,k]-p[i-1,j,k]) )  
   # vnew[i,j,k] = 0

    # i=0, k=kmax
    i=0
    k=n_z-1
    rui=1/ru[i]
    rpi=1/rp[i]
    unew[i,j,k] = u[i,j,k] + dt*(Reyi*(dri**2*(u[i+2,j,0]-2.*u[i+1,j,k]+u[i,j,k]) + rui**2*dphii**2*(u[i,jp(j,n_phi),k]-2.*u[i,j,k]+u[i,jm(j,n_phi),k]) +
                                       dzi**2*(u[i,j,k]-2.*u[i,j,k-1]+u[i,j,k-2]) -
                                       rui**2*dphii*0.5*(0+v[i,j,k]-0-v[i,jm(j,n_phi),k]) - rui**2*u[i,j,k] ) +
                                 (Reyi*rui-u[i,j,k])*dri*(u[i+1,j,k]-u[i,j,k]) +
                                 rui*(1/3.*(v[i,j,k]+v[i,jm(j,n_phi),k]+0))**2 -
                                 rui*1/3.*(v[i,j,k]+v[i,jm(j,n_phi),k]+0)*0.5*dphii*(u[i,jp(j,n_phi),k]-u[i,jm(j,n_phi),k]) -
                                 0.25*(w[i,j,k-1]+w_c[k-1]+0)*dzi*(u[i,j,k]-u[i,j,k-1]) +
                                 2*om*(1/3.*(v[i,j,k]+v[i,jm(j,n_phi),k]+0)) + (om**2)*ru[i] - dri*(p[i,j,k]-p_c[k]) )

    vnew[i,j,k] = v[i,j,k] + dt*(Reyi*(dri**2*(v[i+2,j,k]-2.*v[i+1,j,k]+v[i,j,k]) + rpi**2*dphii**2*(v[i,jp(j,n_phi),k]-2.*v[i,j,k]+v[i,jm(j,n_phi),k]) +
                                       dzi**2*(v[i,j,k]-2.*v[i,j,k-1]+v[i,j,k-2]) +
                                       rpi**2*0.5*dphii*(u[i,j,k]+u[i+1,j,k]-u[i,jp(j,n_phi),k]-u[i+1,jp(j,n_phi),k]) - rpi**2*v[i,j,k]) +
                                 (Reyi*rpi-0.25*(u[i,j,k]+u[i+1,j,k]+u[i,jp(j,n_phi),k]+u[i+1,jp(j,n_phi),k]))*dri*(v[i+1,j,k]-v[i,j,k]) -
                                 rpi*v[i,j,k]*0.25*(u[i,j,k]+u[i+1,j,k]+u[i,jp(j,n_phi),k]+u[i+1,jp(j,n_phi),k]) - 
                                 rpi*v[i,j,k]*0.5*dphii*(v[i,jp(j,n_phi),k]-v[i,jm(j,n_phi),k]) -
                                 0.25*(w[i,j,k-1]+w[i,jp(j,n_phi),k-1]+0+0)*dzi*(v[i,j,k]-v[i,j,k-1]) -
                                 0.5*om*(u[i,j,k]+u[i+1,j,k]+u[i,jp(j,n_phi),k]+u[i+1,jp(j,n_phi),k]) - #
                                 rpi*dphii*(p[i,jp(j,n_phi),k]-p[i,j,k]) )

    # i=i_max, k=k_max
    i=n_r-1
    k=n_z-1
    rui=1/ru[i]
    rpi=1/rp[i]
    unew[i,j,k] = u[i,j,k] + dt*(Reyi*(dri**2*(u[i,j,0]-2.*u[i-1,j,k]+u[i-2,j,k]) + rui**2*dphii**2*(u[i,jp(j,n_phi),k]-2.*u[i,j,k]+u[i,jm(j,n_phi),k]) +
                                       dzi**2*(u[i,j,k]-2.*u[i,j,k-1]+u[i,j,k-2]) -
                                       rui**2*dphii*0.5*(v[i-1,j,k]+v[i,j,k]-v[i-1,jm(j,n_phi),k]-v[i,jm(j,n_phi),k]) - rui**2*u[i,j,k] ) +
                                 (Reyi*rui-u[i,j,k])*dri*(u[i,j,k]-u[i-1,j,k]) +
                                 rui*(0.25*(v[i,j,k]+v[i,jm(j,n_phi),k]+v[i-1,j,k]+v[i-1,jm(j,n_phi),k]))**2 -
                                 rui*0.25*(v[i,j,k]+v[i,jm(j,n_phi),k]+v[i-1,j,k]+v[i-1,jm(j,n_phi),k])*0.5*dphii*(u[i,jp(j,n_phi),k]-u[i,jm(j,n_phi),k]) -
                                 0.25*(0+0+w[i,j,k-1]+w[i-1,j,k-1])*dzi*(u[i,j,k]-u[i,j,k-1]) +
                                 2*om*(0.25*(v[i,j,k]+v[i,jm(j,n_phi),k]+v[i-1,j,k]+v[i-1,jm(j,n_phi),k])) + (om**2)*ru[i] - dri*(p[i,j,k]-p[i-1,j,k]) )
    #vnew[i,j,k] = 0

### Berechung von w aus u,v
for j in range(n_phi):
    for k in range(n_z-1):
        for i in range(n_r-1):
            rui=1/ru[i]
            rpi=1/rp[i]
    
            wnew[i,j,k] = w[i,j,k] - dz*(rpi*0.25*(unew[i,j,k]+unew[i,j,k+1]+unew[i+1,j,k]+unew[i+1,j,k+1]) +
                                         dri*0.5*(unew[i+1,j,k]+unew[i+1,j,k+1]-unew[i,j,k]-unew[i,j,k+1]) +
                                         dphii*0.5*(vnew[i,jp(j,n_phi),k]+vnew[i,jp(j,n_phi),k+1]-vnew[i,j,k]-vnew[i,j,k+1]) )
        i=n_r-1
        w[i,j,k] = w[i,j,k] - dz*(rpi*0.25*(unew[i,j,k]+unew[i,j,k+1]+0+0) +
                                         dri*0.5*(0+0-unew[i,j,k]-unew[i,j,k+1]) +
                                         dphii*0.5*(vnew[i,jp(j,n_phi),k]+vnew[i,jp(j,n_phi),k+1]-vnew[i,j,k]-vnew[i,j,k+1]) )
    
### neues w im Zentrum
for k in range(n_z-1):
    rui=1/ru[0]         # nehme ru, da dies näher an der Mitte ist
    rpi=1/rp[0]
    i=0
    j=1                 # es ist egal, was ich für j wähle
    w_c_new[k] = w_c[k] - dz*(rui*0.25*(unew[i,j,k]+unew[i,j+int(n_phi/2),k]+unew[i,j,k+1]+unew[i,j+int(n_phi/2),k+1]) +
                              dri*0.5*(unew[i,j,k]+unew[i,j,k+1]+unew[i,j-unew[i,j-int(n_phi/2),k]-int(n_phi/2),k+1]) + 0) # v im Mittelpunkt ist 0, phi-Ableitung im Mittelpunkt macht keinen Sinn

u = unew.copy()
v = vnew.copy()
w = wnew.copy()
w_c = w_c_new.copy()

print("mit grad(div(u))")
print("u(r):", unew[:,1,1])
print("u(phi):", unew[1,:,1])
print("u(z):", unew[1,1,:])
print("v(r):", vnew[:,1,1])
print("v(phi):", vnew[1,:,1])
print("v(z):", vnew[1,1,:])
print("w(r):", wnew[:,1,1])
print("w(phi):", wnew[1,:,1])
print("w(z):", wnew[1,1,:])
print("w_c(z):",w_c[:])

## Drucknachregelung
div_max = np.amax(div_u)
count=0
#while (div_max > precision) or count=0:
while div_max < 1e10: 

### Divergenz von (u,v,w) auf p-Gitter
    for i in range(1,n_r):        #Divergenz startet im Zentrum => i ist immer eins höher als für p
        rpi=1/rp[i-1]
        for j in range(n_phi):
            for k in range(1,n_z-1):
                div_u[i,j,k] = (rpi*0.5*(u[i-1,j,k]+u[i,j,k]) + dri*(u[i,j,k]-u[i-1,j,k]) + rpi*dphii*(v[i-1,j,k]-v[i-1,jm(j,n_phi),k]) +
                                dzi*(w[i-1,j,k]-w[i-1,j,k-1]) )
            
    # Randwerte für i
    for j in range(n_phi):
        for k in range(1,n_z-1):
            div_u[n_r,j,k] = (1/rp[n_r-1]*(u[n_r-1,j,k]+0) + dri*(-u[n_r-1,j,k]) + 1/rp[n_r-1]*dphii*(v[n_r-1,j,k]-v[n_r-1,jm(j,n_phi),k]) +
                              dzi*(w[n_r-1,j,k]-w[n_r-1,j,k-1]) )
    for k in range(1,n_z-1):
        div_u[0,0,k] = 1/ru[0]*0.5*(u[0,0,k]+u[0,int(n_phi/2),k]) + dri*(u[0,0,k]-u[0,int(n_phi/2),k]) + dzi*(w_c[k]-w_c[k-1])
        for j in range(n_phi):
            div_u[0,j,k]=div_u[0,0,k]

    # Randwerte für k
    for i in range(1,n_r):
        rpi=1/rp[i]
        for j in range(n_phi):  
            div_u[i,j,n_z-1] = (rpi*0.5*(u[i-1,j,n_z-1]+u[i,j,n_z-1]) + dri*(u[i,j,n_z-1]-u[i-1,j,n_z-1]) +
                                rpi*dphii*(v[i-1,j,n_z-1]-v[i-1,jm(j,n_phi),n_z-1]) + dzi*(-w[i-1,j,n_z-2]) )               # nehme für w(n_z-1) 0 an
            div_u[i,j,0] = (rpi*0.5*(u[i-1,j,0]+u[i,j,0]) + dri*(u[i,j,0]-u[i-1,j,0]) + rpi*dphii*(v[i-1,j,0]-v[i-1,jm(j,n_phi),0]) + dzi*(w[i-1,j,0]) )
    # Doppelte Fälle
    div_u[0,0,0] = 1/ru[0]*0.5*(u[0,0,0]+u[0,int(n_phi/2),0]) + dri*(u[0,0,0]-u[0,int(n_phi/2),0]) + dzi*(w_c[0])
    div_u[0,0,n_z-1] = 1/ru[0]*0.5*(u[0,0,n_z-1]+u[0,int(n_phi/2),n_z-1]) + dri*(u[0,0,n_z-1]-u[0,int(n_phi/2),n_z-1]) + dzi*(-w_c[n_z-2])
    for j in range(n_phi):
        div_u[0,j,0] = div_u[0,0,0]
        div_u[0,j,n_z-1] = div_u[0,0,n_z-1]
        div_u[n_r,j,0] = 1/rp[n_r-1]*(u[n_r-1,j,0]+0) + dri*(0-u[n_r-1,j,0]) + 1/rp[n_r-1]*dphii*(v[n_r-1,j,0]-v[n_r-1,jm(j,n_phi),0]) +dzi*(w[n_r-1,j,0])
        div_u[n_r,j,n_z-1] = (1/rp[n_r-1]*(u[n_r-1,j,n_z-1]+0) + dri*(-u[n_r-1,j,n_z-1]) + 1/rp[n_r-1]*dphii*(v[n_r-1,j,n_z-1]-v[n_r-1,jm(j,n_phi),n_z-1]) +
                              dzi*(-w[n_r-1,j,n_z-2]) )


# Anpassung u,v,w und p
    for i in range(n_r):    # kann nicht bei 0 starten, da rp[0]=0 => rpi=nan !
        rpi=1/rp[i]
        for j in range(n_phi):
            for k in range(n_z):
                pnew[i,j,k] = p[i,j,k] - lamb*div_u[i+1,j,k]
                unew[i,j,k] = u[i,j,k] + dt*lamb*dri*(div_u[i+1,j,k]-div_u[i,j,k])  
                vnew[i,j,k] = v[i,j,k] + dt*lamb*rpi*dphii*(div_u[i+1,jp(j,n_phi),k]-div_u[i+1,j,k])
            for k in range(n_z-1):
                wnew[i,j,k] = w[i,j,k] + dt*lamb*dri*(div_u[i+1,j,k+1]-div_u[i+1,j,k])
    for k in range(n_z-1):
        w_c_new[k] = w_c[k] + dt*lamb*dri*(div_u[0,0,k+1]-div_u[0,0,k])
            
    print("während der Nachregelung", count)
    print("p(r):", pnew[:,1,1])
    print("div(r):", div_u[:,1,1])
    print("div(phi):", div_u[0,:,1])
    print("div(z):", div_u[1,1,:])

    print("udiff", unew[:,1,1]-u[:,1,1])
    print("vdiff", vnew[:,1,1]-v[:,1,1])
    print("wdiff", wnew[:,1,1]-w[:,1,1])

    p = pnew.copy()
    u = unew.copy()
    v = vnew.copy()
    w = wnew.copy()
    div_max = np.amax(div_u)
    count=count+1
    #print("u(r):", unew[:,1,1])
    #print("v(r):", vnew[:,1,1])
    
print("nach der Nachregelung")
print("div(r):", div_u[:,1,1])
