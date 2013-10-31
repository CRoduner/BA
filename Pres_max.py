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
lamb=1.5             # Koeffizient für Drucknachregelung lambda

# time steps
dt = 0.002

# number of grid points
n_r = 20
n_phi = 20
n_z = 10

# stepsizes
dr = r_t/n_r
dphi = 2*pi/n_phi
dz = h_t/n_z

precision=10**(-5)  # Gewünschte Präzision der Divergenzfreiheit

## Abkürzungen, Dimensionen
dri=1/dr
dphii=1/dphi
dzi=1/dz


# r-vector
ru = np.linspace(dr-dr/2,r_t-dr/2, n_r)
rp = np.linspace(0,r_t, n_r)


# matrix initialization
u = np.zeros((n_r-1,n_phi,n_z))      
unew = np.zeros((n_r-1,n_phi,n_z))

v = np.zeros((n_r,n_phi,n_z))
vnew = np.zeros((n_r,n_phi,n_z))

w = np.zeros((n_r,n_phi,n_z-1))
wnew = np.zeros((n_r,n_phi,n_z-1))

p = np.ones((n_r,n_phi,n_z))    # Nachregelung?
pnew = np.zeros((n_r,n_phi,n_z))

div_u = np.zeros((n_r,n_phi,n_z))


# Zeitschlaufe

## Intitialisierung der Randwerte / Spezialfälle:

### Randwerte i=0, i=n_r-1 für v,p und i=n_r-1 für w(Werte noch nicht bestimmt)
for j in range(n_phi):
    for k in range (1,n_z):
        v[n_r-1,j,k] = 0
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
        rui=1/ru[0]
        unew[0,j,k] = u[0,j,k] + dt*(dri**2*(u[2,j,k]-2.*u[1,j,k]+u[0,j,k]) + rui**2*dphii**2*(u[0,jp,k]-2.*u[0,j,k]+u[0,jm,k]) +
                                     dzi**2*(u[0,j,k+1]-2.*u[0,j,k]+u[0,j,k-1]) + (rui-u[0,j,k])*dri*(u[1,j,k]-u[0,j,k]) -
                                     rui/3.*(v[0,j,k]+v[1,jm,k]+v[1,j,k])*0.5*dphii*(u[0,jp,k]-u[0,jm,k]) -
                                     0.25*(w[0,j,k]+w[1,j,k]+w[0,j,k-1]+w[1,j,k-1])*0.5*dzi*(u[0,j,k+1]-u[0,j,k-1]) -
                                     rui**2*dphii*(v[1,j,k]+v[0,j,k]-v[1,jm,k]-v[0,jm,k]) - rui**2*u[0,j,k] +
                                     0.5*om*(v[0,j,k]+v[0,jm,k]+v[1,jm,k]+v[1,j,k]) + (om**2)*ru[0] - dri*(p[1,j,k]-p[0,j,k]) )
        rui=1/ru[n_r-2]
        unew[n_r-2,j,k] = u[n_r-2,j,k] + dt*(dri**2*(u[n_r-2,j,k]-2.*u[n_r-3,j,k]+u[n_r-4,j,k]) + rui**2*dphii**2*(u[n_r-2,jp,k]-2.*u[n_r-2,j,k]+u[n_r-2,jm,k]) +
                                             dzi**2*(u[n_r-2,j,k+1]-2.*u[n_r-2,j,k]+u[n_r-2,j,k-1]) + (rui-u[n_r-2,j,k])*dri*(u[n_r-2,j,k]-u[n_r-3,j,k]) -
                                             rui*0.25*(v[n_z-2,j,k]+v[n_z-2,jm,k]+v[n_z-3,jm,k]+v[n_z-3,j,k])*0.5*dphii*(u[n_z-2,jp,k]-u[n_z-2,jm,k]) -
                                             0.25*(w[n_z-2,j,k]+w[n_z-3,j,k]+w[n_z-2,j,k-1]+w[n_z-3,j,k-1])*0.5*dzi*(u[n_z-2,j,k+1]-u[n_z-2,j,k-1]) -
                                             rui**2*dphii*(v[n_z-3,j,k]+v[n_z-2,j,k]-v[n_z-3,jm,k]-v[n_z-2,jm,k]) - rui**2*u[n_z-2,j,k] +
                                             0.5*om*(v[n_z-2,j,k]+v[n_z-2,jm,k]+v[n_z-3,jm,k]+v[n_z-3,j,k]) + (om**2)*ru[n_z-2] - dri*(p[n_z-2,j,k]-p[n_z-3,j,k]) )
    for k in range(1,n_z-2):
        #rpi=1/rp[0]         # Radius hier ist Null... Durch Null teilen...
        rpi=10**(-10)
        wnew[0,j,k] = w[0,j,k] + dt*(dri**2*(w[2,j,k]-2.*w[1,j,k]+w[0,j,k]) + rpi**2*dphii**2*(w[0,jp,k]-2.*w[0,j,k]+w[0,jm,k]) +
                                     dzi**2*(w[0,j,k+1]-2.*w[0,j,k]+w[0,j,k-1]) +
                                     (rpi-0.25*(u[1,j,k]+u[0,j,k]+u[1,j,k+1]+u[0,j,k+1]))*dri*(w[1,j,k]-w[0,j,k]) -
                                     rpi/3.*(v[0,j,k]+v[0,j,k+1]+v[0,jm,k+1])*0.5*dphii*(w[0,jp,k]-w[0,jm,k]) -
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
    rpi=1/rp[i]
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

        wnew[i,j,0] = w[i,j,0] + dt*(dri**2*(w[i+1,j,0]-2.*w[i,j,0]+w[i-1,j,0]) + rpi**2*dphii**2*(w[i,jp,0]-2.*w[i,j,0]+w[i,jm,0]) +
                                     dzi**2*(w[i,j,2]-2.*w[i,j,1]+w[i,j,0]) +
                                     (rpi-0.25*(u[i,j,0]+u[i-1,j,0]+u[i,j,1]+u[i-1,j,1]))*0.5*dri*(w[i+1,j,0]-w[i-1,j,0]) -
                                     rpi*0.25*(v[i,j,0]+v[i,jm,0]+v[i,j,1]+v[i,jm,1])*0.5*dphii*(w[i,jp,0]-w[i,jm,0]) -
                                     w[i,j,0]*dzi*(w[i,j,1]-w[i,j,0]) - dzi*(p[i,j,1]-p[i,j,0]) - g )
        
        wnew[i,j,n_z-2] = w[i,j,n_z-2] + dt*(dri**2*(w[i+1,j,n_z-2]-2.*w[i,j,n_z-2]+w[i-1,j,n_z-2]) +
                                             rpi**2*dphii**2*(w[i,jp,n_z-2]-2.*w[i,j,n_z-2]+w[i,jm,n_z-2]) +
                                             dzi**2*(w[i,j,n_z-4]-2.*w[i,j,n_z-3]+w[i,j,n_z-2]) +
                                             (rpi-0.25*(u[i,j,n_z-2]+u[i-1,j,n_z-2]+u[i,j,n_z-3]+u[i-1,j,n_z-3]))*0.5*dri*(w[i+1,j,n_z-2]-w[i-1,j,n_z-2]) -
                                             rpi*0.25*(v[i,j,n_z-2]+v[i,jm,n_z-2]+v[i,j,n_z-3]+v[i,jm,n_z-3])*0.5*dphii*(w[i,jp,n_z-2]-w[i,jm,n_z-2]) -
                                             w[i,j,n_z-2]*dzi*(w[i,j,n_z-2]-w[i,j,n_z-3]) - dzi*(p[i,j,n_z-2]-p[i,j,n_z-3]) - g )

        vnew[i,j,n_z-1] = v[i,j,n_z-1] + dt*(dri**2*(v[i+1,j,n_z-1]-2.*v[i,j,n_z-1]+v[i-1,j,n_z-1]) +
                                             rpi**2*dphii**2*(v[i,jp,n_z-1]-2.*v[i,j,n_z-1]+u[i,jm,n_z-1]) +
                                             dzi**2*(v[i,j,n_z-1]-2.*v[i,j,n_z-2]+v[i,j,n_z-3]) +
                                             (rpi-0.25*(u[i,j,n_z-1]+u[i-1,j,n_z-1]+u[i,jp,n_z-1]+u[i-1,jp,n_z-1]))*0.5*dri*(v[i+1,j,n_z-1]-v[i-1,j,n_z-1]) -
                                             rpi*v[i,j,n_z-1]*0.5*dphii*(v[i,jp,n_z-1]-v[i,jm,n_z-1]) -
                                             0.25*(0+0+w[i,j,n_z-2]+w[i,jp,n_z-2])*0.5*dzi*(v[i,j,n_z-1]-v[i,j,n_z-2]) + # Vorfaktor 0.5 oder 0.25 / w ganz aussen nul setzen bzw. vernachlässigen
                                             rpi**2*dphii*(u[i,j,n_z-1]+u[i-1,j,n_z-1]-u[i,jp,n_z-1]-u[i-1,jp,n_z-1]) - rpi**2*v[i,j,n_z-1] -
                                             0.5*om*(u[i,j,n_z-1]+u[i-1,j,n_z-1]+u[i,jp,n_z-1]+u[i-1,jp,n_z-1]) - rpi*dphii*(p[i,jp,n_z-1]-p[i,j,n_z-1]) )


for i in range(1,n_r-2):
    rui=1/ru[i]
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
                                             rui**2*dphii**2*(u[i,jp,n_z-1]-2.*u[i,j,n_z-1]+u[i,jm,n_z-1]) +
                                             dzi**2*(u[i,j,n_z-1]-2.*u[i,j,n_z-2]+u[i,j,n_z-3]) + (rui-u[i,j,n_z-1])*0.5*dri*(u[i+1,j,n_z-1]-u[i-1,j,n_z-1]) -
                                             rui*0.25*(v[i,j,n_z-1]+v[i,jm,n_z-1]+v[i+1,jm,n_z-1]+v[i+1,j,n_z-1])*0.5*dphii*(u[i,jp,n_z-1]-u[i,jm,n_z-1]) -
                                             0.25*(0+0+w[i,j,n_z-2]+w[i+1,j,n_z-2])*0.5*dzi*(u[i,j,n_z-1]-u[i,j,n_z-2]) - # Vorfaktor 0.5 oder 0.25 -> wie v
                                             rui**2*dphii*(v[i+1,j,n_z-1]+v[i,j,n_z-1]-v[i+1,jm,n_z-1]-v[i,jm,n_z-1]) - rui**2*u[i,j,n_z-1] +
                                             0.5*om*(v[i,j,n_z-1]+v[i,jm,n_z-1]+v[i+1,jm,n_z-1]+v[i+1,j,n_z-1]) +
                                             (om**2)*ru[i] - dri*(p[i+1,j,n_z-1]-p[i,j,n_z-1]) )

        
## allgemeiner Fall
for i in range(1,n_r-2):
    rui=1/ru[i]
    rpi=1/rp[i]
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
            
            unew[i,j,k] = u[i,j,k] + dt*(dri**2*(u[i+1,j,k]-2.*u[i,j,k]+u[i-1,j,k]) + rui**2*dphii**2*(u[i,jp,k]-2.*u[i,j,k]+u[i,jm,k]) +
                                         dzi**2*(u[i,j,k+1]-2.*u[i,j,k]+u[i,j,k-1]) + (rui-u[i,j,k])*0.5*dri*(u[i+1,j,k]-u[i-1,j,k]) -
                                         rui*0.25*(v[i,j,k]+v[i,jm,k]+v[i+1,jm,k]+v[i+1,j,k])*0.5*dphii*(u[i,jp,k]-u[i,jm,k]) -
                                         0.25*(w[i,j,k]+w[i+1,j,k]+w[i,j,k-1]+w[i+1,j,k-1])*0.5*dzi*(u[i,j,k+1]-u[i,j,k-1]) -
                                         rui**2*dphii*(v[i+1,j,k]+v[i,j,k]-v[i+1,jm,k]-v[i,jm,k]) - rui**2*u[i,j,k] +
                                         0.5*om*(v[i,j,k]+v[i,jm,k]+v[i+1,jm,k]+v[i+1,j,k]) + (om**2)*ru[i] - dri*(p[i+1,j,k]-p[i,j,k]) )

            vnew[i,j,k] = v[i,j,k] + dt*(dri**2*(v[i+1,j,k]-2.*v[i,j,k]+v[i-1,j,k]) + rpi**2*dphii**2*(v[i,jp,k]-2.*v[i,j,k]+u[i,jm,k]) +
                                         dzi**2*(v[i,j,k+1]-2.*v[i,j,k]+v[i,j,k-1]) +
                                         (rpi-0.25*(u[i,j,k]+u[i-1,j,k]+u[i,jp,k]+u[i-1,jp,k]))*0.5*dri*(v[i+1,j,k]-v[i-1,j,k]) -
                                         rpi*v[i,j,k]*0.5*dphii*(v[i,jp,k]-v[i,jm,k]) -
                                         0.25*(w[i,j,k]+w[i,jp,k]+w[i,j,k-1]+w[i,jp,k-1])*0.5*dzi*(v[i,j,k+1]-v[i,j,k-1]) +
                                         rpi**2*dphii*(u[i,j,k]+u[i-1,j,k]-u[i,jp,k]-u[i-1,jp,k]) - rpi**2*v[i,j,k] -
                                         0.5*om*(u[i,j,k]+u[i-1,j,k]+u[i,jp,k]+u[i-1,jp,k]) - rpi*dphii*(p[i,jp,k]-p[i,j,k]) )

            wnew[i,j,k] = w[i,j,k] + dt*(dri**2*(w[i+1,j,k]-2.*w[i,j,k]+w[i-1,j,k]) + rpi**2*dphii**2*(w[i,jp,k]-2.*w[i,j,k]+w[i,jm,k]) +
                                         dzi**2*(w[i,j,k+1]-2.*w[i,j,k]+w[i,j,k-1]) +
                                         (rpi-0.25*(u[i,j,k]+u[i-1,j,k]+u[i,j,k+1]+u[i-1,j,k+1]))*0.5*dri*(w[i+1,j,k]-w[i-1,j,k]) -
                                         rpi*0.25*(v[i,j,k]+v[i,jm,k]+v[i,j,k+1]+v[i,jm,k+1])*0.5*dphii*(w[i,jp,k]-w[i,jm,k]) -
                                         w[i,j,k]*0.5*dzi*(w[i,j,k+1]-w[i,j,k-1]) - dzi*(p[i,j,k+1]-p[i,j,k]) - g )
u = unew
v = vnew
w = wnew

print("u:", unew[:,1,1])
print("v:", vnew[:,1,1])
print("w:", wnew[:,1,1])

## Drucknachregelung
### Divergenz von (u,v,w) auf p-Gitter
for i in range(1,n_r-1):
    rpi=1/rp[i]
    for j in range(n_phi):
        if j==0:
            jm = n_phi-1
        else:
            jm = j-1
            
        for k in range(1,n_z-2):
            div_u[i,j,k] = rpi*0.5*(u[i-1,j,k]+u[i,j,k]) + dri*(u[i,j,k]-u[i-1,j,k]) + rpi*dphii*(v[i,j,k]-v[i,jm,k]) + dzi*(w[i,j,k+1]-w[i,j,k])
    #print(rpi*0.5*(u[i-1,2,4]+u[i,2,4]), dri*(u[i,2,4]-u[i-1,2,4]), rpi*dphii*(v[i,2,4]-v[i,1,4]), dzi*(w[i,2,5]-w[i,2,4]) )
            
# Randwerte für i
for j in range(n_phi):
    if j==0:
        jm = n_phi-1
    else:
        jm = j-1
        
    for k in range(1,n_z-2):
        # nehme für u den innerst möglichen Wert
        div_u[0,j,k] = rpi*(u[0,j,k]) + dri*(u[1,j,k]-u[0,j,k]) + rpi*dphii*(v[0,j,k]-v[0,jm,k]) + dzi*(w[0,j,k+1]-w[0,j,k])
        # nehme für u den äusserst möglichen Wert
        div_u[n_r-1,j,k] = rpi*(u[n_r-2,j,k]) + dri*(u[n_r-2,j,k]-u[n_r-3,j,k]) + rpi*dphii*(v[n_r-1,j,k]-v[n_r-1,jm,k]) + dzi*(w[n_r-1,j,k+1]-w[n_r-1,j,k])
# Randwerte für k (...)
        

print("vor Nachregelung")
print("div(r):", div_u[:,1,1])
print("p(r):", p[:,1,1])
div_max = np.amax(div_u)

count=0
while div_max > precision:
    for i in range(1,n_r-1):    # kann nicht bei 0 starten, da rp[0]=0 => rpi=nan !
        rpi=1/rp[i]
        for j in range(n_phi):
            if j==0:
                jm = n_phi-1
            else:
                jm = j-1
                
            for k in range(n_z-2):
                pnew[i,j,k] = p[i,j,k] - lamb*div_u[i,j,k]  #+/- lambda
                unew[i,j,k] = u[i,j,k] + dt*(lamb*dri*(div_u[i+1,j,k]-div_u[i,j,k]))    # +/- ?
                vnew[i,j,k] = v[i,j,k] + dt*(lamb*rpi*dphii*(div_u[i,jp,k]-div_u[i,j,k]))
                wnew[i,j,k] = w[i,j,k] + dt*(lamb*dri*(div_u[i,j,k+1]-div_u[i,j,k]))
                #print("p", i,j,k, pnew[i,j,k], div_u[i,j,k])
                

    for i in range(1,n_r-1):
        rpi=1/rp[i]
        for j in range(n_phi):
            if j==0:
                jm = n_phi-1
            else:
                jm = j-1
                
            for k in range(n_z-2):
                div_u[i,j,k] = (rpi*0.5*(unew[i-1,j,k]+unew[i,j,k]) + dri*(unew[i,j,k]-unew[i-1,j,k]) + rpi*dphii*(vnew[i,j,k]-vnew[i,jm,k]) +
                                dzi*(wnew[i,j,k+1]-wnew[i,j,k]) )
    # Randwerte i=0, i=n_r-1
    for j in range(n_phi):
        if j==0:
            jm = n_phi-1
        else:
            jm = j-1
            
        for k in range(1,n_z-2):
            # nehme für u den innerst möglichen Wert
            div_u[0,j,k] = rpi*(unew[0,j,k]) + dri*(unew[1,j,k]-unew[0,j,k]) + rpi*dphii*(vnew[0,j,k]-vnew[0,jm,k]) + dzi*(wnew[0,j,k+1]-wnew[0,j,k])
            # nehme für u den äusserst möglichen Wert
            div_u[n_r-1,j,k] = (rpi*(unew[n_r-2,j,k]) + dri*(unew[n_r-2,j,k]-unew[n_r-3,j,k]) + rpi*dphii*(vnew[n_r-1,j,k]-vnew[n_r-1,jm,k]) +
                                dzi*(wnew[n_r-1,j,k+1]-wnew[n_r-1,j,k]) )
    #Randwerte k=0, k=n_z-1 (...)

    p = pnew
    u = unew
    v = vnew
    w = wnew
    div_max = np.amax(div_u)
    count=count+1
    print("während der Nachregelung", count)
    print("u(r):", unew[:,1,1])
    print("v(r):", vnew[:,1,1])
    print("p(r):", pnew[:,1,1])
    print("div(r):", div_u[:,1,1])
print("nach der Nachregelung")
print("div(r):", div_u[:,1,1])
