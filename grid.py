import math
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt


pi=math.pi
# u Radialkomponente der Geschwindigkeit, v Azimutalkomponente der Geschwindigkeit, p Druck
# Tankgrenzen bzw. Tankraender
r_t=8
h_t=1

# Anzahl Gitterpunkte
n_r,n_phi,n_z = 50,50,25

# p-Gitter p(r,phi,z):
r_p = np.linspace(r_t/n_r,r_t, n_r)
phi_p = np.linspace(pi/2,2*pi+pi/2, n_phi)
z_p = np.linspace(0,h_t, n_z)

p_grid = [r_p,phi_p,z_p]

# u-Gitter u(r,phi,z):
r_u = np.linspace(r_t/(2*n_r),r_t-r_t/(2*n_r), n_r)
phi_u = np.linspace(pi/2,2*pi+pi/2, n_phi)
z_u = np.linspace(0,h_t, n_z)

u_grid = [r_u,phi_u,z_u]

# v-Gitter v(r,phi,z):
r_v = np.linspace(r_t/n_r,r_t, n_r)
phi_v = np.linspace(0,2*pi, n_phi)
z_v = np.linspace(0,h_t, n_z)

v_grid = [r_v,phi_v,z_v]

plt.polar(phi_p,r_p)
plt.show()




