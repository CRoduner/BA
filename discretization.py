# discretization

dri=1/dr
dphii=1/dphi
dzi=1/dz

ri=1/r[i]



unew[i,j,k] = u[i,j,k] + dt*(
     dri^2*(u[i+1,j,k]-2.*u[i,j,k]+u[i-1,j,k]) + ri^2*dphii^2*(u[i,j+1,k]-2.*u[i,j,k]+u[i,j-1,k]) + dzi^2*(u[i,j,k+1]-2.*u[i,j,k]+u[i,j,k-1]) +
     (ri-u[i,j,k])*0.5*dri*(u[i+1,j,k]-u[i-1,j,k]) - ri*0.25*(v[i,j,k]+v[i,j-1,k]+v[i+1,j-1,k]+v[i+1,j,k])*0.5*dphii*(u[i,j+1,k]-u[i,j-1,k]) -
     0.25*(w[i,j,k]+w[i+1,j,k]+w[i,j,k-1]+w[i+1,j,k-1])*0.5*dzi*(u[i,j,k+1]-u[u,j,k-1]) - ri^2*dphii*(v[i+1,j,k]+v[i,j,k]-v[i+1,j-1,k]-v[i,j-1,k]) -
     ri^2*u[i,j,k] + 0.5*omega*(v[i,j,k]+v[i,j-1,k]+v[i+1,j-1,k]+v[i+1,j,k]) + omega^2*r[i] - dri*(p[i+1,j,k]-p[i,j,k]) )

vnew[i,j,k] = v[i,j,k] + dt*(
    dri^2*(v[i+1,j,k]-2.*v[i,j,k]+v[i-1,j,k]) + ri^2*dphii^2*(v[i,j+1,k]-2.*v[i,j,k]+u[i,j-1,k]) + dzi^2*(v[i,j,k+1]-2.*v[i,j,k]+v[i,j,k-1]) +
    (ri-0.25*(u[i,j,k]+u[i-1,j,k]+u[i,j+1,k]+u[i-1,j+1,k]))*0.5*dri*(v[i+1,j,k]-v[i-1,j,k]) - ri*v[i,j,k]*0.5*dphii*(v[i,j+1,k]-v[i,j-1,k]) -
    0.25*(w[i,j,k]+w[i,j+1,k]+w[i,j,k-1]+w[i,j+1,k-1])*0.5*dzi*(v[i,j,k+1]-v[i,j,k-1]) + ri^2*dphii*(u[i,j,k]+u[i-1,j,k]-u[i,j+1,k]-u[i-1,j+1,k]) -
    ri^2*v[i,j,k] - 0.5*omega*(u[i,j,k]+u[i-1,j,k]+u[i,j+1,k]+u[i-1,j+1,k]) - ri*dphii*(p[i,j+1,k]-p[i,j,k]) )

wnew[i,j,k] = w[i,j,k] + dt*(
    dri^2*(w[i+1,j,k]-2.*w[i,j,k]+w[i-1,j,k]) + ri^2*dphii^2*(w[i,j+1,k]-2.*w[i,j,k]+w[i,j-1,k]) + dzi^2*(w[i,j,k+1]-2.*w[i,j,k]+w[i,j,k-1]) +
    (ri-0.25*(u[i,j,k]+u[i-1,j,k]+u[i,j,k+1]+u[i-1,j,k+1]))*0.5*dri*(w[i+1,j,k]-w[i-1,j,k]) -
    ri*0.25(v[i,j,k]+v[i,j-1,k]+v[i,j,k+1]+v[i,j-1,k+1])*0.5*dphii*(w[i,j+1,k]-w[i,j-1,k]) - w[i,j,k]*0.5*dzi*(w[i,j,k+1]-w[i,j,k-1]) -
    dzi*(p[i,j,k+1]-p[i,j,k]) - g )

