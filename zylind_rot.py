import math

r=1
h=1
steps=16

for phi in range(steps+1):
    x=r*math.cos(phi/(steps)*2*math.pi)
    y=r*math.sin(phi/(steps)*2*math.pi)
    z=h
    point=[x,y,z]

    print('({0:.3f},{1:.3f},{2:.3f})'.format(point[0],point[1],point[2]))
    
