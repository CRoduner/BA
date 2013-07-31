import math

r=1
h=1
angle_steps=16
radius_steps=5
height_steps=5

for phi in range(angle_steps+1):
    for n in range(radius_steps):
        r=(n+1)/radius_steps
        x=r*math.cos(phi/(angle_steps)*2*math.pi)
        y=r*math.sin(phi/(angle_steps)*2*math.pi)
        z=h
        point=[x,y,z]

        print('({0:.3f},{1:.3f},{2:.3f})'.format(point[0],point[1],point[2]), end=',')
    print(end='\n')
