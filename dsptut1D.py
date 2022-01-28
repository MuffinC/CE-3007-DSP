# importing the required module
import matplotlib.pyplot as plt
import numpy as np

n = np.arange(-20,20,1)
y1 = 3 * np.cos(np.pi*n/6 + np.pi/3)
y2 = np.sin(1.8*np.pi*n)
y3 = np.cos(0.5*n)

fig,ax = plt.subplots(3,1)

ax[0].set_title("Q1 D Solution")
ax[0].stem(n,y1,'g')# this will give the points on the graph

ax[1].stem(n,y2,'g')

ax[2].stem(n,y3,'g')
for x in range(0,3,1):
    ax[x].grid() #this gives the grid background
fig.show()


