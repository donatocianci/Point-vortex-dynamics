


import numpy as np
from scipy.integrate import solve_ivp
pi = np.pi
# Simulation setup

#initial conditions: self-similar configuration of equilateral triangle
Gamma=np.array([1,1,1])
Tmax=140
t_vals = np.linspace(0,Tmax,300)
x=np.array([1, 3, 2])
y=np.array([3, 3, 3+2*np.cos(pi/6.0)])


#reformat position data
pos = np.array(list(zip(x,y)))
x0 = pos.flatten()
N=len(Gamma) # number of vortices
colors = list(map(lambda x: 'b' if x<=0 else 'r',Gamma))

def f(x,Gamma):
    '''
    This function takes in a position x, which is a numpy array with shape (2N,).
    The array x encodes the positions by concatenating the (x,y)-coords of the
    point vortices. So if there are two point vortices with coords (1,1) and (2,4),
    respectively, then the corresponding x varible will be np.array([1,1,2,4]).

    Gamma is a numpy array with shape (N,) where Gamma[i] is the strength of the
    i-th point vortex.

    This function returns a numpy array with the same shape as x that yields the
    velocity vector for the point vortices. The velocities are encoded the same
    way that x encodes the position.
    '''

    x = x.reshape(-1,2)
    vel = np.zeros(x.shape)
    for alpha in range(len(pos)):
        for beta in range(len(pos)):
            if alpha!=beta:
                dist_sq = (x[alpha,0]-x[beta,0])**2 + (x[alpha,1]-x[beta,1])**2
                vel[alpha,0] =vel[alpha,0]+Gamma[beta]*(-x[alpha,1]+x[beta,1])/(4*pi**2*dist_sq)
                vel[alpha,1] =vel[alpha,1]+Gamma[beta]*(x[alpha,0]-x[beta,0])/(4*pi**2*dist_sq)

    vel = vel.flatten()
    return vel

#solve the ODE
sol = solve_ivp(lambda t,x : f(x,Gamma), (0,Tmax), x0, t_eval = t_vals)

#Reorganize solution array. Indices are 0:vortex number, 1:coordinate, 2: t-val
pos = sol.y.reshape(N,2,len(sol.t))

delta = 0.025
x = np.arange(0,5.0, delta)
y = np.arange(0,5.0, delta)
X,Y = np.meshgrid(x,y)

#get flow lines
def streamlines(X,Y,posx,posy,Gamma):
    '''
    This function takes in numpy arrays X and Y of shape (W,H), where W and H
    are the width and height of the (x,y)-grid in the plane, respectively.

    posx and posy are numpy arrays of shape (N,) that encode the x and y coordinates
    (respectively) of the point vortices. So posx[i] yields the x-coordinate of the
    ith vortex.

    Gamma is a numpy array with shape (N,) where Gamma[i] is the strength of the
    i-th point vortex.

    The function returns psi, a numpy array of size (W,H) that gives the value of the
    streamfunction at the grid points. So psi[i,j] gives the value of the streamfunction
    at (X[i],Y[j]).
    '''

    psi = 0.0*X+0.0*Y
    for i in range(N):
        dist= np.sqrt( (X-posx[i])**2+(Y-posy[i])**2);
        psi= psi-Gamma[i]/(4*pi**2)*np.log(dist);

    return psi

#animate the dynamics
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
levels = np.linspace(-0.30,0.30)
(fig, ax) = plt.subplots()

ax.set_xlim(x[0], x[-1])
ax.set_ylim(y[0], y[-1])
xdata = pos[:,0,0]
ydata = pos[:,1,0]
Z = streamlines(X,Y,xdata,ydata,Gamma)

ax.contour(X,Y,Z,levels,colors = 'k',linestyles = 'solid',linewidths = 1.0, zorder = -1)
scat = ax.scatter(xdata, ydata, s = np.abs(50*Gamma), c = colors)

def update(frame):

    ax.clear()

    frame = int(frame)
    xdata = pos[:,0,frame]
    ydata = pos[:,1,frame]
    scat.set_offsets(np.c_[xdata, ydata])
    Z = streamlines(X,Y,xdata,ydata,Gamma)
    ax.contour(X,Y,Z,levels, colors = 'k', linestyles = 'solid', linewidths = 1.0, zorder = -1)
    ax.set_aspect('equal','box')

    return (scat,ax)

ani = FuncAnimation(fig, update, frames=range(len(sol.t)), blit=True, repeat=True)
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)

plt.show()
