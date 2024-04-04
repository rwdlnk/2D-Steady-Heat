'''
Solve steady state two-dimensional Heat Diffusion equation in
a rectangular domain with constant thermal conductivity
with Dirichlet, Neumann, and/or Robin BC's.
'''
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

I = lambda k, Nx: k % Nx
J = lambda k, Nx: int(k/Nx)
IJ = lambda k, Nx: [I(k,Nx),J(k,Nx)]
K = lambda i, j, Nx: i + j*Nx

print("2D Steady Heat Equation Solver")

Nx = 41
Ny = 61
Nxy = Nx*Ny

A = np.zeros((Nxy,Nxy))
b = np.zeros(Nxy)

x_plate_length = 2. # m
y_plate_length = 3. # m

AR = str(y_plate_length/x_plate_length)

dx = x_plate_length/(Nx-1)
dy = y_plate_length/(Ny-1)

# Thermal conductivity, W/m-K
kappa = 5.0

# Set boundary conditions for the 4 sides of the rectangular domain.
# For the 4 sides of the domain, set BCtype: either 'D' = Dirichlet, 'F' = flux,
# or 'C' = convective using the following dictionary whose format is: 
#   side number : (type letter, value/s)
# where: 
#    side 0 : bottom, 
#    side 1 : right, 
#    side 2 : top,
#    side 3 : left
# NOTE: The heat flux vector is given by: q_i = -kappa dT/dx_i. For heat flux out of the domain
# qm > 0. 
# 
#   For side m: 
#   If a Dirichlet condition of fixed Tm (deg. C) enter:
#       m : ('D', Tm)
#   If a fixed flux, or Neumann condition, of magnitude qm  (W/m**2)enter:
#       m : ('F', qm)
#   If a convective condition (i.e., a Robin condition) of the form qm = hm (Tm - Tinf)
#       with convection coefficient, hm (W/m*2-C), and Tinf (deg C) enter:
#       m : ('C', hm, Tinf)
#
BCoptions = ['C', 'D', 'F']
BCtype = { 0 : ('D', 40.) , 1 : ('C', 1., 120.), 2 : ('F', -150.), 3 : ('D',100.) }
next_side = [1, 2, 3, 0]

def calculate():

    for i in range(1,Nx-1,1):
        for j in range(1,Ny-1,1):

            ij = K(i,j,Nx)  # K index for node i,j
            ip1j = K(i+1,j,Nx)
            im1j = K(i-1,j,Nx)
            ijp1 = K(i,j+1,Nx)
            ijm1 = K(i,j-1,Nx)

            A[ij,ij] = -2.*(1./dx**2 + 1./dy**2)
            A[ij,ip1j] = 1./dx**2
            A[ij,im1j] = 1./dx**2
            A[ij,ijp1] = 1./dy**2
            A[ij,ijm1] = 1./dy**2

    # Apply BC's

    for side in BCtype:
        if side ==0:    # Bottom side
            if BCtype[side][0] == "D":
                for i in range(0,Nx,1):
                    ij = K(i,0,Nx)
                    A[ij,ij] = 1.
                    b[ij] = BCtype[side][1]
            elif BCtype[side][0] == 'F':
                for i in range(1,Nx-1,1):
                    j = 0

                    ij = K(i,j,Nx)  # K index for node i,j
                    ip1j = K(i+1,j,Nx)
                    im1j = K(i-1,j,Nx)
                    ijp1 = K(i,j+1,Nx)

                    A[ij,ij] = -2.*(1./dx**2 + 1./dy**2)
                    A[ij,ip1j] = 1./dx**2
                    A[ij,im1j] = 1./dx**2
                    A[ij,ijp1] = 2./dy**2
                    b[ij] = 2.*BCtype[side][1]/(kappa*dy)
            elif BCtype[side][0] == 'C':
                h = BCtype[side][1]
                Tinf = BCtype[side][2]
                Nu = h*dy/kappa

                for i in range(1,Nx-1,1):
                    j = 0

                    ij = K(i,j,Nx)  # K index for node i,j
                    ip1j = K(i+1,j,Nx)
                    im1j = K(i-1,j,Nx)
                    ijp1 = K(i,j+1,Nx)

                    A[ij,ij] = -2.*(1./dx**2 + (1.+Nu)/dy**2)
                    A[ij,ip1j] = 1./dx**2
                    A[ij,im1j] = 1./dx**2
                    A[ij,ijp1] = 2./dy**2
                    b[ij] = -2.*Nu*Tinf/dy**2

            else:
                assert BCtype[side][0] in BCoptions, "Error: Incorrect BCtype, either 'D', 'C', or 'F'"
                exit()
        elif side == 1:  # Right side
            if BCtype[side][0] == "D":
                for j in range(0,Ny,1):
                    ij = K(Nx-1,j,Nx)
                    A[ij,ij] = 1.
                    b[ij] = BCtype[side][1]
            elif BCtype[side][0] == 'F':
                for j in range(1,Ny-1,1):
                    i = Nx-1

                    ij = K(i,j,Nx)  # K index for node i,j
                    im1j = K(i-1,j,Nx)
                    ijp1 = K(i,j+1,Nx)
                    ijm1 = K(i,j-1,Nx)

                    A[ij,ij] = -2.*(1./dx**2 + 1./dy**2)
                    A[ij,im1j] = 2./dx**2
                    A[ij,ijm1] = 1./dy**2
                    A[ij,ijp1] = 1./dy**2
                    b[ij] = 2.*BCtype[side][1]/(kappa*dx)
            elif BCtype[side][0] == 'C':
                h = BCtype[side][1]
                Tinf = BCtype[side][2]
                Nu = h*dx/kappa     # Nusselt number

                for j in range(1,Ny-1,1):
                    i = Nx-1
                    ij = K(i,j,Nx)  # K index for node i,j
                    im1j = K(i-1,j,Nx)
                    ijp1 = K(i,j+1,Nx)
                    ijm1 = K(i,j-1,Nx)

                    A[ij,ij] = -2.*((1.+Nu)/dx**2 + 1./dy**2)
                    A[ij,ijm1] = 1./dy**2
                    A[ij,im1j] = 2./dx**2
                    A[ij,ijp1] = 1./dy**2
                    b[ij] = -2.*Nu*Tinf/dx**2
            else:
                assert BCtype[side][0] in BCoptions, "Error: Incorrect BCtype, either 'D', 'C', or 'F'"
                exit()
        elif side == 2: # Top side
            if BCtype[side][0] == "D":
                for i in range(0,Nx,1):
                    ij = K(i,Ny-1,Nx)
                    A[ij,ij] = 1.
                    b[ij] = BCtype[side][1]
            elif BCtype[side][0] == 'F':
                for i in range(1,Nx-1,1):
                    j = Ny-1

                    ij = K(i,j,Nx)  # K index for node i,j
                    ip1j = K(i+1,j,Nx)
                    im1j = K(i-1,j,Nx)
                    ijm1 = K(i,j-1,Nx)

                    A[ij,ij] = -2.*(1./dx**2 + 1./dy**2)
                    A[ij,ip1j] = 1./dx**2
                    A[ij,im1j] = 1./dx**2
                    A[ij,ijm1] = 2./dy**2
                    b[ij] = 2.*BCtype[side][1]/(kappa*dy)
            elif BCtype[side][0] == 'C':
                h = BCtype[side][1]
                Tinf = BCtype[side][2]
                Nu = h*dy/kappa     # Nusselt number

                for i in range(1,Nx-1,1):
                    j = Ny-1

                    ij = K(i,j,Nx)  # K index for node i,j
                    ip1j = K(i+1,j,Nx)
                    im1j = K(i-1,j,Nx)
                    ijm1 = K(i,j-1,Nx)

                    A[ij,ij] = -2.*(1./dx**2 + (1.+Nu)/dy**2)
                    A[ij,ip1j] = 1./dx**2
                    A[ij,im1j] = 1./dx**2
                    A[ij,ijm1] = 2./dy**2
                    b[ij] = -2.*Nu*Tinf/dy**2
            else:
                assert BCtype[side][0] in BCoptions, "Error: Incorrect BCtype, either 'D', 'C', or 'F'"
                exit()
        else:  # Left side
            if BCtype[side][0] == "D":
                for j in range(0,Ny,1):
                    ij = K(0,j,Nx)
                    A[ij,ij] = 1.
                    b[ij] = BCtype[side][1]
            elif BCtype[side][0] == 'F':
                for j in range(1,Ny-1,1):
                    i = 0

                    ij = K(i,j,Nx)  # K index for node i,j
                    ip1j = K(i+1,j,Nx)
                    ijp1 = K(i,j+1,Nx)
                    ijm1 = K(i,j-1,Nx)

                    A[ij,ij] = -2.*(1./dx**2 + 1./dy**2)
                    A[ij,ip1j] = 2./dx**2
                    A[ij,ijm1] = 1./dx**2
                    A[ij,ijp1] = 1./dy**2
                    b[ij] = 2.*BC[side][1]/(kappa*dx)
            elif BCtype[side][0] == 'C':
                h = BCtype[side][1]
                Tinf = BCtype[side][2]
                Nu = h*dx/kappa     # Nusselt number
                for j in range(1,Ny-1,1):
                    i = 0

                    ij = K(i,j,Nx)  # K index for node i,j
                    ip1j = K(i+1,j,Nx)
                    ijp1 = K(i,j+1,Nx)
                    ijm1 = K(i,j-1,Nx)

                    A[ij,ij] = -2.*((1.+Nu)/dx**2 + 1./dy**2)
                    A[ij,ip1j] = 2./dx**2
                    A[ij,ijm1] = 1./dx**2
                    A[ij,ijp1] = 1./dy**2
                    b[ij] = -2.*Nu*Tinf/dx**2
            else:
                assert BCtype[side][0] in BCoptions, "Error: Incorrect BCtype, either 'D', 'C', or 'F'"
                exit()

    # Now apply BC's to the 4 corners
    for side in range(0,4,1):
        s_next = next_side[side]
        if side == 0:   # corner (Nx-1,0)
            i = Nx-1
            j = 0
            ij = K(i,j,Nx)
            ijp1 = K(i,j+1,Nx)
            im1j = K(i-1,j,Nx)
            c1 = dx/(dx+dy)
            c2 = dy/(dx+dy)
            if BCtype[side][0] == 'D' and BCtype[s_next][0] == 'D':
                A[ij,ij] = 1.
                b[ij] = (BCtype[side][1] + BCtype[s_next][1])/2.
            elif BCtype[side][0] == 'F' and BCtype[s_next][0] == 'F':
                A[ij,ij] = -(1./dx**2 + 1./dy**2)
                A[ij,im1j] = 1./dx**2
                A[ij,ijp1] = 1./dy**2
                b[ij] = BCtype[side][1]/(kappa*dy) + BCtype[s_next][1]/(kappa*dx)
            elif BCtype[side][0] == 'F' and BCtype[s_next][0] == 'C':
                h = BCtype[s_next][1]
                Tinf = BCtype[s_next][2]
                Nu = h*dx/kappa     # Nusselt number
                A[ij,ij] = -((1.+Nu)/dx**2 + 1./dy**2)
                A[ij,im1j] = 1./dx**2
                A[ij,ijp1] = 1./dy**2
                b[ij] = BCtype[side][1]/(kappa*dy) - Nu*Tinf/dx**2
            elif BCtype[side][0] == 'C' and BCtype[s_next][0] == 'F':
                h = BCtype[side][1]
                Tinf = BCtype[side][2]
                Nu = h*dy/kappa     # Nusselt number
                A[ij,ij] = -(1./dx**2 + (1.+Nu)/dy**2)
                A[ij,im1j] = 1./dx**2
                A[ij,ijp1] = 1./dy**2
                b[ij] = -Nu*Tinf/dy**2 + BCtype[s_next][1]/(kappa*dx)
            elif BCtype[side][0] == 'C' and BCtype[s_next][0] == 'C':
                hs = BCtype[side][1]
                Tinfs = BCtype[side][2]
                Nus = hs*dy/kappa     # Nusselt number
                hn = BCtype[s_next][1]
                Tinfn = BCtype[s_next][2]
                Nun = hn*dx/kappa     # Nusselt number
                A[ij,ij] = -((1.+Nun)/dx**2 + (1.+Nus)/dy**2)
                A[ij,im1j] = 1./dx**2
                A[ij,ijp1] = 1./dy**2
                b[ij] = -Nus*Tinfs/dy**2 - Nun*Tinfn/dx**2
            else:
                assert BCtype[side][0] in BCoptions, "Error: Incorrect BCtype, either 'D', 'C', or 'F'"
        elif side == 1:   # Corner (Nx-1, Ny-1)
            i = Nx-1
            j = Ny-1
            ij = K(i,j,Nx)
            ijm1 = K(i,j-1,Nx)
            im1j = K(i-1,j,Nx)
            #elif BCtype[side][0] == 'F' and BCtype[s_next][0] == 'F':   # both side and s_next are flux BC's
            if BCtype[side][0] == 'D' and BCtype[s_next][0] == 'D':
                A[ij,ij] = 1.
                b[ij] = (BCtype[side][1] + BCtype[s_next][1])/2.
            elif BCtype[side][0] == 'F' and BCtype[s_next][0] == 'F':   # both side and s_next are flux BC's
                A[ij,ij] = -(1./dx**2 + 1./dy**2)
                A[ij,im1j] = 1./dx**2
                A[ij,ijm1] = 1./dy**2
                b[ij] = BCtype[side][1]/(kappa*dx) + BCtype[s_next][0]/(kappa*dy)
            elif BCtype[side][0] == 'F' and BCtype[s_next][0] == 'C':
                h = BCtype[s_next][1]
                Tinf = BCtype[s_next][2]
                Nu = h*dy/kappa     # Nusselt number
                A[ij,ij] = -(1./dx**2 + (1.+Nu)/dy**2)
                A[ij,im1j] = 1./dx**2
                A[ij,ijm1] = 1./dy**2
                b[ij] = BCtype[side][1]/(kappa*dx) - Nu*Tinf/dy**2
            elif BCtype[side][0] == 'C' and BCtype[s_next][0] == 'F':
                h = BCtype[side][1]
                Tinf = BCtype[side][2]
                Nu = h*dy/kappa     # Nusselt number
                A[ij,ij] = -2.*(1./dx**2 + 1./dy**2 + Nu/(dx*dy))
                A[ij,im1j] = 2./dx**2
                A[ij,ijm1] = 2./dy**2
                b[ij] = -Nu*Tinf/(dx*dy) + BCtype[s_next][1]/(kappa*dy)
            elif BCtype[side][0] == 'C' and BCtype[s_next][0] == 'C':
                hs = BCtype[side][1]
                Tinfs = BCtype[side][2]
                Nus = hs*dx/kappa     # Nusselt number
                hn = BCtype[s_next][1]
                Tinfn = BCtype[s_next][2]
                Nun = hn*dy/kappa     # Nusselt number
                A[ij,ij] = -((1.+Nus)/dx**2 + (1.+Nun)/dy**2)
                A[ij,im1j] = 1./dx**2
                A[ij,ijm1] = 1./dy**2
                b[ij] = -Nus*Tinfs/dx**2 - Nun*Tinfn/dy**2
            else:
                assert BCtype[side][0] in BCoptions, "Error: Incorrect BCtype, either 'D', 'C', or 'F'"
        elif side == 2:   # corner (0, Ny-1)
            i = 0
            j = Ny-1
            ij = K(i,j,Nx)
            ip1j = K(i+1,j,Nx)
            ijm1 = K(i,j-1,Nx)
            if BCtype[side][0] == 'D' and BCtype[s_next][0] == 'D':
                A[ij,ij] = 1.
                b[ij] = (BCtype[side][1] + BCtype[s_next][1])/2.
            elif BCtype[side][0] == "F" and BCtype[s_next][0] == "F":
                A[ij,ij] = -(1./dx**2 + 1./dy**2)
                A[ij,ip1j] = 1./dx**2
                A[ij,ijm1] = 1./dy**2
                b[ij] = BCtype[side][1]/(kappa*dy) + BCtype[s_next][1]/(kappa*dx)
            elif BCtype[side][0] == "F" and BCtype[s_next][0] == "C":
                h = BCtype[s_next][1]
                Tinf = BCtype[s_next][2]
                Nu = h*dx/kappa     # Nusselt number

                A[ij,ij] = -((1.+Nu)/dx**2 + 1./dy**2)
                A[ij,ip1j] = 1./dx**2
                A[ij,ijm1] = 1./dy**2
                b[ij] = BCtype[side][1]/(kappa*dy) - Nu*Tinf/dx**2
            elif BCtype[side][0] == 'C' and BCtype[s_next][0] == 'F':
                h = BCtype[side][1]
                Tinf = BCtype[side][2]
                Nu = h*dy/kappa     # Nusselt number
                A[ij,ij] = -(1./dx**2 + (1.+Nu)/dy**2)
                A[ij,ip1j] = 1./dx**2
                A[ij,ijm1] = 1./dy**2
                b[ij] = -Nu*Tinf/dy**2 + BCtype[s_next][1]/(kappa*dx)
            elif BCtype[side][0] == 'C' and BCtype[s_next][0] == 'C':
                hs = BCtype[side][1]
                Tinfs = BCtype[side][2]
                Nus = hs*dy/kappa     # Nusselt number
                hn = BCtype[s_next][1]
                Tinfn = BCtype[s_next][2]
                Nun = hn*dx/kappa     # Nusselt number
                A[ij,ij] = -((1.+Nun)/dx**2 + (1.+Nus)/dy**2)
                A[ij,ip1j] = 1./dx**2
                A[ij,ijm1] = 1./dy**2
                b[ij] = -Nus*Tinfs/dy**2 - Nun*Tinfn/dx**2
            else:
                assert BCtype[side][0] in BCoptions, "Error: Incorrect BCtype, either 'D', 'C', or 'F'"
        elif side == 3:   # corner (0, 0)
            i = 0
            j = 0
            ij = K(i,j,Nx)
            ijp1 = K(i,j+1,Nx)
            ip1j = K(i+1,j,Nx)
            #elif BCtype[side][0] == "F" and BCtype[s_next][0] == "F":
            if BCtype[side][0] == 'D' and BCtype[s_next][0] == 'D':
                A[ij,ij] = 1.
                b[ij] = (BCtype[side][1] + BCtype[s_next][1])/2.
            elif BCtype[side][0] == "F" and BCtype[s_next][0] == "F":
                A[ij,ij] = -(1./dx**2 + 1./dy**2)
                A[ij,ip1j] = 1./dx**2
                A[ij,ijp1] = 1./dy**2
                b[ij] = BCtype[side][1]/(kappa*dx) + BCtype[s_next][1]/(kappa*dy)
            elif BCtype[side][0] == 'F' and BCtype[s_next][0] == 'C':
                h = BCtype[s_next][1]
                Tinf = BCtype[s_next][2]
                Nu = h*dy/kappa     # Nusselt number
                A[ij,ij] = -(1./dx**2 + (1.+Nu)/dy**2)
                A[ij,ip1j] = 1./dx**2
                A[ij,ijp1] = 1./dy**2
                b[ij] = BCtype[side][1]/(kappa*dx) - Nu*Tinf/dy**2
            elif BCtype[side][0] == 'C' and BCtype[s_next][0] == 'F':
                h = BCtype[side][1]
                Tinf = BCtype[side][2]
                Nu = h*dx/kappa     # Nusselt number
                A[ij,ij] = -((1.+Nu)/dx**2 + 1./dy**2)
                A[ij,ip1j] = 1./dx**2
                A[ij,ijp1] = 1./dy**2
                b[ij] = -Nu*Tinf/dx**2 + BCtype[s_next][1]/(kappa*dy)
            elif BCtype[side][0] == 'C' and BCtype[s_next][0] == 'C':
                hs = BCtype[side][1]
                Tinfs = BCtype[side][2]
                Nus = hs*dx/kappa     # Nusselt number
                hn = BCtype[s_next][1]
                Tinfn = BCtype[s_next][2]
                Nun = hn*dy/kappa     # Nusselt number
                A[ij,ij] = -((1.+Nus)/dx**2 + (1.+Nun)/dy**2)
                A[ip1j,ip1j] = 1./dx**2
                A[ij,ijp1] = 1./dy**2
                b[ij] = -Nus*Tinfs/dx**2 - Nun*Tinfn/dy**2
            else:
                assert BCtype[side][0] in BCoptions, "Error: Incorrect BCtype, either 'D', 'C', or 'F'"

    #print("A: ", A, "\n")

    x = linalg.spsolve(sparse.csr_matrix(A), b)

    #print('x = ', x)
    print("u(Nx-1,0): %6.2f \n" % x[K(Nx-1,0,Nx)])
    print("u(Nx-1,Ny-1): %6.2f \n" % x[K(Nx-1,Ny-1,Nx)])
    print("u(0,Ny-1): %6.2f \n" % x[K(0,Ny-1,Nx)])
    print("u(0,0): %6.2f \n" % x[K(0,0,Nx)])

    sol = np.zeros((Ny,Nx))
    for kk in range(0,Nxy,1):
        i = I(kk,Nx)
        j = J(kk,Nx)
        sol[j,i] = x[kk]

    return sol

# Do the calculation
u = calculate()
#print("u(x,y): \n", u)

# Plot the solution
im = plt.imshow(u[:,:],cmap=plt.cm.RdBu_r,extent=(0,Nx-1, 0., Ny-1),origin='lower',aspect=AR)

# add contour lines with labels
Tmax = round(u.max())
Tmin = round(u.min())
cset = plt.contour(u[:,:],np.arange(Tmin,Tmax,10.),linewidths=2,cmap=plt.cm.Set2)
plt.clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
plt.colorbar(im) # adding the colobar on the right
# latex fashion title
plt.title(f'$u(x,y)$')
name = "u-"+str(BCtype[0][0])+"-"+str(BCtype[1][0])+"-"+str(BCtype[2][0])+"-"+str(BCtype[3][0])+".pdf"
plt.savefig(name, format="pdf", bbox_inches="tight")
plt.show()

print("Done!")
