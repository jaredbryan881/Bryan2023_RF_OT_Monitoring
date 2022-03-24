from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.stats import gaussian_kde

def main():
    N=100
    x = np.linspace(-3,4,N)
    y = np.linspace(-4,4,N)
    X,Y = np.meshgrid(x,y)
    Z_surf = curved_surf(X,Y)

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=18, azim=-150)

    # Plot curved surface
    surf = ax.plot_surface(X, Y, Z_surf, alpha=0.2, cmap=cm.cool)
    wire = ax.plot_wireframe(X, Y, Z_surf, rstride=20, cstride=20, color='k', lw=1)

    # Plot tangent space
    wire_plane = ax.plot_wireframe(X, Y, 0*Z_surf, rstride=20, cstride=20, color='k', lw=1)

    # Plot 3 major points on both surfaces and the lines between them
    p1=[-1,0]
    p2=[2.5,-3]
    p3=[3.5,2.5]
    p1.append(curved_surf(p1[0],p1[1]))
    p2.append(curved_surf(p2[0],p2[1]))
    p3.append(curved_surf(p3[0],p3[1]))

    # plot random points
    npts_rand=5000
    x_rand=1.5*np.random.randn(npts_rand)+p1[0]
    y_rand=np.random.randn(npts_rand)+p1[1]
    valid_pts = (x_rand>x[0]) & (x_rand<x[-1]) & (y_rand>y[0]) & (y_rand<y[-1])
    # on the surface
    ax.scatter(x_rand[valid_pts], y_rand[valid_pts], curved_surf(x_rand[valid_pts], y_rand[valid_pts]), c='k', s=1, alpha=0.2, zorder=0)
    # as a contour on the plane    
    positions=np.vstack([X.ravel(), Y.ravel()])
    values=np.vstack([x_rand,y_rand])
    kernel=gaussian_kde(values)
    f=np.reshape(kernel(positions).T, X.shape)
    ax.contourf(X,Y,f,cmap='Blues', levels=3, alpha=0.9, zorder=0)

    # points on the magic carpet
    ax.scatter(p1[0], p1[1], p1[2], c='k', s=50, zorder=2)
    ax.scatter(p2[0], p2[1], p2[2], c='k', s=50, zorder=2)
    ax.scatter(p3[0], p3[1], p3[2], c='k', s=50, zorder=2)
    # points on the plane
    ax.scatter(p1[0], p1[1], 0, c='k', s=50, zorder=4)
    ax.scatter(p2[0], p2[1], 0, c='k', s=50, zorder=4)
    ax.scatter(p3[0], p3[1], 0, c='k', s=50, zorder=4)

    # lines on the carpet and plane
    x12=np.linspace(p1[0],p2[0],50)
    y12=np.linspace(p1[1],p2[1],50)
    ax.plot(x12,y12,curved_surf(x12,y12),c='crimson', zorder=3)
    ax.plot(x12,y12,0,c='crimson', zorder=3)

    x13=np.linspace(p1[0],p3[0],50)
    y13=np.linspace(p1[1],p3[1],50)
    ax.plot(x13,y13,curved_surf(x13,y13),c='springgreen', zorder=3)
    ax.plot(x13,y13,0,c='springgreen', zorder=3)

    x23=np.linspace(p2[0],p3[0],50)
    y23=np.linspace(p2[1],p3[1],50)
    ax.plot(x23,y23,curved_surf(x23,y23),c='dodgerblue', zorder=3)
    ax.plot(x23,y23,0,c='dodgerblue', zorder=3)

    ax.set_zlim(0, np.max(Z_surf))
    ax.set_axis_off()
    plt.tight_layout()

    plt.show()

def curved_surf(X,Y):
    return X**3 + Y**2 + 100

if __name__=="__main__":
    main()