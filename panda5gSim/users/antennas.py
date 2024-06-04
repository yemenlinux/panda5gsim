import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def get_3d_power_radiation_pattern(
                                transformed_phi = None,
                                transformed_theta = None,
                                theta_3db = 65, 
                                phi_3db = 65, 
                                SLA_v = 30, 
                                A_max = 30):
    """ Generate 3D power radiation pattern for 3GPP antenna for 5G
    """
    A_double_abostroph_theta = - np.minimum(12 * 
            (transformed_theta / theta_3db)**2, SLA_v)
    A_double_abostroph_phi = -np.minimum(12 * 
            (transformed_phi / phi_3db)**2, A_max)
    A_double_abostroph = - np.minimum(
        -(A_double_abostroph_theta + A_double_abostroph_phi), A_max)
    return A_double_abostroph
    


def generate_3d_power_radiation_pattern(requested_phi = None,
                                        requested_theta = None,
                                        theta_3db = 15, 
                                        phi_3db = 65, 
                                        antenna_down_tilt = 0, 
                                        SLA_v = 10, 
                                        A_max = 30, 
                                        fron_back_ratio = -40):
    """ Generate 3D power radiation pattern for 3GPP antenna for 5G
    """
    theta = np.linspace(0, 180, 181)
    #theta = np.where(theta<181, theta, theta*0)
    phi = np.linspace(-180, 180, 361)
    theta_mesh, phi_mesh = np.meshgrid(phi, theta)
    # print(theta_mesh.shape, phi_mesh.shape)
    # calculate the the antenna pattern
    # source: 3GPP TR 38.901 version 14.0.0 (Release 14) 
    # Vertical cut of the radiation power pattern (dB)
    A_double_abostroph_theta = - np.minimum(12 * ((theta_mesh - 90 - antenna_down_tilt)/theta_3db)**2, SLA_v) 
    A_double_abostroph_phi = -np.minimum(12 * (phi_mesh/phi_3db)**2, A_max) 
    #
    A_double_abostroph = - np.minimum(-(A_double_abostroph_theta + A_double_abostroph_phi), A_max)
    if requested_theta is not None and requested_phi is not None:
        return A_double_abostroph[int(requested_theta)][int(requested_phi)]
    else:
        return A_double_abostroph
    
# rp = generate_3d_power_radiation_pattern()

# rp.shape

# theta = np.linspace(0, 180, 181)
# #theta = np.where(theta<181, theta, theta*0)
# phi = np.linspace(-180, 180, 361)
# array_size = rp.shape
# vt = np.ones(array_size[0])
# vp = np.ones(array_size[1])
# Th = np.radians(np.outer(theta, vp))
# Ph = np.radians(np.outer(vt, phi))
# print(Th.shape, Ph.shape, rp.shape)
# V = rp
# # convert polar to xyz
# X = abs(V) * np.cos(Ph) * np.sin(Th)
# Y = abs(V) * np.sin(Ph) * np.sin(Th)
# Z = abs(V) * np.cos(Th)
# print(f'x {X.shape}')
# # color
# colors =plt.cm.autumn( (X-X.min())/float((X-X.min()).max()) )
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0, antialiased=False)
# #surf = ax.plot_wireframe(X,Y,Z, linewidth=1, rstride=5, cstride=5 )

# # Customize the z axis.
# #ax.set_zlim(-1.01, 1.01)
# #ax.zaxis.set_major_locator(LinearLocator(10))

# # A StrMethodFormatter is used automatically
# #ax.zaxis.set_major_formatter('{x:.02f}')

# # Add a color bar which maps values to colors.
# #fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# #plt.gca().invert_xaxis()
# #plt.gca().invert_yaxis()

# plt.show() 