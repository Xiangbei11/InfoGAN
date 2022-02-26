import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy import interpolate

from lsdo_utils.comps.bspline_comp import  get_bspline_mtx

curve_y_coords = np.loadtxt('txt_files/training_airfoils_y_coordinates.txt').T.astype(np.float32)
curve_x_coords = np.loadtxt('txt_files/training_airfoils_x_coordinates.txt').astype(np.float32)

training_size = curve_y_coords.shape[0]
print('training_size',training_size)
x_range = np.linspace(0,1,300)
i_vec = np.arange(0,len(x_range)+1)
x_interp = 1 - np.cos(np.pi/(2 * len(x_range)) * i_vec )
camber_cp_num = 10
thickness_cp_num = 10
camber_B_spline_mat = get_bspline_mtx(camber_cp_num, len(x_interp), order = 4)
thickness_B_spline_mat = get_bspline_mtx(thickness_cp_num, len(x_interp), order = 4)
ctrl_points_camber = np.zeros((camber_cp_num,2,training_size))
ctrl_points_thickness = np.zeros((thickness_cp_num,2,training_size))

training_camber = np.zeros((len(x_interp),training_size))
training_thickness = np.zeros((len(x_interp),training_size))

for i in range(training_size):
    upper_surface = curve_y_coords[i,0:301]
    lower_surface = curve_y_coords[i,301:602]

    # Camber
    camber = (upper_surface + lower_surface) / 2
    training_camber[:,i] = camber

    # Thickness
    thickness = abs(upper_surface - lower_surface)
    training_thickness[:,i] = thickness

    camber_data = np.transpose(np.vstack((x_interp, camber)))
    ATA_camber = np.dot(np.transpose(camber_B_spline_mat),camber_B_spline_mat)
    ATbx_camber = np.dot(np.transpose(camber_B_spline_mat.todense()),camber_data)
    L_camber = np.linalg.cholesky(ATA_camber.todense())
    w_camber = np.linalg.solve(L_camber,ATbx_camber)
    x_camber = np.linalg.solve(np.transpose(L_camber),w_camber)
    x_camber[0,0] = 0
    x_camber[0,1] = 0
    x_camber[-1,0] = 1
    x_camber[-1,1] = 0
    ctrl_points_camber[:,:,i] = x_camber   

    thickness_data = np.transpose(np.vstack((x_interp, thickness)))
    ATA_thickness = np.dot(np.transpose(thickness_B_spline_mat),thickness_B_spline_mat)
    ATbx_thickness = np.dot(np.transpose(thickness_B_spline_mat.todense()),thickness_data)
    L_thickness = np.linalg.cholesky(ATA_thickness.todense())
    w_thickness = np.linalg.solve(L_thickness,ATbx_thickness)
    x_thickness = np.linalg.solve(np.transpose(L_thickness),w_thickness)
    x_thickness[0,0] = 0
    x_thickness[0,1] = 0
    x_thickness[-1,0] = 1
    x_thickness[-1,1] = 0
    ctrl_points_thickness[:,:,i] = x_thickness

total_control_points = np.concatenate((ctrl_points_camber[:,1,:],ctrl_points_thickness[:,1,:]))
total_camber_thickness = np.concatenate((training_camber,training_thickness))
print('total_control_points',total_control_points.shape)
print('total_camber_thickness',total_camber_thickness.shape)

np.savetxt('txt_files/20_camber_thickness_b_spline_cp.txt', total_control_points)
np.savetxt('txt_files/training_airfoils_camber_thickness.txt', total_camber_thickness )

curve_x_coords = np.loadtxt('txt_files/training_airfoils_x_coordinates.txt').astype(np.float32)
print('curve_x_coords',curve_x_coords.shape)
# curve_camber_thickness = np.loadtxt('txt_files/training_airfoils_camber_thickness.txt').T.astype(np.float32)
# print('curve_camber_thickness',curve_camber_thickness.shape) #(1527, 602)
# curve_camber_thickness = curve_camber_thickness[0:-2,:]
# print('curve_camber_thickness',curve_camber_thickness.shape) #(1525, 602)

control_camber_thickness = np.loadtxt('txt_files/20_camber_thickness_b_spline_cp.txt').T.astype(np.float32)
print('control_camber_thickness ',control_camber_thickness.shape) #(1527, 20)
control_camber_thickness = control_camber_thickness[0:-2,:]
print('control_camber_thickness ',control_camber_thickness.shape) #(1525, 20)

bspline_matrix = get_bspline_mtx(10, 301, order = 4)
bspline_matrix = bspline_matrix.todense()
bspline_matrix = bspline_matrix.astype(np.float32)
curve_y_coords_camber = np.matmul(bspline_matrix,control_camber_thickness[:,0:10].T).T
curve_y_coords_thickness = np.matmul(bspline_matrix,control_camber_thickness[:,10:20].T).T


curve_camber_thickness = np.concatenate((curve_y_coords_camber, curve_y_coords_thickness),-1) 


range0 = range(0, 301, 10)
range1 = range(301, 602, 10)
test_size0 = 9
test_size1 = 6
fig, axs = plt.subplots(3, 3, figsize=(15,7.2))#
axs = axs.flatten()
for i in range(test_size0):
    color = plt.cm.rainbow(np.linspace(0, 1, test_size1))
    for j, c in zip(range(test_size1), color):
        axs[i].plot(curve_x_coords[0:301,0].reshape((301,1)), (curve_camber_thickness[test_size1*i+j,0:301]+0.5*curve_camber_thickness[test_size1*i+j,301:602]).reshape((301,1)), c = c)
        axs[i].plot(curve_x_coords[0:301,0].reshape((301,1)), (curve_camber_thickness[test_size1*i+j,0:301]-0.5*curve_camber_thickness[test_size1*i+j,301:602]).reshape((301,1)), c = c)
        axs[i].plot(curve_x_coords[range0,0], curve_y_coords[test_size1*i+j,range0], c = c, marker = '*', markersize = 3, linestyle = 'None',)
        axs[i].plot(curve_x_coords[range0,0], curve_y_coords[test_size1*i+j,range1], c = c, marker = '*', markersize = 3, linestyle = 'None',)    

fig.tight_layout()
plt.show()
#change string and generated txt