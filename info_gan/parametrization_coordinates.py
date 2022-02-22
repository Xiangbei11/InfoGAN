import openmdao.api as om 
import omtools.api as ot
import os
import numpy as np 
from lsdo_utils.comps.bspline_comp import  get_bspline_mtx, BsplineComp
import matplotlib.pyplot as plt
from geomdl import BSpline
from scipy import interpolate
parent_directory = os.getcwd()
path  = 'coord_seligFmt-2'
x_range = np.linspace(0,1,300)
j_vec = np.arange(0,len(x_range)+1)
# x_range_stretched = 1 / (1 + np.exp(-12 * (x_range-0.5)))
# x_range_stretched[0] = 0
# x_range_stretched[-1] = 1
x_range_stretched = 1 - np.cos(np.pi/(2 * len(x_range)) * j_vec )
x_interp = x_range_stretched
upper_cp_num = 8
lower_cp_num = 8
ctrl_points_upper = np.zeros((upper_cp_num,2,1532))
ctrl_points_lower = np.zeros((lower_cp_num,2,1532))
upper_B_spline_mat = get_bspline_mtx(upper_cp_num, len(x_interp), order = 4)
lower_B_spline_mat = get_bspline_mtx(lower_cp_num, len(x_interp), order = 4)
file_list = []
counter = 0
for file in os.listdir():
    # Check whether file is in .dat format or not
    if file.endswith(".dat"):
        file_path = f"{path}/{file}"
        # printing airfoil file name to track anomalies/ bad airfoils
        print('#########################')
        print(file)
        file_list.append(file)
        airfoil = open(file_path)
        lines = airfoil.readlines()[1:]
        # loading lines and converting them to vectors containing x,y coordinates 
        airfoil_coordinates = np.loadtxt(lines)
        x_coord = airfoil_coordinates[:,0]
        y_coord = airfoil_coordinates[:,1]
        # finding min index of x coordinate data
        min_index = np.min(np.where(x_coord == np.min(x_coord)))
        # select x,y coordinates for upper and lower surface from data 
        x_coord_upper = x_coord[0:min_index+1]
        y_coord_upper = y_coord[0:min_index+1]
        # print(x_coord_upper[-1])
        # print(y_coord_upper[-1])
        x_coord_lower = x_coord[min_index:]
        y_coord_lower = y_coord[min_index:]
        # print(y_coord_lower[0])
        # print(x_coord_lower[0])
        if abs(y_coord_upper[-1]) > 0.01 or abs(y_coord_lower[0]) > 0.01:
            print('WARNING: LEADING EDGE NOT LOCATED AT (0,0)')
         # x upper and lower surfaces
        y_interp_upper = np.interp(x_interp, np.flip(x_coord_upper), np.flip(y_coord_upper))
        y_interp_lower = np.interp(x_interp, x_coord_lower, y_coord_lower)
        upper_data = np.transpose(np.vstack((x_interp, y_interp_upper)))
        lower_data = np.transpose(np.vstack((x_interp, y_interp_lower)))
        ATA_upper = np.dot(np.transpose(upper_B_spline_mat),upper_B_spline_mat)
        ATbx_upper = np.dot(np.transpose(upper_B_spline_mat.todense()),upper_data)
        L_upper = np.linalg.cholesky(ATA_upper.todense())
        w_upper = np.linalg.solve(L_upper,ATbx_upper)
        x_upper = np.linalg.solve(np.transpose(L_upper),w_upper)
        x_upper[0,0] = 0
        x_upper[0,1] = 0
        x_upper[-1,0] = 1
        x_upper[-1,1] = 0
        ATA_lower = np.dot(np.transpose(lower_B_spline_mat),lower_B_spline_mat)
        ATbx_lower = np.dot(np.transpose(lower_B_spline_mat.todense()),lower_data)
        L_lower = np.linalg.cholesky(ATA_lower.todense())
        w_lower = np.linalg.solve(L_lower,ATbx_lower)
        x_lower = np.linalg.solve(np.transpose(L_lower),w_lower)
        x_lower[0,0] = 0
        x_lower[0,1] = 0
        x_lower[-1,0] = 1
        x_lower[-1,1] = 0
    ctrl_points_upper[:,:,counter] = x_upper
    ctrl_points_lower[:,:,counter] = x_lower
    counter = counter + 1
    print(counter)
    # print('#########################')
y_ctrl_points_upper = ctrl_points_upper[:,1,:]
y_ctrl_points_lower = ctrl_points_lower[:,1,:]
x_ctrl_points_upper = ctrl_points_upper[:,0,:]
x_ctrl_points_lower = ctrl_points_lower[:,0,:]
total_y_control_points = np.concatenate((y_ctrl_points_upper,y_ctrl_points_lower))
total_x_control_points = np.concatenate((x_ctrl_points_upper,x_ctrl_points_lower))
# np.savetxt('X_CONTROL_POINTS_NEW.txt', total_x_control_points)
# np.savetxt('Y_CONTROL_POINTS_NEW.txt', total_y_control_points)
# os.chdir(parent_directory)
# index_vec = np.loadtxt('LARGE_LE_RADIUS_INDEX_VEC.txt')
# for i in range(len(index_vec)):
#     print(file_list[int(index_vec[i])])
# exit()
os.chdir(parent_directory)
prob = om.Problem()
group = ot.Group()
group.create_indep_var('upper_cp', shape = (upper_cp_num))
group.create_indep_var('lower_cp', shape = (lower_cp_num))
prob.model.add_subsystem('external_inputs_group', group, promotes = ['*'])
comp = BsplineComp(
    num_pt=int(len(x_interp)),
    num_cp=upper_cp_num,
    in_name='upper_cp',
    jac=upper_B_spline_mat,
    out_name='upper_B_spline_curve',
)
prob.model.add_subsystem('upper_airfoil_curve_bspline_comp', comp, promotes = ['*'])
comp = BsplineComp(
    num_pt=int(len(x_interp)),
    num_cp=lower_cp_num,
    in_name='lower_cp',
    jac=lower_B_spline_mat,
    out_name='lower_B_spline_curve',
)
prob.model.add_subsystem('lower_airfoil_curve_bspline_comp', comp, promotes = ['*'])
prob.setup(check=True)
fig, axs = plt.subplots(2,1, figsize= (14,8))
B_spline_airfoil_y_coordinates = np.empty((602,1532))
B_spline_airfoil_x_coordinates = np.empty((602,1532))
camber_line = np.empty((301,1532))
LE_radius_vec = np.empty((1532,))
LE_upper_angle_vec = np.empty((1532,))
LE_lower_angle_vec = np.empty((1532,))
TE_upper_angle_vec = np.empty((1532,))
TE_lower_angle_vec = np.empty((1532,))
max_camber_vec = np.empty((1532,))
max_camber_x_loc = np.empty((1532,))
max_thickness_vec = np.empty((1532,))
max_thickness_x_loc = np.empty((1532,))
for i in range(1532):
    prob['upper_cp'] = ctrl_points_upper[:,1,i] #np.linspace(0,1,upper_cp_num)
    prob['lower_cp'] = ctrl_points_lower[:,1,i]# np.linspace(0,1,upper_cp_num)
    prob.run_model()
    upper_curve = prob['upper_B_spline_curve'].flatten()
    lower_curve = prob['lower_B_spline_curve'].flatten()
    total_curve = np.concatenate((upper_curve,lower_curve))
    total_x_coord = np.concatenate((x_interp,x_interp))
    B_spline_airfoil_y_coordinates[:,i] = total_curve
    B_spline_airfoil_x_coordinates[:,i] = total_x_coord
    # Leading edge radius
    LE_radius_center = 0.005
    LE_x_range = np.linspace(0,LE_radius_center,20)
    tck_upper = interpolate.splrep(x_interp,upper_curve)
    tck_lower = interpolate.splrep(x_interp,lower_curve)
    dydx_upper = interpolate.splev(LE_x_range,tck_upper,der=1)
    d2ydx2_upper = interpolate.splev(LE_x_range,tck_upper,der=2)
    dydx_lower = interpolate.splev(LE_x_range,tck_lower,der=1)
    d2ydx2_lower = interpolate.splev(LE_x_range,tck_lower,der=2)
    R_upper = (1 + (dydx_upper)**2)**(3/2) / (abs(d2ydx2_upper))
    R_lower = (1 + (dydx_lower)**2)**(3/2) / (abs(d2ydx2_lower))
    R_average = (np.average(R_upper) + np.average(R_lower)) / 2
    LE_radius_vec[i] = R_average
    # Leading edge angle 
    x_der = 0.03
    y_der_upper = interpolate.splev(x_der,tck_upper)
    dydx_upper = interpolate.splev(x_der,tck_upper,der=1)
    tngnt_upper = lambda x: dydx_upper * x + (y_der_upper - dydx_upper * x_der)
    y_der_lower = interpolate.splev(x_der,tck_lower)
    dydx_lower = interpolate.splev(x_der,tck_lower,der=1)
    tngnt_lower = lambda y: dydx_lower * y + (y_der_lower - dydx_lower * x_der)
    LE_upper_angle = np.arctan(dydx_upper) * 180 / np.pi
    LE_lower_angle = np.arctan(dydx_lower) * 180 / np.pi
    LE_upper_angle_vec[i] = LE_upper_angle
    LE_lower_angle_vec[i] = LE_lower_angle
    # Trailing edge angle 
    x_der_TE = 0.97
    y_der_upper_TE = interpolate.splev(x_der_TE,tck_upper)
    dydx_upper_TE = interpolate.splev(x_der_TE,tck_upper,der=1)
    tngnt_upper_TE = lambda x: dydx_upper_TE * x + (y_der_upper_TE - dydx_upper_TE * x_der_TE)
    y_der_lower_TE = interpolate.splev(x_der_TE,tck_lower)
    dydx_lower_TE = interpolate.splev(x_der_TE,tck_lower,der=1)
    tngnt_lower_TE = lambda y: dydx_lower_TE * y + (y_der_lower_TE - dydx_lower_TE * x_der_TE)
    LE_upper_angle_TE = np.arctan(dydx_upper_TE) * 180 / np.pi
    LE_lower_angle_TE = np.arctan(dydx_lower_TE) * 180 / np.pi
    TE_upper_angle_vec[i] = LE_upper_angle_TE
    TE_lower_angle_vec[i] = LE_upper_angle_TE
    # Camber
    camber = (upper_curve + lower_curve) / 2
    camber_line[:,i] = camber
    max_camber = max(camber)
    max_camber_vec[i] = max_camber
    max_camber_index = np.where(camber == max_camber)[0][0]
    max_camber_x = x_interp[max_camber_index]
    max_camber_y = camber[max_camber_index]
    max_camber_x_loc[i] = max_camber_x
    # Thickness
    thickness = abs(upper_curve) + abs(lower_curve)
    max_thickness = max(thickness)
    max_thickness_vec[i] = max_thickness
    max_thickness_index = np.where(thickness == max_thickness)[0][0]
    max_thickness_x = x_interp[max_thickness_index]
    max_thickness_y_upper = upper_curve[max_thickness_index]
    max_thickness_y_lower = lower_curve[max_thickness_index]
    x_thickness = [max_thickness_x, max_thickness_x]
    y_thickness = [max_thickness_y_upper, max_thickness_y_lower] 
    max_thickness_x_loc[i] = max_thickness_x
    # Plotting
    # plt.plot(x_interp,y_interp_upper, label= 'Interpolation')
    os.chdir(path)
    # file_path = f"{path}/{file}"
    airfoil = open(file_list[i])
    lines = airfoil.readlines()[1:]
    # loading lines and converting them to vectors containing x,y coordinates 
    airfoil_coordinates = np.loadtxt(lines)
    x_coord = airfoil_coordinates[:,0]
    y_coord = airfoil_coordinates[:,1]
    # finding min index of x coordinate data
    min_index = np.min(np.where(x_coord == np.min(x_coord)))
    # select x,y coordinates for upper and lower surface from data 
    x_coord_upper = x_coord[0:min_index+1]
    y_coord_upper = y_coord[0:min_index+1]
    x_coord_lower = x_coord[min_index:]
    y_coord_lower = y_coord[min_index:]
    axs[0].plot(x_coord_upper,y_coord_upper,marker = 'o',markersize = 2, color = 'maroon', label = 'Raw data')
    axs[0].plot(x_coord_lower,y_coord_lower,marker = 'o',markersize = 2, color = 'maroon')
    os.chdir(parent_directory)
    axs[0].plot(x_interp, prob['upper_B_spline_curve'],color = 'navy', label = 'B-spline')
    axs[0].plot(x_interp, prob['lower_B_spline_curve'], color = 'navy' )
    xs = ctrl_points_upper[:,0,i] 
    ys = ctrl_points_upper[:,1,i]
    ys_lower = ctrl_points_lower[:,1,i]
    axs[0].plot(xs,ys,marker = '*', linestyle = 'none', color = 'gold', label = 'Ctrl points')
    axs[0].plot(xs,ys_lower,marker = '*', linestyle = 'none', color = 'gold')
    # zip joins x and y coordinates in pairs
    for x,y in zip(xs,ys):
        label = "({:.3f},{:.3f})".format(x,y)
        axs[0].annotate(label, # this is the text
                    (x,y), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,5), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center
    for x,y in zip(xs,ys_lower):
        label = "({:.3f},{:.3f})".format(x,y)
        axs[0].annotate(label, # this is the text
                    (x,y), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,5), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center
    # axs[0].set_ylim([-0.2,0.3])
    # plt.xlim([0,1])
    axs[0].legend()
    B_sp, = axs[1].plot(x_interp, prob['upper_B_spline_curve'],color = 'navy', label = 'B-spline')
    axs[1].plot(x_interp, prob['lower_B_spline_curve'], color = 'navy' )
    camber, = axs[1].plot(x_interp, camber, color = 'green', label = 'camber line')
    max_camber_point, = axs[1].plot(max_camber_x,max_camber_y, color = 'green',linestyle = 'none', marker = '*', label = 'Max camber: {:.2f}'.format(max_camber))
    max_thickness, = axs[1].plot(max_thickness_x,max_thickness_y_upper, linestyle = 'none', marker = 'o', color  = 'gray', label = 'Max thickness: {:.2f}'.format(max_thickness))
    axs[1].plot(max_thickness_x,max_thickness_y_lower, linestyle = 'none', marker = 'o', color  = 'gray')
    x = np.linspace(0,0.2,100)
    upper_tngt, = axs[1].plot(x, tngnt_upper(x), color = 'red', label = 'LE upper angle: {:.2f}'.format(LE_upper_angle))
    lower_tngt, = axs[1].plot(x, tngnt_lower(x), color = 'red', linestyle = '--', label = 'LE lower angle: {:.2f}'.format(LE_lower_angle))
    x = np.linspace(0.97,1.2,100)
    upper_tngt_TE, = axs[1].plot(x, tngnt_upper_TE(x), color = 'gold', label = 'TE upper angle: {:.2f}'.format(LE_upper_angle_TE))
    lower_tngt_TE, = axs[1].plot(x, tngnt_lower_TE(x), color = 'gold', linestyle = '--', label = 'TE lower angle: {:.2f}'.format(LE_lower_angle_TE))
    C = plt.Circle((LE_radius_center + abs(R_average - LE_radius_center),0), R_average, fill = False, label = 'LE radius = {:.3f}'.format(R_average))
    # C_TE = plt.Circle((x_97,0), avg_TE_radius, fill = False, label = 'TE radius = {:.3f}'.format(avg_LE_radius))
    axs[1].add_artist(C)
    # axs[1].add_artist(C_TE)
    axs[1].legend(handles= [B_sp, camber, max_camber_point,max_thickness, upper_tngt, lower_tngt, upper_tngt_TE, lower_tngt_TE, C], ncol = 2)
    fig.suptitle('Airfoil file: ' + file_list[i] + ' | Airfoil number: {}'.format(i))
    # axs[0].set_ylim([-0.2,0.3])
    # axs[1].set_ylim([-0.2,0.3])
    plt.pause(0.05)
    axs[1].cla()
    axs[0].cla()
# np.savetxt('B_SPLINE_Y_COORDINATES_NEW.txt', B_spline_airfoil_y_coordinates)
# np.savetxt('B_SPLINE_X_COORDINATES_NEW.txt', B_spline_airfoil_x_coordinates )
# np.savetxt('B_SPLINE_CAMBER_LINE_NEW.txt', camber_line )
# np.savetxt('LE_RADIUS.txt',LE_radius_vec )
# np.savetxt('LE_UPPER_ANGLE.txt',LE_upper_angle_vec )
# np.savetxt('LE_LOWER_ANGLE.txt',LE_lower_angle_vec )
# np.savetxt('TE_UPPER_ANGLE.txt',TE_upper_angle_vec )
# np.savetxt('TE_LOWER_ANGLE.txt',TE_lower_angle_vec )
# np.savetxt('MAX_CAMBER.txt',max_camber_vec )
# np.savetxt('MAX_CAMBER_X_LOC.txt',max_camber_x_loc )
# np.savetxt('MAX_THICKNESS.txt',max_thickness_vec )
# np.savetxt('MAX_THICKNESS_LOC.txt',max_thickness_x_loc )
plt.show()