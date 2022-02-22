"""Runs an XFOIL analysis for a given airfoil and flow conditions"""
import os
import numpy as np
# import subprocess

from xfoil_python_wrapper import XfoilPythonWrapper
from plot_xfoil_data import plot_xfoil_data

# Change the current working directory to the given path (airfoil_data_path).
cwd = os.getcwd()
airfoil_data_path = cwd + '/airfoil_dat_files'
os.chdir(airfoil_data_path)
exit()

# airfoil_name = "svd_generated_airfoil_2"
airfoil_name = "naca0012"

# Set range for angle of attack as well as step size 
alpha_i = -10
alpha_f = 16
alpha_step = 0.25

# set number of iterations for xfoil
n_iter = 100

# Set range for Mach and Reynolds number
M_vec  = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ,0.7])
Re_vec = np.array([0.15e6, 0.2e6, 0.3e6, 0.5e6, 1e6, 2e6, 3e6, 4e6, 5e6, 6e6 ])

# Pre-allocate space for xfoild data 
alpha_vec = np.arange(alpha_i,alpha_f + alpha_step, 0.05)
airfoil_data_sweeps = np.zeros((7, len(M_vec),len(Re_vec), len(alpha_vec)))




for i in range(len(M_vec)):
    for j in range(len(Re_vec)):
        Re = Re_vec.copy()[j]
        M = M_vec.copy()[i]
        polar_data, dCl_da, dCd_da, dCm_da = XfoilPythonWrapper.run_xfoil(airfoil_name, Re, M, n_iter, alpha_i, alpha_f, alpha_step)

        data_size = polar_data[:,0].shape[0]
        airfoil_data_sweeps[0,i,j,0:data_size] = polar_data[:,0]
        airfoil_data_sweeps[1,i,j,0:data_size] = polar_data[:,1]
        airfoil_data_sweeps[2,i,j,0:data_size] = polar_data[:,2]
        airfoil_data_sweeps[3,i,j,0:data_size] = polar_data[:,4]
        airfoil_data_sweeps[4,i,j,0:data_size] = dCl_da
        airfoil_data_sweeps[5,i,j,0:data_size] = dCd_da
        airfoil_data_sweeps[6,i,j,0:data_size] = dCm_da



xfoil_data_path = cwd + '/xfoil_txt_files'
os.chdir(xfoil_data_path)

airfoil_data_sweeps = airfoil_data_sweeps.flatten()
txt_file_name = airfoil_name + '_polar_sweep.txt'

np.savetxt(txt_file_name,airfoil_data_sweeps)


np.loadtxt('svd_generated_airfoil_2_polar_sweep.txt')

plot_xfoil_data(M_vec,Re_vec,alpha_vec, txt_file_name)




exit()