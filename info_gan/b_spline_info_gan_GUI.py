
import tkinter as Tk
from tkinter import ttk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from matplotlib.pyplot import cm
from lsdo_utils.comps.bspline_comp import  get_bspline_mtx
from scipy import interpolate

import torch
from torch import nn
from d2l import torch as d2l

import numpy as np
import os

from torch.autograd import Variable
FloatTensor = torch.FloatTensor




cwd = os.getcwd() 
txt_file_path = cwd + '/txt_files'
os.chdir(txt_file_path)


curve_y_coords = np.loadtxt('training_airfoils_y_coordinates.txt').T.astype(np.float32)
curve_y_coords = curve_y_coords[0:-2,:]

control_y_coords = np.loadtxt('20_y_b_spline_cp.txt').T.astype(np.float32)
control_y_coords = control_y_coords[0:-2,:]

latent_parameters = np.loadtxt('fft_latent_parameters.txt').astype(np.float32)
latent_parameters = latent_parameters[0:-2,:]

# Normalize the training latent parameters to fit in the range -2 to 2 (can be any range e.g. -3 to 3 by changing 4 to 6 and 2 to 3)
for i in range(20):
    norm_constant =  max(latent_parameters[:,i]) - min(latent_parameters[:,i]) 
    min_value     =  min(latent_parameters[:,i])
    latent_parameters[:,i] = 4 * (latent_parameters[:,i] - min_value) / norm_constant - 2


# Generate B-spline matrix 
bspline_matrix = get_bspline_mtx(10, 301, order = 4)
bspline_matrix = bspline_matrix.todense()
bspline_matrix = bspline_matrix.astype(np.float32)

# Playing around with the dimension of the noise vector can improve training of the generator 
noise_dim = 40
latent_dim = 20
G_input_dim = noise_dim + latent_dim
G_output_dim = control_y_coords.shape[1]
D_input_dim = curve_y_coords.shape[1]


data_curve_y = torch.from_numpy(curve_y_coords)
data_latent_parameters = torch.from_numpy(latent_parameters)



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.G_net = nn.Sequential(nn.Linear(G_input_dim, 200), nn.ReLU(), 
                      nn.Linear(200, 300), nn.ReLU(),
                      nn.Linear(300, 400), nn.ReLU(),
                      nn.Linear(400, 400), nn.ReLU(),
                      nn.Linear(400, 400), nn.ReLU(),
                      nn.Linear(400, 400), nn.ReLU(), 
                      nn.Linear(400, 300), nn.ReLU(), 
                      nn.Linear(300, G_output_dim))
    def forward(self, zc):
        control_y_coords_fake = self.G_net(zc)
        control_y_coords_fake[:,0] = 0
        control_y_coords_fake[:,9] = 0
        control_y_coords_fake[:,10] = 0
        control_y_coords_fake[:,19] = 0
        curve_y_coords_upper_fake = (torch.matmul(Variable(FloatTensor(bspline_matrix)),control_y_coords_fake[:,0:10].T)).T
        curve_y_coords_lower_fake = (torch.matmul(Variable(FloatTensor(bspline_matrix)),control_y_coords_fake[:,10:20].T)).T
        curve_y_coords_fake = torch.cat((curve_y_coords_upper_fake, curve_y_coords_lower_fake),-1) 
        cp_adjacent = torch.norm(control_y_coords_fake[:,1:] - control_y_coords_fake[:,:-1], dim = 1)
        r_adjacent = torch.mean(cp_adjacent)
        cp_nonintersection = curve_y_coords_fake[:,:10] - control_y_coords_fake[:,10:]
        r_nonintersection = torch.mean(torch.maximum(torch.zeros(cp_nonintersection.shape), cp_nonintersection))
        return curve_y_coords_fake, r_adjacent, r_nonintersection
        return curve_y_coords_fake, r_adjacent, r_nonintersection


generator_name       = '/FFT_GAN_GENERATOR_20_CP'
generator_model_path =  cwd + generator_name
Gen = Generator()
Gen.load_state_dict(torch.load(generator_model_path))







x_bspline = np.loadtxt('airfoil_x_coordinates.txt').astype(np.float32)
x_interp = x_bspline[0:301,0]
class Interface:

    def __init__(self):

        # Create main window regardless of what's being plotted
        self.root = Tk.Tk()

        # Slider values
        self.w_list = []
        for i in range(20):
            self.w_list.append(Tk.DoubleVar(self.root))

        # Main frame
        self.frame = ttk.Frame(
            self.root, borderwidth=1, relief='solid')
        self.frame.grid()

        # Frame for plotting
        self.plotframe = ttk.Frame(self.frame)
        self.plotframe.grid()

        # Instantiate the MPL figure
        self.figure = plt.figure(facecolor="white", figsize = (12,8))
        plt.plot([0.0, 0.01, 0.02], [0.0, 0.01, 0.03])

        # Link the MPL figure onto the TK canvas_tab1 and pack it
        self.canvas = FigureCanvasTkAgg(
            self.figure, master=self.plotframe)
        self.canvas.get_tk_widget().grid()
        self.plotframe.grid(row=0, column=0)

        slider_frame = ttk.Frame(
            self.plotframe, borderwidth=5, relief='ridge')
        slider_frame.grid(row=0, column=1, sticky=(Tk.W), columnspan=2)
        self.root.wm_title("Airfoil B-Spline InfoGAN")
        
        mode_feature = [' Max Camber',' Max thickness',' Max thickness x-coord',' LE angle','TE angle','LE radius']

        for i in range(10):
            print(i)
            for j in range(2):
                iteration_slider = Tk.Scale(slider_frame, orient="horizontal",
                                    from_=-2, to=2,
                                    showvalue=True,
                                    length=200,
                                    resolution=0.01,
                                    variable=self.w_list[2 * i + j],
                                    tickinterval=10)
                iteration_slider.grid(
                    column=20 * j, row=2 * i+1, columnspan=3, sticky='nesw')
                label = ttk.Label(slider_frame, text='C '+ str(2 * i + j + 1)) #+ ':' + mode_feature[2 * i + j])
                label.grid(column=20 * j, row=2 * i)

        refreshbutton = Tk.Button(
            self.plotframe, text="Update Plot", command=self.update_plot, height=3)
        refreshbutton.grid(column=0, row=len(self.w_list)+1, sticky='nesw')

    def update_plot(self):
        print('\n----------updated plot----------')
        w_plot = []
        x_axis = []
        for index, i in enumerate(self.w_list):
            print('W', index, ' = ', i.get())
            w_plot.append(float(i.get()))
            x_axis.append(index)
        print(w_plot)
        Z = torch.zeros(1,noise_dim)
        C = torch.zeros(1,latent_dim)
        for i in range(latent_dim):
            C[0,i] = w_plot[i]

        print(C)
        fake_data_test = Gen(torch.cat((Z,C),-1))[0].detach().numpy()
        average_upper_surface = fake_data_test[0,0:301]
        average_lower_surface = fake_data_test[0,301:602]

        # Camber
        camber = (average_upper_surface + average_lower_surface) / 2
        average_camber = np.average(camber)
        if average_camber < 0:
            max_camber = max(abs(camber))
            max_camber_index = np.where(camber == -1 * max_camber)[0][0]
            max_camber_x = x_interp[max_camber_index]
            max_camber_y = camber[max_camber_index]
        else:
            max_camber = max(camber)
            max_camber_index = np.where(camber == max_camber)[0][0]
            max_camber_x = x_interp[max_camber_index]
            max_camber_y = camber[max_camber_index]


        # LE angle 
        x_der = 0.02
        tck_camber = interpolate.splrep(x_interp,camber)

        y_der_camber_LE = interpolate.splev(x_der,tck_camber)
        dydx_camber_LE= interpolate.splev(x_der,tck_camber,der=1)
        tngnt_camber_LE = lambda x: dydx_camber_LE * x + (y_der_camber_LE - dydx_camber_LE * x_der)
        LE_angle = np.arctan(dydx_camber_LE) * 180 / np.pi

        # TE angle 
        x_der_TE = 0.98

        y_der_camber_TE = interpolate.splev(x_der_TE,tck_camber)
        dydx_camber_TE = interpolate.splev(x_der_TE,tck_camber,der=1)
        tngnt_camber_TE = lambda x: dydx_camber_TE * x + (y_der_camber_TE - dydx_camber_TE * x_der_TE)
        TE_angle = np.arctan(dydx_camber_TE) * 180 / np.pi

        # Thickness
        thickness = abs(average_upper_surface) + abs(average_lower_surface)
        max_thickness = max(thickness)
    
        max_thickness_index = np.where(thickness == max_thickness)[0][0]
        max_thickness_x = x_interp[max_thickness_index]

        max_thickness_y_upper = average_upper_surface[max_thickness_index]
        max_thickness_y_lower = average_lower_surface[max_thickness_index]
        x_thickness = [max_thickness_x, max_thickness_x]
        y_thickness = [max_thickness_y_upper, max_thickness_y_lower] 

        max_thickness_x_loc = max_thickness_x
        average_thickness = np.average(thickness)

        # LE Radius
        n = 30
        radius_upper = np.empty((n,))
        radius_lower = np.empty((n,))
        for i in range(n):
            radius_upper[i] = ((average_upper_surface[i] - 0)**2 + (x_interp[i] - 0.02)**2)**0.5
            radius_lower[i] = ((average_lower_surface[i] - 0)**2 + (x_interp[i] - 0.02)**2)**0.5

        print(abs(average_upper_surface[30] - average_lower_surface[30]))
        print(abs(average_upper_surface[35] - average_lower_surface[35]))
        LE_diameter = abs(average_upper_surface[35] - average_lower_surface[35])

        RU = np.average(radius_upper)
        RL = np.average(radius_lower)
        
        R = (RU + RL) / 2
        

        self.figure.clf()
        a = self.figure.add_subplot(111)
        a.plot(x_bspline[0:301,0], fake_data_test[0,0:301], color = 'navy')
        a.plot(x_bspline[0:301,0], fake_data_test[0,301:602], color = 'navy')

        camber, = a.plot(x_interp,camber,label = 'Camber line', color = 'green')
        max_camber_point, = a.plot(max_camber_x,max_camber_y, color = 'green',linestyle = 'none', marker = '*', label = 'Max camber: {:.2f} at x = {:.2f}'.format(max_camber, max_camber_x))
        
        max_thickness, = a.plot(max_thickness_x,max_thickness_y_upper, linestyle = 'none', marker = 'o', color  = 'gray', label = 'Max thickness: {:.2f} at x = {:.2f}'.format(max_thickness,max_thickness_x_loc))
        a.plot(max_thickness_x,max_thickness_y_lower, linestyle = 'none', marker = 'o', color  = 'gray')

        x = np.linspace(0,0.2,100)
        upper_tngt, = a.plot(x, tngnt_camber_LE(x), color = 'red', label = 'LE upper angle: {:.2f}'.format(LE_angle))
    
        x = np.linspace(0.97,1.2,100)
        upper_tngt_TE, = a.plot(x, tngnt_camber_TE(x), color = 'gold', label = 'TE upper angle: {:.2f}'.format(TE_angle))

        C = plt.Circle((x_interp[35],0), LE_diameter/2, fill  = False, label = 'LE radius = {:.3f}'.format(R))
        a.add_artist(C)

        a.text(0.98, 0.51, 'Mean camber: {:.3f}'.format(average_camber) + '\n' + 'Mean thickness: {:.3f}'.format(average_thickness), bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10},
        verticalalignment='center', horizontalalignment='right',
        transform=a.transAxes,
        color='black', fontsize=10)

        a.legend(handles= [camber, max_camber_point,max_thickness, upper_tngt, upper_tngt_TE,  C], ncol = 2)
        a.set_ylim([-0.2, 0.3])
        self.canvas.draw()

    def run(self):
        Tk.mainloop()


Interface().run()
