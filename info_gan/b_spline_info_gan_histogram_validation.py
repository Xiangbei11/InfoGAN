import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy import interpolate
from lsdo_utils.comps.bspline_comp import  get_bspline_mtx

cwd = os.getcwd()
os.chdir(cwd + '/txt_files')


fig, axs = plt.subplots(2,3, figsize= (14,8))
generated_latent_code = np.loadtxt('generated_original_latent_parameters.txt')
training_latent_code = np.loadtxt('original_latent_parameters.txt')
fft_latent_parameter = np.arange(1,21,1)
original_latent_parameter = ['Max camber', 'Max thickness', 'x-coord max thickness', 'LE angle', 'TE angle', 'LE radius']

for i in range(2):
    for j in range(3):
        index = 3 * i + j 
        axs[i,j].hist(generated_latent_code[:,index],density = True, bins = 120, color = 'blue',alpha = 0.5, label = 'GAN generated original latent code')
        axs[i,j].hist(training_latent_code[:,index],density = True, bins = 120, color = 'red',alpha = 0.5, label = 'Training original latent code')
        title_string = original_latent_parameter[index]
        axs[i,j].title.set_text(title_string)
        if (i == 1) and (j == 2):
            axs[i,j].legend()
            # handles, labels = axs[i,j].get_legend_handles_labels()

# fig.legend(handles, labels, ncol= 2,loc='upper center')
fig.tight_layout()
plt.show()
exit()


control_y_coords = np.loadtxt('20_y_b_spline_cp.txt').T.astype(np.float32)
bspline_matrix = get_bspline_mtx(10, 301, order = 4)
bspline_matrix = bspline_matrix.todense()
bspline_matrix = bspline_matrix.astype(np.float32)

FloatTensor = torch.FloatTensor


noise_dim = 40
latent_dim = 20
G_input_dim = noise_dim + latent_dim
G_output_dim = control_y_coords.shape[1]

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

generator_model_path =  cwd + '/FFT_GAN_GENERATOR_20_CP'

Gen = Generator()
Gen.load_state_dict(torch.load(generator_model_path))

test_size = 150000
Z = torch.zeros(test_size,noise_dim)
C = torch.zeros(test_size,latent_dim)
for i in range(test_size):
    latent_code = torch.normal(0,1,size = (1,latent_dim))
    for j in range(latent_dim):
        if latent_code[0][:][j] > 2:
            latent_code[0][:][j] = 2
        if latent_code[0][:][j] < -2:
            latent_code[0][:][j] = -2
    C[i,:] = latent_code
fake_data_test,_ ,_ = Gen(torch.cat((Z,C),-1))
fake_data_test = fake_data_test.detach().numpy()
x_bspline = np.loadtxt('airfoil_x_coordinates.txt').astype(np.float32)



fig, axs = plt.subplots(1,1, figsize= (14,8))
original_latent_parameters = np.zeros((test_size,6))
fft_latent_parameters = np.zeros((test_size,latent_dim))

for i in range(test_size):
    print(i)
    upper_surface = fake_data_test[i,0:301]
    lower_surface = fake_data_test[i,301:602]

    # Camber
    camber = (upper_surface + lower_surface) / 2
    average_camber = np.average(camber)
    max_camber = max(abs(camber))
    # if average_camber < 0:
    #     max_camber = max(abs(camber))
    #     max_camber_index = np.where(camber == -1 * max_camber)[0][0]
    #     max_camber_x = x_bspline[0:301,0][max_camber_index]
    #     max_camber_y = camber[max_camber_index]
    # else:
    #     max_camber = max(camber)
    #     max_camber_index = np.where(camber == max_camber)[0][0]
    #     max_camber_x = x_bspline[0:301,0][max_camber_index]
    #     max_camber_y = camber[max_camber_index]

    

    # LE/TE angle
    x_der = 0.02
    x_der_TE = 0.98
    tck_camber = interpolate.splrep(x_bspline[0:301,0],camber)

    y_der_camber_LE = interpolate.splev(x_der,tck_camber)
    dydx_camber_LE= interpolate.splev(x_der,tck_camber,der=1)
    LE_angle = np.arctan(dydx_camber_LE) * 180 / np.pi

    

    y_der_camber_TE = interpolate.splev(x_der_TE,tck_camber)
    dydx_camber_TE = interpolate.splev(x_der_TE,tck_camber,der=1)
    TE_angle = np.arctan(dydx_camber_TE) * 180 / np.pi

    


    # Thickness
    thickness = abs(upper_surface) + abs(lower_surface)
    max_thickness = max(thickness)

    max_thickness_index = np.where(thickness == max_thickness)[0][0]
    max_thickness_x = x_bspline[0:301,0][max_thickness_index]

    max_thickness_y_upper = upper_surface[max_thickness_index]
    max_thickness_y_lower = lower_surface[max_thickness_index]
    x_thickness = [max_thickness_x, max_thickness_x]
    y_thickness = [max_thickness_y_upper, max_thickness_y_lower] 

    max_thickness_x_loc = max_thickness_x


    n = 30
    radius_upper = np.empty((n,))
    radius_lower = np.empty((n,))
    for k in range(n):
        radius_upper[k] = ((upper_surface[k] - 0)**2 + (x_bspline[0:301,0][k] - 0.02)**2)**0.5
        radius_lower[k] = ((lower_surface[k] - 0)**2 + (x_bspline[0:301,0][k] - 0.02)**2)**0.5

    # print(abs(upper_surface[30] - lower_surface[30]))
    # print(abs(upper_surface[35] - lower_surface[35]))
    LE_diameter = abs(upper_surface[35] - lower_surface[35])

    RU = np.average(radius_upper)
    RL = np.average(radius_lower)
    
    R = (RU + RL) / 2
    # print(R)

    original_latent_parameters[i,0] = max_camber
    original_latent_parameters[i,1] = max_thickness
    original_latent_parameters[i,2] = max_thickness_x_loc
    original_latent_parameters[i,3] = LE_angle
    original_latent_parameters[i,4] = TE_angle
    original_latent_parameters[i,5] = R

    camber_fft = np.fft.fft(camber)
    thickness_fft = np.fft.fft(thickness)


    # latent_parameters[i,0] = camber[10]
    # latent_parameters[i,1] = camber[20]
    # latent_parameters[i,2] = camber[50]
    # latent_parameters[i,3] = camber[90]
    # latent_parameters[i,4] = camber[130]
    # latent_parameters[i,5] = camber[170]
    # latent_parameters[i,6] = camber[210]
    # latent_parameters[i,7] = camber[250]
    # latent_parameters[i,8] = camber[280]
    # latent_parameters[i,9] = camber[290]
    # latent_parameters[i,10] = thickness[10]
    # latent_parameters[i,11] = thickness[20]
    # latent_parameters[i,12] = thickness[50]
    # latent_parameters[i,13] = thickness[90]
    # latent_parameters[i,14] = thickness[130]
    # latent_parameters[i,15] = thickness[170]
    # latent_parameters[i,16] = thickness[210]
    # latent_parameters[i,17] = thickness[250]
    # latent_parameters[i,18] = thickness[280]
    # latent_parameters[i,19] = thickness[290]


    fft_latent_parameters[i,0] = camber_fft[0].real
    fft_latent_parameters[i,1] = camber_fft[1].real
    fft_latent_parameters[i,2] = camber_fft[2].real
    fft_latent_parameters[i,3] = camber_fft[3].real
    fft_latent_parameters[i,4] = camber_fft[4].real
    fft_latent_parameters[i,5] = camber_fft[5].real
    fft_latent_parameters[i,6] = camber_fft[6].real
    fft_latent_parameters[i,7] = camber_fft[7].real
    fft_latent_parameters[i,8] = camber_fft[8].real
    fft_latent_parameters[i,9] = camber_fft[9].real
    fft_latent_parameters[i,10] = thickness_fft[0].real
    fft_latent_parameters[i,11] = thickness_fft[1].real
    fft_latent_parameters[i,12] = thickness_fft[2].real
    fft_latent_parameters[i,13] = thickness_fft[3].real
    fft_latent_parameters[i,14] = thickness_fft[4].real
    fft_latent_parameters[i,15] = thickness_fft[5].real
    fft_latent_parameters[i,16] = thickness_fft[6].real
    fft_latent_parameters[i,17] = thickness_fft[7].real
    fft_latent_parameters[i,18] = thickness_fft[8].real
    fft_latent_parameters[i,19] = thickness_fft[9].real



np.savetxt('generated_original_latent_parameters.txt',original_latent_parameters)
np.savetxt('generated_fft_latent_parameters.txt',fft_latent_parameters)

exit()



