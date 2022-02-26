import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy import interpolate

from lsdo_utils.comps.bspline_comp import  get_bspline_mtx

import torch
from torch import nn
from torch.autograd import Variable

curve_camber_thickness = np.loadtxt('txt_files/training_airfoils_camber_thickness.txt').T.astype(np.float32)
#print('curve_camber_thickness',curve_camber_thickness.shape) #(1527, 602)
curve_camber_thickness = curve_camber_thickness[0:-2,:]
#print('curve_camber_thickness',curve_camber_thickness.shape) #(1525, 602)
control_camber_thickness = np.loadtxt('txt_files/20_camber_thickness_b_spline_cp.txt').T.astype(np.float32)
#print('control_camber_thickness ',control_camber_thickness.shape) #(1527, 20)
control_camber_thickness = control_camber_thickness[0:-2,:]
#print('control_camber_thickness ',control_camber_thickness.shape) #(1525, 20)
latent_parameters = np.loadtxt('txt_files/parameters_fft_latent.txt').astype(np.float32)
# print('latent_parameters',latent_parameters.shape) (1527, 20)
latent_parameters = latent_parameters[0:-2,:] 
# print('latent_parameters',latent_parameters.shape) (1525, 20)

# Normalize the training latent parameters to fit in the range -2 to 2 (can be any range e.g. -3 to 3 by changing 4 to 6 and 2 to 3)
for i in range(20):
    norm_constant =  max(latent_parameters[:,i]) - min(latent_parameters[:,i]) 
    min_value     =  min(latent_parameters[:,i])
    latent_parameters[:,i] = 4 * (latent_parameters[:,i] - min_value) / norm_constant - 2

# Generate B-spline matrix 
bspline_matrix = get_bspline_mtx(10, 301, order = 4)
bspline_matrix = bspline_matrix.todense()
bspline_matrix = bspline_matrix.astype(np.float32)
# print('bspline_matrix', bspline_matrix.shape) (301, 10)

# Playing around with the dimension of the noise vector can improve training of the generator 
noise_dim = 2
latent_dim = 18
G_input_dim = noise_dim + latent_dim
G_output_dim = control_camber_thickness.shape[1]
D_input_dim = curve_camber_thickness.shape[1]
print('G_input_dim', G_input_dim) #60
print('G_output_dim', G_output_dim) #20
print('D_input_dim', D_input_dim) #602


data_curve_y = torch.from_numpy(curve_camber_thickness)
data_latent_parameters = torch.from_numpy(latent_parameters)

batch_size = 61
dataset = torch.utils.data.TensorDataset(*(data_curve_y,data_latent_parameters))
data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
print('data_iter type:', type(data_iter))

# Making the neural net of the Generator and Discriminator deeper with wider (more neurons) can help, but it slows down training significantly
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.G_net = nn.Sequential(nn.Linear(G_input_dim, 30), nn.ReLU(), 
                      nn.Linear(30, 40), nn.ReLU(),
                      nn.Linear(40, 30), nn.ReLU(), 
                      nn.Linear(30, G_output_dim))
    def forward(self, zc, fft = False):
        control_camber_thickness_fake = self.G_net(zc)
        control_camber_thickness_fake[:,0] = 0
        control_camber_thickness_fake[:,9] = 0
        control_camber_thickness_fake[:,10] = 0
        control_camber_thickness_fake[:,19] = 0
        curve_camber_fake = (torch.matmul(Variable(FloatTensor(bspline_matrix)),control_camber_thickness_fake[:,0:10].T)).T
        curve_thickness_fake = (torch.matmul(Variable(FloatTensor(bspline_matrix)),control_camber_thickness_fake[:,10:20].T)).T
        curve_camber_thickness_fake = torch.cat((curve_camber_fake, curve_thickness_fake),-1)        
        #thickness_positive = control_camber_thickness_fake[:,:10] - control_camber_thickness_fake[:,10:]
        #r_positive = torch.mean(torch.maximum(torch.zeros(cp_nonintersection.shape), cp_nonintersection))
        fft_latent_parameters_fake = torch.zeros(curve_camber_thickness_fake.shape[0],latent_dim)
        if fft:           
            for i in range(curve_camber_thickness_fake.shape[0]):
                upper_surface = curve_camber_thickness_fake[i,0:301]+0.5*curve_camber_thickness_fake[i,301:602]
                lower_surface = curve_camber_thickness_fake[i,0:301]-0.5*curve_camber_thickness_fake[i,301:602]
                # Camber
                camber = (upper_surface + lower_surface) / 2
                # Thickness
                thickness = abs(upper_surface - lower_surface)
                camber_fft = torch.fft.fft(camber)
                thickness_fft = torch.fft.fft(thickness)
                fft_latent_parameters_fake[i,0:9] = camber_fft[0:9].real
                fft_latent_parameters_fake[i,9:18] = thickness_fft[0:9].real
        return curve_camber_thickness_fake, fft_latent_parameters_fake
        

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.D_net = nn.Sequential(nn.Linear(D_input_dim,400), nn.ReLU(), 
                      nn.Linear(400, 300), nn.ReLU(),
                      nn.Linear(300, 200), nn.ReLU(),
                      nn.Linear(200, 100), nn.ReLU())
        self.adv_layer = nn.Sequential(nn.Linear(100,1))
        self.latent_layer = nn.Sequential(nn.Linear(100, latent_dim))
    def forward(self, curve_y_coords_fake):
        out = self.D_net(curve_y_coords_fake)
        validity = self.adv_layer(out)
        latent_code = self.latent_layer(out)
        return validity, latent_code


loss_G_list = []
loss_D_list = []
loss_info_list = []
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
net_G = Generator()
net_D = Discriminator()

adversarial_loss = torch.nn.MSELoss()
continuous_loss = torch.nn.MSELoss() 

lambda_con = 0.5
lambda_fft = 1e-6 #test0:0.5
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, lr_info, noise_dim, latent_dim):
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)#betas=(opt.b1, opt.b2)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)
    trainer_info = torch.optim.Adam(
        itertools.chain(net_G.parameters(), net_D.parameters()), lr=lr_info)
    for epoch in range(num_epochs):
        for i, (Xy,Xc) in enumerate(data_iter):
            Z = torch.normal(0, 1, size=(batch_size, noise_dim))
            C = torch.normal(0, 1, size=(batch_size, latent_dim))

            #  Train Generator
            trainer_G.zero_grad()
            ones = torch.ones((batch_size,), device=Z.device)
            G_Xy,_ = net_G(torch.cat((Z,C),-1))
            validity, _ = net_D(G_Xy)
            loss_G = adversarial_loss(validity, ones.reshape(validity.shape))
            loss_G.backward()
            trainer_G.step()
            
            #  Train Discriminator
            trainer_D.zero_grad()
            ones = torch.ones((batch_size,), device=torch.cat((Xy,Xc),-1).device)
            zeros = torch.zeros((batch_size,), device=torch.cat((Xy,Xc),-1).device)            
            real_pred,_ = net_D(Xy)
            fake_pred,_ = net_D(G_Xy.detach())
            loss_D = (adversarial_loss(real_pred, ones.reshape(real_pred.shape)) +
                      adversarial_loss(fake_pred, zeros.reshape(fake_pred.shape))) / 2
            loss_D.backward()
            trainer_D.step()
            
            # Information Loss
            trainer_info.zero_grad()
            G_Xy,G_latent = net_G(torch.cat((Z,C),-1), True)
            _, pred_c = net_D(G_Xy)
            loss_info = lambda_con * continuous_loss(pred_c, C) + lambda_fft * adversarial_loss(G_latent, C)
            loss_info.backward()
            trainer_info.step() 

            if i % 80 == 0:           
                print( "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                % (epoch, num_epochs, i, len(data_iter), loss_D.item(), loss_G.item(), loss_info.item()))
        loss_D_list.append(loss_D.item())
        loss_G_list.append(loss_G.item())
        loss_info_list.append(loss_info.item())
        

# Comment out lines 177 to 187 if you just want to run a trained generator
lr_D, lr_G,lr_info, num_epochs = 0.00001, 0.00001, 0.00001, 2000 #50
test_str = '_camber_thickness_modified_test0_1e-6_' + "%s"%num_epochs

train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, lr_info,noise_dim, latent_dim)

plt.figure(figsize=(8,6))
plt.plot(range(num_epochs), loss_G_list)
plt.plot(range(num_epochs), loss_D_list)
plt.plot(range(num_epochs), loss_info_list)
plt.legend(['Generator', 'Discriminator','Infomation'])
plt.savefig('figures/Loss'+test_str)

# # Saving trained generator: IF YOU JUST WANT TO RUN A TRAINED GENERATOR, COMMENT OUT LINE 192! 
generator_model_path =  'GAN_GENERATOR_coordinates' +test_str
torch.save(net_G.state_dict(),generator_model_path)

Gen = Generator()
Gen.load_state_dict(torch.load(generator_model_path))
curve_x_coords = np.loadtxt('txt_files/training_airfoils_x_coordinates.txt').astype(np.float32)

# Generating random airfoils for each latent variable 
test_size = 10
Z = torch.zeros(test_size,noise_dim)
C = torch.zeros(test_size,latent_dim)
latent_code  = np.linspace(-2,2,test_size)
fig, axs = plt.subplots(5, 4, figsize=(15,7.2))#
axs = axs.flatten()
for latent_c in range(latent_dim):
    for i in range(test_size):
        C[i,latent_c] = latent_code[i]
    fake_data_test,_ = Gen(torch.cat((Z,C),-1))
    fake_data_test = fake_data_test.detach().numpy()
    color = plt.cm.rainbow(np.linspace(0, 1, test_size))
    for i, c in zip(range(test_size), color):
        axs[latent_c].plot(curve_x_coords[0:301,0], fake_data_test[i,0:301]+0.5*fake_data_test[i,301:602], c = c, label='C_'+' = {0}'.format(round(latent_code[i],2)))
        axs[latent_c].plot(curve_x_coords[0:301,0], fake_data_test[i,0:301]-0.5*fake_data_test[i,301:602], c = c)     

fig.tight_layout()
plt.savefig('figures/Fake_airfoil'+test_str)

print('Start')
test_size = 100000
Z = torch.zeros(test_size,noise_dim)
C = torch.zeros(test_size,latent_dim)
for i in range(test_size):
    latent_code = torch.normal(0,1,size = (1,latent_dim))
    C[i,:] = latent_code
C[C>2] = 2
C[C<-2] = -2
fake_data_test,_ = Gen(torch.cat((Z,C),-1))
fake_data_test = fake_data_test.detach().numpy()
print('End')
physical_parameters = np.zeros((test_size,6))
fft_latent_parameters = np.zeros((test_size,latent_dim))

for i in range(test_size):
    if i % 5000 == 0:
        print(i)
    upper_surface = fake_data_test[i,0:301]+0.5*fake_data_test[i,301:602]
    lower_surface = fake_data_test[i,0:301]-0.5*fake_data_test[i,301:602]

    # Camber
    camber = (upper_surface + lower_surface) / 2
    average_camber = np.average(camber)
    max_camber = max(abs(camber))
    # if average_camber < 0:
    #     max_camber = max(abs(camber))
    #     max_camber_index = np.where(camber == -1 * max_camber)[0][0]
    #     max_camber_x = curve_x_coords[0:301,0][max_camber_index]
    #     max_camber_y = camber[max_camber_index]
    # else:
    #     max_camber = max(camber)
    #     max_camber_index = np.where(camber == max_camber)[0][0]
    #     max_camber_x = curve_x_coords[0:301,0][max_camber_index]
    #     max_camber_y = camber[max_camber_index]

    # Thickness
    thickness = abs(upper_surface - lower_surface)
    max_thickness = max(thickness)

    max_thickness_index = np.where(thickness == max_thickness)[0][0]
    max_thickness_x = curve_x_coords[0:301,0][max_thickness_index]
    max_thickness_y_upper = upper_surface[max_thickness_index]
    max_thickness_y_lower = lower_surface[max_thickness_index]
    x_thickness = [max_thickness_x, max_thickness_x]
    y_thickness = [max_thickness_y_upper, max_thickness_y_lower] 
    max_thickness_x_loc = max_thickness_x

    # LE/TE angle
    x_der = 0.02
    x_der_TE = 0.98
    tck_camber = interpolate.splrep(curve_x_coords[0:301,0],camber)

    y_der_camber_LE = interpolate.splev(x_der,tck_camber)
    dydx_camber_LE= interpolate.splev(x_der,tck_camber,der=1)
    LE_angle = np.arctan(dydx_camber_LE) * 180 / np.pi

    
    y_der_camber_TE = interpolate.splev(x_der_TE,tck_camber)
    dydx_camber_TE = interpolate.splev(x_der_TE,tck_camber,der=1)
    TE_angle = np.arctan(dydx_camber_TE) * 180 / np.pi

    n = 30
    radius_upper = np.empty((n,))
    radius_lower = np.empty((n,))
    for k in range(n):
        radius_upper[k] = ((upper_surface[k] - 0)**2 + (curve_x_coords[0:301,0][k] - 0.02)**2)**0.5
        radius_lower[k] = ((lower_surface[k] - 0)**2 + (curve_x_coords[0:301,0][k] - 0.02)**2)**0.5
    #LE_diameter = abs(upper_surface[35] - lower_surface[35])
    RU = np.average(radius_upper)
    RL = np.average(radius_lower)    
    R = (RU + RL) / 2

    physical_parameters[i,0] = max_camber
    physical_parameters[i,1] = max_thickness
    physical_parameters[i,2] = max_thickness_x_loc
    physical_parameters[i,3] = LE_angle
    physical_parameters[i,4] = TE_angle
    physical_parameters[i,5] = R

    camber_fft = np.fft.fft(camber)
    thickness_fft = np.fft.fft(thickness)

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
    # fft_latent_parameters[i,18] = thickness_fft[8].real
    # fft_latent_parameters[i,19] = thickness_fft[9].real



np.savetxt('txt_files/generated_parameters_physical_camber_thickness.txt', physical_parameters)
np.savetxt('txt_files/generated_parameters_fft_latent_camber_thickness.txt',fft_latent_parameters)

fig, axs = plt.subplots(2,3, figsize= (14,8))
generated_physical_parameters = physical_parameters#np.loadtxt('generated_parameters_physical.txt')
training_physical_parameters = np.loadtxt('txt_files/parameters_physical.txt')
physical_parameters_label = ['Max camber', 'Max thickness', 'x-coord max thickness', 'LE angle', 'TE angle', 'LE radius']
for i in range(2):
    for j in range(3):
        index = 3 * i + j 
        axs[i,j].hist(generated_physical_parameters[:,index],density = True, bins = 120, color = 'blue',alpha = 0.5, label = 'GAN generated airfoils')
        axs[i,j].hist(training_physical_parameters[:,index],density = True, bins = 120, color = 'red',alpha = 0.5, label = 'Training airfoils')
        title_string = physical_parameters_label[index]
        axs[i,j].title.set_text(title_string)
        if (i == 0) and (j == 0):
            axs[i,j].legend(loc = 'upper right')

# fig.legend(handles, labels, ncol= 2,loc='upper center')
fig.tight_layout()
plt.savefig('figures/Histogram_physical'+test_str)

fig, axs = plt.subplots(6,3, figsize= (14,8))
generated_fft_latent_parameters = fft_latent_parameters
training_fft_latent_parameters = np.loadtxt('txt_files/parameters_fft_latent.txt')
for i in range(6):
    for j in range(3):
        index = 3 * i + j 
        axs[i,j].hist(generated_fft_latent_parameters[:,index],density = True, bins = 120, color = 'blue',alpha = 0.5, label = 'GAN generated airfoils')
        axs[i,j].hist(training_fft_latent_parameters[:,index],density = True, bins = 120, color = 'red',alpha = 0.5, label = 'Training airfoils')
        axs[i,j].title.set_text('Latent parameter c_'+ "%s"%index)
        if (i == 0) and (j == 0):
            axs[i,j].legend(loc = 'upper right')

# fig.legend(handles, labels, ncol= 2,loc='upper center')
fig.tight_layout()
plt.savefig('figures/Histogram_fft_latent'+test_str)
plt.show()
exit()
