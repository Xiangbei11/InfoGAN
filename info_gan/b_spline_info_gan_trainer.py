import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# import tkinter as Tk
# from tkinter import ttk

from lsdo_utils.comps.bspline_comp import  get_bspline_mtx

import torch
from torch import nn
# from d2l import torch as d2l

# from scipy import interpolate
import numpy as np
import os

cwd = os.getcwd() 
txt_file_path = cwd + '/txt_files'
os.chdir(txt_file_path)

curve_y_coords = np.loadtxt('training_airfoils_y_coordinates.txt').T.astype(np.float32)
# print('curve_y_coords',curve_y_coords.shape) (1527, 602)
curve_y_coords = curve_y_coords[0:-2,:]
# print('curve_y_coords',curve_y_coords.shape) (1525, 602)
control_y_coords = np.loadtxt('20_y_b_spline_cp.txt').T.astype(np.float32)
# print('control_y_coords ',control_y_coords.shape) (1527, 20)
control_y_coords = control_y_coords[0:-2,:]
# print('control_y_coords ',control_y_coords.shape) (1525, 20)
latent_parameters = np.loadtxt('fft_latent_parameters.txt').astype(np.float32)
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
noise_dim = 40
latent_dim = 20
G_input_dim = noise_dim + latent_dim
G_output_dim = control_y_coords.shape[1]
D_input_dim = curve_y_coords.shape[1]
print('G_input_dim', G_input_dim) #60
print('G_output_dim', G_output_dim) #20
print('D_input_dim', D_input_dim) #602


data_curve_y = torch.from_numpy(curve_y_coords)
data_latent_parameters = torch.from_numpy(latent_parameters)

batch_size = 61
#data_iter = d2l.load_array((data_curve_y,data_latent_parameters), batch_size)
dataset = torch.utils.data.TensorDataset(*(data_curve_y,data_latent_parameters))
data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
print('data_iter type:', type(data_iter))

# Making the neural net of the Generator and Discriminator deeper with wider (more neurons) can help, but it slows down training significantly
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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.D_net = nn.Sequential(nn.Linear(D_input_dim,300), nn.ReLU(), 
                      nn.Linear(300, 300), nn.ReLU(),
                      nn.Linear(300, 400), nn.ReLU(),
                      nn.Linear(400, 400), nn.ReLU(),
                      nn.Linear(400, 300), nn.ReLU(),
                      nn.Linear(300, 300), nn.ReLU())
        self.adv_layer = nn.Sequential(nn.Linear(300,1))
        self.latent_layer = nn.Sequential(nn.Linear(300, latent_dim))
    def forward(self, curve_y_coords_fake):
        out = self.D_net(curve_y_coords_fake)
        validity = self.adv_layer(out)
        latent_code = self.latent_layer(out)
        return validity, latent_code


import itertools
import sys
from torch.autograd import Variable

loss_G_list = []
loss_D_list = []
loss_info_list = []
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
net_G = Generator()
net_D = Discriminator()

adversarial_loss = torch.nn.MSELoss()
continuous_loss = torch.nn.MSELoss() 

lambda_con = 1
lambda_nonintersection = 0.25
lambda_adjacent = 0.25
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
            G_Xy,_ ,_ = net_G(torch.cat((Z,C),-1))
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
            G_Xy, r_adjacent, r_nonintersection = net_G(torch.cat((Z,C),-1))
            _, pred_c = net_D(G_Xy)
            loss_info = lambda_con * continuous_loss(pred_c, C) + lambda_adjacent * r_adjacent+ lambda_nonintersection * r_nonintersection
            loss_info.backward()
            trainer_info.step() 

            if i % 30 == 0:           
                print( "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                % (epoch, num_epochs, i, len(data_iter), loss_D.item(), loss_G.item(), loss_info.item()))
        loss_D_list.append(loss_D.item())
        loss_G_list.append(loss_G.item())
        loss_info_list.append(loss_info.item())
        

# Comment out lines 177 to 187 if you just want to run a trained generator
lr_D, lr_G,lr_info, num_epochs = 0.00001, 0.00001, 0.00001,  4000 #50
train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, lr_info,
      noise_dim, latent_dim)


# plt.figure(figsize=(8,6))
# plt.plot(range(num_epochs), loss_G_list)
# plt.plot(range(num_epochs), loss_D_list)
# plt.plot(range(num_epochs), loss_info_list)
# plt.legend(['Generator', 'Discriminator','Infomation'])
# plt.show()
exit()


# # Saving trained generator: IF YOU JUST WANT TO RUN A TRAINED GENERATOR, COMMENT OUT LINE 192! 
generator_model_path =  cwd + '/FFT_GAN_GENERATOR_20_CP'
# torch.save(net_G.state_dict(),generator_model_path)

Gen = Generator()
Gen.load_state_dict(torch.load(generator_model_path))


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
    fake_data_test,_ ,_ = Gen(torch.cat((Z,C),-1))
    fake_data_test = fake_data_test.detach().numpy()
    x_bspline = np.loadtxt('B_SPLINE_X_COORDINATES_NEW.txt').astype(np.float32)
    #fig, axs = plt.subplots(1, 1, figsize=(11,7.2))#
    color = plt.cm.rainbow(np.linspace(0, 1, test_size))
    for i, c in zip(range(test_size), color):
        axs[latent_c].plot(x_bspline[0:301,0], fake_data_test[i,0:301], c = c, label='C_'+' = {0}'.format(round(latent_code[i],2)))
        axs[latent_c].plot(x_bspline[0:301,0], fake_data_test[i,301:602], c = c)    

fig.tight_layout()
plt.savefig('Fake_Airfoil_latent_test6_original_ReLU_3000')
plt.show()






# Generator and discriminator for FFT_GAN_GENERATOR_30_CP

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.G_net = nn.Sequential(nn.Linear(G_input_dim, 300), nn.ReLU(), 
#                       nn.Linear(300, 400), nn.ReLU(),
#                       nn.Linear(400, 500), nn.ReLU(),
                    #   nn.Linear(500, 500), nn.ReLU(),
                    #   nn.Linear(500, 500), nn.ReLU(),
#                       nn.Linear(500, 400), nn.ReLU(), 
#                       nn.Linear(400, 300), nn.ReLU(), 
#                       nn.Linear(300, G_output_dim))
#     def forward(self, zc):
#         control_y_coords_fake = self.G_net(zc)
#         control_y_coords_fake[:,0] = 0
#         control_y_coords_fake[:,14] = 0
#         control_y_coords_fake[:,15] = 0
#         control_y_coords_fake[:,29] = 0
#         curve_y_coords_upper_fake = (torch.matmul(Variable(FloatTensor(bspline_matrix)),control_y_coords_fake[:,0:15].T)).T
#         curve_y_coords_lower_fake = (torch.matmul(Variable(FloatTensor(bspline_matrix)),control_y_coords_fake[:,15:30].T)).T
#         curve_y_coords_fake = torch.cat((curve_y_coords_upper_fake, curve_y_coords_lower_fake),-1) 
#         cp_adjacent = torch.norm(control_y_coords_fake[:,1:] - control_y_coords_fake[:,:-1], dim = 1)
#         r_adjacent = torch.mean(cp_adjacent)
#         cp_nonintersection = curve_y_coords_fake[:,:15] - control_y_coords_fake[:,15:]
#         r_nonintersection = torch.mean(torch.maximum(torch.zeros(cp_nonintersection.shape), cp_nonintersection))
#         return curve_y_coords_fake, r_adjacent, r_nonintersection

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.D_net = nn.Sequential(nn.Linear(D_input_dim,400), nn.ReLU(), 
#                       nn.Linear(400, 500), nn.ReLU(),
                    #   nn.Linear(400, 500), nn.ReLU(),
                    #   nn.Linear(500, 500), nn.ReLU(),
#                       nn.Linear(500, 400), nn.ReLU(),
#                       nn.Linear(400, 400), nn.ReLU())
#         self.adv_layer = nn.Sequential(nn.Linear(400,1))
#         self.latent_layer = nn.Sequential(nn.Linear(400, latent_dim))
#     def forward(self, curve_y_coords_fake):
#         out = self.D_net(curve_y_coords_fake)
#         validity = self.adv_layer(out)
#         latent_code = self.latent_layer(out)
#         return validity, latent_code

