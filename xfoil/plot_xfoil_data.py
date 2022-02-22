import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm
import numpy as np 


def plot_xfoil_data(M,Re,alpha, text_file):
    
    airfoil_data = np.loadtxt(text_file)
    airfoil_data = airfoil_data.reshape((7, len(M),len(Re), len(alpha)))
    color_vec  = cm.rainbow(np.linspace(0, 1, len(Re)))
    fig, axs = plt.subplots(2,4, figsize=(15, 6) )

    for i in range(2):
        for j in range(4):
            for k in range(10):
                max_value = max(airfoil_data[0,4 * i + j,k,:])
                max_value_index = np.where(airfoil_data[0,4 * i + j,k,:] == max_value)[0][0]
                axs[i,j].plot(airfoil_data[0,4 * i + j,k,0:max_value_index],airfoil_data[2,4 * i + j,k,0:max_value_index], color = color_vec[k],label = 'Re = {:.1e}'.format(Re[k]))
                title_string = 'M = {}'.format(M[4 * i + j])
                axs[i,j].title.set_text(title_string)
                if (i == 1) and (j == 3):
                    handles, labels = axs[i,j].get_legend_handles_labels()
    fig.legend(handles, labels, ncol= 1,loc='center right')
    
    fig2, axs2 = plt.subplots(2,4, figsize=(15, 6) )
    for i in range(2):
        for j in range(4):
            for k in range(10):
                max_value = max(airfoil_data[0,4 * i + j,k,:])
                max_value_index = np.where(airfoil_data[0,4 * i + j,k,:] == max_value)[0][0]
                axs2[i,j].plot(airfoil_data[0,4 * i + j,k,0:max_value_index],airfoil_data[1,4 * i + j,k,0:max_value_index], color = color_vec[k],label = 'Re = {:.1e}'.format(Re[k]))
                title_string = 'M = {}'.format(M[4 * i + j])
                axs2[i,j].title.set_text(title_string)
                if (i == 1) and (j == 3):
                    handles, labels = axs2[i,j].get_legend_handles_labels()
    fig2.legend(handles, labels, ncol= 1,loc='center right')



    plt.show()
