import os
import subprocess
import numpy as np
import shlex
import time
import signal

class XfoilPythonWrapper():

    # defining run_xfoil method which writes input_file.in and passes that into xfoil.exe 
    def run_xfoil(airfoil_name, Re, M, num_iter, alpha_i, alpha_f, alpha_step):
        
        import signal

        # defining handle function to raise error if xfoil takes too long
        def handler(signum, frame):
            print('Signal handler called with signal', signum)
            raise OSError("Max run time exceeded --> Refining Xfoil inputs")     
        
       

        if os.path.exists( "polar_file.txt"):
            os.remove( "polar_file.txt")

        cwd = os.getcwd()

        # defining call_xfoil function which actually runs xfoil 
        def call_xfoil(alpha_i,alpha_f,alpha_step):
            input_file = open("input_file.in", 'w')
            input_file.write("LOAD {0}.dat\n".format(airfoil_name))
            input_file.write(airfoil_name + '\n')
            input_file.write("PANE\n")
            input_file.write("OPER\n")
            input_file.write("Visc {0}\n".format(Re))
            input_file.write('Mach {0}\n'.format(M))
            input_file.write("PACC\n")
            input_file.write("polar_file.txt\n\n")
            input_file.write("ITER {0}\n".format(num_iter))
            input_file.write("ASeq {0} {1} {2}\n".format(alpha_i, alpha_f,
                                                        alpha_step))
            input_file.write("\n\n")
            input_file.write("quit\n")
            input_file.close()
    

            cmd_string = cwd + '/xfoil.exe < ' + cwd + '/input_file.in'
            print(cmd_string,'COMMAND STRING')
            # instantiate signal object and start timer 
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(30)
            try:
                # Open subprocess to run xfoil.exe 
                process = subprocess.Popen(cmd_string,stdout=subprocess.PIPE, shell=True)# , preexec_fn=os.setsid)
                stdout  = process.communicate()[0]
            except Exception as exc: 
                print(exc)  
            
            # Turn off alarm
            signal.alarm(0)
            


            # load polar file that xfoil created
            polar_data = np.loadtxt("polar_file.txt", skiprows=12)
            return polar_data, process
        
        # os.chdir(cwd)
        total_xfoil_data = (alpha_i-0.1) * np.ones((1,9))
        while True:
            xfoil_data, process = call_xfoil(alpha_i,alpha_f,alpha_step)
            
            time.sleep(2)
            if os.path.exists( "polar_file.txt"):
                os.remove( "polar_file.txt")
            if xfoil_data.shape[0] == 9:
                if alpha_i >= alpha_f:
                    break
                else:
                    alpha_i = alpha_i + 0.5
                    
            elif xfoil_data.shape[0] == 0:
                if alpha_i >= alpha_f:
                    break
                else:
                    alpha_i = alpha_i + 0.5
            elif xfoil_data[:,0][-1] < 8:
                print('below stall region')
                alpha_i = xfoil_data[:,0][-1] + 0.1
                total_xfoil_data = np.vstack((total_xfoil_data,xfoil_data))
            elif xfoil_data[:,0][-1] >= 8:
                print('approaching stall region')
                if xfoil_data[:,0][-1] >= alpha_f:
                    total_xfoil_data = np.vstack((total_xfoil_data,xfoil_data))
                    break
                else:
                    alpha_i = xfoil_data[:,0][-1] + 0.1
                    alpha_step = 0.05
                    total_xfoil_data = np.vstack((total_xfoil_data,xfoil_data))


            
        # Delete "double data": sometimes the Xfoil outputs in 'polar_file.txt' get replicated and we need to delete those
        total_xfoil_data = np.delete(total_xfoil_data,0,0)
        while True:
            for i in range(total_xfoil_data.shape[0]-1):
                if total_xfoil_data[i+1,0]<= total_xfoil_data[i,0]:
                    print('mixup happened')
                    total_xfoil_data[i+1,0] = -20
            delete_index = np.where(total_xfoil_data == -20)[0]
            total_xfoil_data = np.delete(total_xfoil_data, np.where(total_xfoil_data == -20)[0],axis= 0)
            if delete_index.shape[0] == 0:
                break
                
        
        # Compute derivatives 
        def d_dalpha(data):
            n = len(data[:,0])
            dCl_da = np.zeros((n,))
            dCd_da = np.zeros((n,))
            dCm_da = np.zeros((n,))
            for i in range(n):
                if i == (n-1):
                    dCl_da[i] = (data[i,1] - data[i-1,1]) / (data[i,0] - data[i-1,0])
                    dCd_da[i] = (data[i,2] - data[i-1,2]) / (data[i,0] - data[i-1,0])
                    dCm_da[i] = (data[i,4] - data[i-1,4]) / (data[i,0] - data[i-1,0])        
                else:
                    dCl_da[i] = (data[i+1,1] - data[i,1]) / (data[i+1,0] - data[i,0])
                    dCd_da[i] = (data[i+1,2] - data[i,2]) / (data[i+1,0] - data[i,0])
                    dCm_da[i] = (data[i+1,4] - data[i,4]) / (data[i+1,0] - data[i,0])
            
            return dCl_da, dCd_da, dCm_da
        
        dCl_da, dCd_da, dCm_da = d_dalpha(total_xfoil_data)


        process.terminate()
        # os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        print(total_xfoil_data[:,1])
  
        return total_xfoil_data, dCl_da, dCd_da, dCm_da