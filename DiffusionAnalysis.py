import matplotlib.pyplot as plt
import numpy as np
import MDAnalysis as mda
import pydiffusion
from pydiffusion import plotting
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-psf', '--psf', type=str)
parser.add_argument('-pdb', '--pdb', type=str)
parser.add_argument('-dcd', '--dcd', type=str)
args = vars(parser.parse_args())

# params
dt = 0.05 # nanoseconds
T = 155 
u = mda.Universe(args['psf'], args['dcd'])
disc = mda.Universe(args['pdb'])
rm = pydiffusion.rotation.RotationMatrix(u.atoms, disc.atoms, verbose=True).run()
u = pydiffusion.rotation.quaternion_covariance(rm.R, t=155, verbose=5, n_jobs=-1)
u3 = u.T[1:, 1:]
time = np.arange(u3.shape[2]) * dt
cfig = plotting.plot_covariance(u3, time=time)
cfig.set_all_axes(xlabel='time [ns]')

cfig.tight_layout()
plt.show()
