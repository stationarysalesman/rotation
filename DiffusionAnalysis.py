import matplotlib.pyplot as plt
import numpy as np
import MDAnalysis as mda
import pydiffusion
from pydiffusion import plotting

# params
dt = 0.1 # nanoseconds
T = 87128
u = mda.Universe('D1_fix_combined.psf', 'D1_long_unwrap.dcd')
disc = mda.Universe('D1_fix_combined.pdb')
rm = pydiffusion.rotation.RotationMatrix(u.atoms, disc.atoms, verbose=True).run()
u = pydiffusion.rotation.quaternion_covariance(rm.R, t=87128, verbose=5, n_jobs=-1)
u3 = u.T[1:, 1:]
time = np.arange(u3.shape[2]) * dt
cfig = plotting.plot_covariance(u3, time=time)
cfig.set_all_axes(xlabel='time [ns]')

cfig.tight_layout()
plt.show()
