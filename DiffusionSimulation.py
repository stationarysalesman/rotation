import matplotlib.pyplot as plt
import numpy as np
import MDAnalysis as mda
import pydiffusion
from pydiffusion import plotting
import sys
sys.path.append("/home/tyler/Documents/research/tylermath")
from Physics import calcDiffusionConstants
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-psf', '--psf', type=str)
parser.add_argument('-pdb', '--pdb', type=str)
args = vars(parser.parse_args())

# Estimate diffusion coefficients based on hydronynamic considerations
k = 1.38e-23 # Boltzmann's constant, m^2kgs^(-2)K^(-1)
T = 293.15 # temperature (K)
eta = 1.003e-3 # viscosity of water in Pas * s
r = 4.85e-9 # meters
d = 9.7e-9 # meters
h = 5e-9 # meters
V = np.pi * (r ** 2) * h

# Calculate radius of equivalent sphere and its diffusion coefficient
r0 = np.power(((3 * V) / (4 * np.pi)), 1/3.)
D0 = (k * T) / (6 * np.pi *  eta * V)

# Calculate diffusion coefficients for ellipsoid of revolution via Lakowicz
rho = h / d
D_parallel, D_perp = calcDiffusionConstants(rho)
D_par_abs = D_parallel * D0
D_perp_abs = D_perp * D0
D_vector = [D_par_abs, D_perp_abs, D_perp_abs]
D_vector = np.array(sorted(D_vector, reverse=True))
D_vector = D_vector * 1e-9 # nanoscale
print(D_vector)
# params
dt = 0.005 # nanoseconds 
T = 1000 # nanoseconds
niter = int(T / dt)
trj = pydiffusion.quaternionsimulation.run(D_vector, niter, dt)
R = [pydiffusion.quaternionsimulation.quaternion_to_matrix(q) for q in trj]

disc = mda.Universe(args['pdb'])
pos = disc.atoms.positions.copy()
with mda.Writer('disc.dcd', len(disc.atoms)) as w:
    for r in R:
        disc.atoms.positions = pos.copy()
        disc.atoms.rotate(r)
        w.write(disc)
disc.atoms.positions = pos.copy()

u = mda.Universe(args['psf'], 'disc.dcd')
rm = pydiffusion.rotation.RotationMatrix(u.atoms, disc.atoms, verbose=True).run()
u = pydiffusion.rotation.quaternion_covariance(rm.R, t=5000, verbose=5, n_jobs=4)
u3 = u.T[1:, 1:]
time = np.arange(u3.shape[2]) * dt
cfig = plotting.plot_covariance(u3, time=time)
cfig.set_all_axes(xlabel='time [ns]')


cfig.tight_layout()

print("fitting...")
tmax = 25 
idx = int(tmax / dt)
res = [pydiffusion.rotation.anneal(u3[:,:, :idx], time[:idx], D=[3, 2, 1], eps=1e-5) for _ in range(20)]
fig, ax = plt.subplots(ncols=2)
for r in res:
    ax[0].plot([1, 2, 3], r[0].D, '.', color='#808080')
    
score = [r[1] for r in res]
ax[1].plot(score, '.')
ax[1].set(ylabel=r'$\chi^2$', xlabel='annealing run')

argmin = np.argmin(score)
ax[0].plot([1, 2, 3], res[argmin][0].D, 'o', color='indianred')
ax[0].set(xticks=[1, 2, 3], xticklabels=['D_1', 'D_2', 'D_3'])
ax[1].plot([argmin,], score[argmin], 'o', color='indianred')

fig.tight_layout()

print("D optimal = {}".format(res[argmin][0].D))
print("chi2 = {}".format(res[argmin][1]))
plt.show()
