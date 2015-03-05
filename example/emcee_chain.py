from scalpy.fluids import *
import emcee
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as pl
import triangle
from matplotlib.ticker import MaxNLocator
from emcee.utils import MPIPool

#####################################################
# JLA data
#####################################################
dataSN = np.loadtxt('jla_mub.txt')
zsn = dataSN[:,0]
mu_b = dataSN[:,1]

covdata = np.loadtxt('jla_mub_covmatrix.dat')
covarray = covdata[1:len(covdata)]

covmat = covarray.reshape(31,31)

def chijla(Om0,w0,wa):
	ld = np.vectorize(w0waCDM(Om0,w0,wa).lum_dis_z)
	x = 5.0*np.log10(ld(zsn))+25.-mu_b
	covinv = np.linalg.inv(covmat)
	return np.dot(x,np.dot(covinv,x))

######################################################
# fsigma8 data
######################################################
datafs8 = np.loadtxt('fs8.dat')
zfs8 = datafs8[:,0]
fs8 = datafs8[:,1]
sgm_fs8 = datafs8[:,2]

def chifs8(Om0,w0,wa):
	fs8_th = w0waCDM(Om0,w0,wa).fsigma8z(zfs8)
	return np.sum((fs8-fs8_th)**2./sgm_fs8**2)

######################################################
# BAO data
######################################################
covinv = [[0.48435, -0.101383, -0.164945, -0.0305703, -0.097874, -0.106738], [-0.101383, 3.2882, -2.45497, -0.0787898, -0.252254, -0.2751], [-0.164945, -2.454987, 9.55916, -0.128187, -0.410404, -0.447574], [-0.0305703, -0.0787898, -0.128187, 2.78728, -2.75632, 1.16437], [-0.097874, -0.252254, -0.410404, -2.75632, 14.9245, -7.32441], [-0.106738, -0.2751, -0.447574, 1.16437, -7.32441, 14.5022]]

def chibao(Om0,w0,wa):
	cpl = w0waCDM(Om0,w0,wa)
	Dh = cpl.D_H()
	invdh = 1/Dh
	
	xx = [cpl.co_dis_z(1091.)*invdh/((cpl.co_dis_z(0.106)*invdh)**2*0.106/(cpl.hubz(-0.106)))**(1./3) - 30.95, cpl.co_dis_z(1091.)*invdh/((cpl.co_dis_z(0.2)*invdh)**2*0.2/(cpl.hubz(0.2)))**(1./3) - 17.55, cpl.co_dis_z(1091.)*invdh/((cpl.co_dis_z(0.35)*invdh)**2*0.35/(cpl.hubz(0.35)))**(1./3) - 10.11, cpl.co_dis_z(1091.)*invdh/((cpl.co_dis_z(0.44)*invdh)**2*0.44/(cpl.hubz(0.44)))**(1./3) - 8.44, cpl.co_dis_z(1091.)*invdh/((cpl.co_dis_z(0.6)*invdh)**2*0.6/(cpl.hubz(0.6)))**(1./3) - 6.69, cpl.co_dis_z(1091.)*invdh/((cpl.co_dis_z(0.73)*invdh)**2*0.73/(cpl.hubz(0.73)))**(1./3) - 5.45]
	chibao1 = np.dot(xx,np.dot(covinv,xx))
	return chibao1

######################################################
# CMB shift parameter
#####################################################
def Rth(z,Om0,w0,wa):
	cpl = w0waCDM(Om0,w0,wa)
	Dh = cpl.D_H()
	return np.sqrt(cpl.om_z(0.))*(1+z)*cpl.ang_dis_z(z)/Dh

def chishift(Om0,w0,wa):
	return (1.7499-Rth(1090.43,Om0,w0,wa))**2/(0.0088**2)
#####################################################

def chi2(Om0,w0,wa):
	return chibao(Om0,w0,wa)+chijla(Om0,w0,wa)+chifs8(Om0,w0,wa)+chishift(Om0,w0,wa)


def lik(Om0,w0,wa):
	return np.exp(-chi2(Om0,w0,wa)/2.)

def lnprior(Om0,w0,wa):
	if 0.1 < Om0 < 0.9 and -1.6 < w0 < -0.4 and -4. < wa < 4.:
		return 0.0
	return -np.inf

def lnlike(Om0,w0,wa):
	return np.log(lik(Om0,w0,wa))

def lnprob(Om0,w0,wa):
	lp = lnprior(Om0,w0,wa)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(Om0,w0,wa)

def ff(theta):
	Om0,w0,wa=theta
	return -2.*lnprob(Om0,w0,wa)

def lnp(theta):
	Om0,w0,wa=theta
	return lnprob(Om0,w0,wa)


# Find the maximum likelihood value.

result = opt.minimize(ff, [0.3,-1.0, 0.])
Om0_ml,w0_ml,wa_ml = result['x']
print("""Maximum likelihood result:
    Om0 = {0} (truth: {1})
    w0 = {2} (truth: {3})
    wa = {4} (truth: {5})
""".format(Om0_ml, 0.3, w0_ml, -1.0, wa_ml, 0.0))

# Set up the sampler.
ndim, nwalkers = 3, 200
pos = [result['x'] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp)
#print pos
# Clear and run the production chain.
print("Running MCMC...")
sampler.run_mcmc(pos, 1000)
print("Done.")

pl.clf()
fig, axes = pl.subplots(3, 1, sharex=True, figsize=(8, 9))
axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].axhline(Om0_ml, color="#888888", lw=2)
axes[0].set_ylabel("$\Omega_{m0}$")

axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].axhline(w0_ml, color="#888888", lw=2)
axes[1].set_ylabel("$w_0$")

axes[2].plot(sampler.chain[:, :, 2].T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].axhline(wa_ml, color="#888888", lw=2)
axes[2].set_ylabel("$w_a$")


fig.tight_layout(h_pad=0.0)
fig.savefig("line-time-bao_jla_fs8_shift.png")

# Make the triangle plot.
burnin = 100
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

fig = triangle.corner(samples, labels=["$\Omega_{m0}$", "$w_0$",  "$w_a$"])
fig.savefig("line-triangle-bao_jla_fs8_shift.png")

# Writing samples to a file
length=len(samples[:,0])
test = open("cpl_bao_jla_fs8_shift.dat","w")
for i in range(length):
	print i, samples[:,0][i], samples[:,1][i], samples[:,2][i]
	test.write('%s' % (samples[:,0][i],))
	test.write('%s' % (" "))
	test.write('%s' % (samples[:,1][i],))
	test.write('%s' % (" "))
	test.write('%s\n' % (samples[:,2][i],))

# write maximizing results in a file
test1 = open("ctp_bao_jla_fs8_shift","w")
test1.write('%s' % (Om0_ml),)
test1.write('%s' % (" "))
test1.write('%s' % (w0_ml),)
test1.write('%s' % (" "))
test1.write('%s\n' % (wa_ml),)
