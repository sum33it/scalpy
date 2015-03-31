import numpy as np
from scipy.integrate import odeint
from scipy import interpolate
from scipy.integrate import quad
from scipy.interpolate import splrep
from scipy.interpolate import splev
from scipy.interpolate import UnivariateSpline
from scipy.misc import derivative
from scipy.optimize import fsolve
import transfer_func as tf
######################################


class LCDM(object):
	"""
	A Class to calculate cosmological observables for concordance LCDM model.
	Equation of state, w = -1

	parameters are:
	Om0 : present day density parameter for total matter (baryonic + dark)
	Ob0 : present day density parameter for baryons (or visible matter)
	ns : spectral index for primordial power spectrum
	h : dimensionless parameter related to Hubble constant H0 = 100*h km s^-1 MPc^-1
	sigma_8 : r.m.s. mass fluctuation on 8h^-1 MPc scale
	"""
	def __init__(self,Om0,h=0.7,Ob0=0.045,ns=0.96,sigma_8 = 0.8):
		self.Om0 = float(Om0)
		self.Ob0 = float(Ob0)
		self.ns = float(ns)
		self.h = float(h)
		self.sigma_8 = float(sigma_8)
	
	def hubz(self,z):
		return np.sqrt(self.Om0*(1.+z)**3. + (1.-self.Om0))

	def invhub(self,z):
		return 1./self.hubz(z)

	def huba(self,a):
		return self.hubz(1./a-1)

	def hubpa(self,a):
		"""
		Derivative of dimensionless hubble constant w.r.t. a
		"""
		return derivative(self.huba,a,dx=1e-6)

	def D_H(self):
		"""
		Hubble distance in units of MPc (David Hogg arxiv: astro-ph/9905116v4)
		"""
		return 2997.98/self.h

	def co_dis_z(self,z1):
		"""
		Line of sight comoving distance as a function of redshift z
		as described in David Hogg paper
		in units of MPc
		"""
		return self.D_H()*quad(self.invhub,0,z1)[0]

	#angular diameter distance in units of MPc
	def ang_dis_z(self,z):
		"""
		Angular diameter distance as function of redshift z
		in units of MPc
		"""
		d_c = self.co_dis_z(z)/(1.+z)
		return d_c

	#luminosity distance in units of MPc
	def lum_dis_z(self,z):
		"""
		Luminosity distance in as function of redshift z
		in units of MPc
		"""
		d_l = self.co_dis_z(z)*(1.+z)
		return d_l

	def time_delay_dis(self,zd,zs):
		"""
		Time delay distance used in strong lensing cosmography
		(See Suyu et. al., Astrophys. J. 766, 70, 2013; arxiv:1208.6010)
		zs = redshift at which the source is placed
		zd = redshift at which the deflector (or lens) is present		
		"""
		D_ds = self.D_H()/(1+zs)*quad(self.invhub,zd,zs)[0]
		return (1+zd)*self.ang_dis_z(zd)*self.ang_dis_z(zs)/D_ds

	def Rth(self,z):
		"""
		CMB Shift parameter (see  )
		"""
		return np.sqrt(self.om_z(0.))*(1+z)*self.ang_dis_z(z)/self.D_H()


	def t_H(self):
		"""
		Hubble time in units of Billion years
		"""
		return 9.78/self.h

	def lookback_time_z(self,z):
		"""
		Lookback time as a function of redshift z
		in units of billion years
		"""
		def integrand(z):
			return self.invhub(z)/(1.+z)
		lt = quad(integrand,0,z)[0]
		return self.t_H()*lt

	def age_by(self):
		"""
		Age of the Universe in units of billion years
		"""
		def integrand(z1):
			return self.invhub(z1)/(1.+z1)
		t0 = quad(integrand,0,np.inf)[0]
		return self.t_H()*t0

	def om_a(self,a):
		"""
		density parameter \Omega_{m} for matter as a function of scale factor a
		"""
		return self.Om0*a**(-3)/self.huba(a)**2.

	def om_z(self,z):
		"""
		density parameter \Omega_{m} for matter as a function of scale factor a
		"""
		return self.om_a(1./(1.+z))
		
	def Om_diag(self,z):
		"""
		Om diagnostic
		See Varun Sahni et. al., Phys. Rev. D. 78, 103502 (2008) for more details
		"""
		x = 1+z
		return (self.hubz(x)**2. - 1)/(x**3. - 1.)


	a11 = np.linspace(0.001,1,1000)

	def deriv(self,y,a):
		return [ y[1], -(3./a + self.hubpa(a)/self.huba(a))*y[1] + 1.5*self.om_a(a)*y[0]/(a**2.)]

	def sol1(self):
		yinit = (0.001,1.)
		return odeint(self.deriv,yinit,self.a11)

	def D_p(self,a):
	        yn11=self.sol1()[:,0]
	        ynn11=UnivariateSpline(self.a11,yn11,k=3,s=0)
	        return ynn11(a)

	def D_plus_a(self,a):
		return self.D_p(a)/self.D_p(1.0)

	def D_plus_z(self,z):
		"""
		Normalized solution for the growing mode as a function of redshift
		"""
		return self.D_plus_a(1./(1.+z))

	def growth_rate_a(self,a):
		d = self.sol1()[:,:]
		d1 = d[:,0]
		dp = d[:,1]
		gr1 = UnivariateSpline(self.a11,self.a11*dp/d1,k=3,s=0)
		return gr1(a)

	def growth_rate_z(self,z):
		"""
		Growth Rate f = D log(Dplus)/ D Log(a) as a function of redshift
		"""
		return self.growth_rate_a(1./(1.+z))

	def fsigma8z(self,z):
		"""
		fsigma_{8} as a function of redshift
		"""
		return self.growth_rate_z(z)*self.D_plus_z(z)*self.sigma_8
	
	# Defining window function

	def Wf(self,k):
		return 3.*(np.sin(k*8.) - (k*8.)*np.cos(k*8.))/(k*8.)**3.

	def integrand_bbks(self,k):
		return k**(self.ns + 2.)*tf.Tbbks(k,self.Om0,self.Ob0,self.h)**2.*(self.Wf(k))**2.

	def integrand_wh(self,k):
		return k**(self.ns + 2.)*tf.Twh(k,self.Om0,self.Ob0,self.h)**2.*(self.Wf(k))**2.

	def A0bbks(self):
		return (2.0*np.pi**2*self.sigma_8**2.)/(quad(self.integrand_bbks,0,np.inf)[0])

	def A0wh(self):
		return (2.0*np.pi**2*self.sigma_8**2.)/(quad(self.integrand_wh,0,np.inf)[0])

	def Pk_bbks(self,k,z):
		"""
		Matter Power Spectra Pk in units if h^{-3}Mpc^{3} as a function of k in units of [h Mpc^{-1}]
		and z;
		Transfer function is taken to be BBKS
		Ref: Bardeen et. al., Astrophys. J., 304, 15 (1986)
		"""
		return self.A0bbks()*k**self.ns*tf.Tbbks(k,self.Om0,self.Ob0,self.h)**2.*self.D_plus_z(z)**2.

	def Pk_wh(self,k,z):
		"""
		Matter Power Spectra Pk in units if h^{-3}Mpc^{3} as a function of k in units of [h Mpc^{-1}]
		and z;
		Transfer function is taken to be Eisenstein & Hu 
		Ref: Eisenstein and Hu, Astrophys. J., 496, 605 (1998)
		"""
		return self.A0wh()*k**self.ns*tf.Twh(k,self.Om0,self.Ob0,self.h)**2.*self.D_plus_z(z)**2.

	def DPk_bbks(self,k,z):
		"""
		Dimensionless Matter Power Spectra Pk  as a function of k in 
		units of [h Mpc^{-1}] and z;
		Transfer function is taken to be BBKS 
		Ref: Bardeen et. al., Astrophys. J., 304, 15 (1986)
		"""
		return k**3.*self.Pk_bbks(k,z)/(2.0*np.pi**2.)

	def DPk_wh(self,k,z):
		"""
		Dimensionless Matter Power Spectra Pk  as a function of k in 
		units of [h Mpc^{-1}] and z;
		Transfer function is taken to be Eisenstein & Hu 
		Ref: Eisenstein and Hu, Astrophys. J., 496, 605 (1998)
		"""
		return k**3.*self.Pk_wh(k,z)/(2.0*np.pi**2.)

class wCDM(LCDM):
	"""
	A Class to calculate cosmological observables for a model with CDM and dark energy
	for which equation of state w is constant but not equal to -1. See hubz function 
	for details.  

	parameters are:
	Om0 : present day density parameter for total matter (baryonic + dark)
	Ob0 : present day density parameter for baryons (or visible matter)
	w : equation of state for "dark energy"
	ns : spectral index for primordial power spectrum
	h : dimensionless parameter related to Hubble constant H0 = 100*h km s^-1 MPc^-1
	sigma_8 : r.m.s. mass fluctuation on 8h^-1 MPc scale
	"""
	def __init__(self,Om0,w,h=0.7,Ob0=0.045,ns=0.96,sigma_8=0.8):
		self.Om0 = float(Om0)
		self.w = float(w)
		self.Ob0 = float(Ob0)
		self.ns = float(ns)
		self.h = float(h)
		self.sigma_8 = float(sigma_8)
	
	def hubz(self,z):
		return np.sqrt(self.Om0*(1.+z)**3. + (1.-self.Om0)*(1.+z)**(3*(1+self.w)))

class w0waCDM(LCDM):
	"""
	A Class to calculate cosmological observables for a model with CDM and dark energy
	for which equation of state w is parametrized as:
	w(a) = w0 + wa*(1-a)
	where 'a' is the scale factor and w0,wa are constants in Taylor expansion of 
	variable equation of state w(a)

	parameters are:
	Om0 : present day density parameter for total matter (baryonic + dark)
	w0 and wa: coefficients of Taylor expansion of equation of state w(a) near a=1
	Ob0 : present day density parameter for baryons (or visible matter)
	ns : spectral index for primordial power spectrum
	h : dimensionless parameter related to Hubble constant H0 = 100*h km s^-1 MPc^-1
	sigma_8 : r.m.s. mass fluctuation on 8h^-1 MPc scale
	"""
	def __init__(self,Om0,w0,wa,h=0.7,Ob0=0.045,ns=0.96,sigma_8=0.8):
		self.Om0 = float(Om0)
		self.w0 = float(w0)
		self.wa = float(wa)
		self.Ob0 = float(Ob0)
		self.ns = float(ns)
		self.h = float(h)
		self.sigma_8 = float(sigma_8)
	
	def hubz(self,z):
		return np.sqrt(self.Om0*(1.+z)**3. + (1.-self.Om0)*(1.+z)**(3*(1+self.w0+self.wa))*np.exp(-3.*self.wa*(z/(1.+z))))

class GCG(LCDM):
	"""
	A Class to calculate cosmological observables for a model with CDM and dark energy
	for which equation of state is parametrized as that by GCG $p= -A/rho^{alpha}$:
	w(z) = -As/(As+(1-As)*(1+z)**(3*(1+alpha)))
	where 'z' is the redshift and As,alpha are model parameters.

	parameters are:
	Om0 : present day density parameter for total matter (baryonic + dark)
	As and alpha: parameters involved in eqn. of state for GCG (ref())
	Ob0 : present day density parameter for baryons (or visible matter)
	ns : spectral index for primordial power spectrum
	h : dimensionless parameter related to Hubble constant H0 = 100*h km s^-1 MPc^-1
	sigma_8 : r.m.s. mass fluctuation on 8h^-1 MPc scale
	"""
	def __init__(self,Om0,As,alpha,h=0.7,Ob0=0.045,ns=0.96,sigma_8=0.8):
		self.Om0 = float(Om0)
		self.As = float(As)
		self.alpha = float(alpha)
		self.Ob0 = float(Ob0)
		self.ns = float(ns)
		self.h = float(h)
		self.sigma_8 = float(sigma_8)
	
	def hubz(self,z):
		return np.sqrt(self.Om0*(1.+z)**3. + (1.-self.Om0)*(self.As+(1-self.As)*(1+z)**(3*(1+self.alpha)))**(1/(1+self.alpha)))

