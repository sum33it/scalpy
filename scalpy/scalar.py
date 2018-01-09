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
import hope

#######################################


class scalarpow(object):
	"""
	A Class to calculate cosmological observables for quintessence field
	with power law potential.

	parameters are:
	Ophi_i : initial value (at a = 0.001) of density parameter for scalar field
	li : initial value (at a = 0.001) for the slope	of potential (-V'(phi)/V(phi))
	n : order of power law potential V(phi)=phi**n
	Ob0 : present day density parameter for baryons (or visible matter)
	ns : spectral index for primordial power spectrum
	h : dimensionless parameter related to Hubble constant H0 = 100*h km s^-1 Mpc^-1
	sigma_8 : r.m.s. mass fluctuation on 8h^-1 Mpc scale
	"""
	def __init__(self,Ophi_i,li,n,Orad_i,h=0.7,Ob0=0.045,ns=0.96,sigma_8 = 0.8):
		self.Ophi_i = float(Ophi_i)
		self.li = float(li)
		self.n = float(n)
		self.Orad_i = float(Orad_i)
		self.h = float(h)
		self.Ob0 = float(Ob0)
		self.ns = float(ns)
		self.sigma_8 = float(sigma_8)
	
	n1 = np.linspace(np.log(1./1101),0.0,1000)

	@hope.jit
	def f(self,x,efold):
		g = (self.n-1.)/self.n
		y = np.empty(6)
		y[0] = 3.0*x[0]*(x[0]-2)+x[3]*(3*x[1]*x[0])**0.5*(2-x[0])
		y[1] = 3*x[1]*(1-x[0])*(1-x[1])+x[1]*x[2]
		y[2] = x[2]*(x[2]-1)+3*x[2]*x[1]*(x[0]-1)
		y[3] = -(3*x[0]*x[1])**(0.5)*x[3]**2.*(g-1.0)
		y[4] = x[5]
		y[5] = -(0.5-0.5*x[2]-1.5*x[1]*(x[0]-1))*x[5]+1.5*(1-x[1]-x[2])*x[4]
		return y


	def sol(self):
		inic = [1e-4,self.Ophi_i*1e-9,self.Orad_i,self.li,1./1101,1./1101]
		res = odeint(self.f, inic, self.n1)
		return res


	def Omega_phi_n(self,N):
		"""
		Density parameter for scalar field as a function of log(a)
		where 'a' is the scale factor of the Universe.
		Input:
		  N: log(a)
		Output:
		  Omega_phi(N): density parameter for scalar field as a function of e-folding
		------------------
		Examples:
		1) Calculating Omega_phi for scalar field with power law potential V=phi^n with n=1
		  > import scalpy.scalar as s1
		  > s1.scalarpow(2.,0.2,1.,0.1).Omega_phi_n(-2)

		2) Calculating present day density parameter scalar field with exponential potential

		  > import scalpy.scalar as s1
		  > s1.scalarexp(2.,0.2,0.1).Omega_phi_n(0)
		"""
        	ophi1 = self.sol()[:,1]
        	ophi2=UnivariateSpline(self.n1,ophi1,k=3,s=0)
        	return ophi2(N)

	def Omega_phi_z(self,z):
		"""
		Density parameter for scalar field as a function of redshift z
		Input:
		  z: redshift
		Output:
		  Omega_phi_z(z): density parameter for scalar field as a function of redshift
		------------------
		Examples:
		1) Calculating Omega_phi for scalar field with power law potential V=phi^n with n=1
		  > import scalpy.scalar as s1
		  > s1.scalarpow(2.,0.2,1.,0.1).Omega_phi_z(2)

		2) Calculating present day density parameter scalar field with exponential potential

		  > import scalpy.scalar as s1
		  > s1.scalarexp(2.,0.2,0.1).Omega_phi_z(0)
		"""
		return self.Omega_phi_n(np.log(1./(1.+z)))
	def Omega_r_n(self,N):
		"""
		Density parameter for radiation as a function of log(a)
		Input:
		  N: log(a)
		Output:
		  Omega_r_n(z): density parameter for radiation component as a function of e-folding
		------------------
		Examples:
		1) Calculating Omega_r for radiation component with power law potential V=phi^n with n=1
		  > import scalpy.scalar as s1
		  > s1.scalarpow(2.,0.2,1.,0.1).Omega_r_n(-2)

		2) Calculating present day density parameter of radiation component with scalar field
		  and exponential potential

		  > import scalpy.scalar as s1
		  > s1.scalarexp(2.,0.2,0.1).Omega_r_n(0)
		"""
        	orad1 = self.sol()[:,2]
        	orad2=UnivariateSpline(self.n1,orad1,k=3,s=0)
        	return orad2(N)

	def Omega_r_z(self,z):
		"""
		Density parameter for radiation as a function of redshift z
		Input:
		  z: redshift
		Output:
		  Omega_r_z(z): density parameter for radiation component as a function of redshift
		------------------
		Examples:
		1) Calculating Omega_r for scalar field with power law potential V=phi^n with n=1
		  > import scalpy.scalar as s1
		  > s1.scalarpow(2.,0.2,1.,0.1).Omega_r_z(2)

		2) Calculating present day density parameter Omega_r with scalar field for exponential potential

		  > import scalpy.scalar as s1
		  > s1.scalarexp(2.,0.2,0.1).Omega_r_z(0)
		"""
		return self.Omega_r_n(np.log(1./(1.+z)))

	def Omega_m_n(self,N):
		"""
		Density parameter for matter component as a function of log(a)
		Input:
		  N: log(a)
		Output:
		  Omega_m_n(z): density parameter for matter component as a function of e-folding
		------------------
		Examples:
		1) Calculating Omega_m with scalar field with power law potential V=phi^n: n=1 at log(a) = -2
		  > import scalpy.scalar as s1
		  > s1.scalarpow(2.,0.2,1.,0.1).Omega_m_n(-2)

		2) Calculating present day density parameter Omega_m with scalar field for exponential potential

		  > import scalpy.scalar as s1
		  > s1.scalarexp(2.,0.2,0.1).Omega_m_n(0)
		"""
		om1 = (1.-self.sol()[:,1]-self.sol()[:,2])
		omg = UnivariateSpline(self.n1,om1,k=3,s=0)
		return omg(N)

	def Omega_m_a(self,a):
		return self.Omega_m_n(np.log(a))

	def Omega_m_z(self,z):
		"""
		Density parameter for matter component as a function of redshift z
		Input:
		  z: redshift
		Output:
		  Omega_m_z(z): density parameter for matter component as a function of redshift
		------------------
		Examples:
		1) Calculating Omega_m with scalar field with power law potential V=phi^n: n=1 at z=2
		  > import scalpy.scalar as s1
		  > s1.scalarpow(2.,0.2,1.,0.1).Omega_m_z(2)

		2) Calculating present day density parameter Omega_m with scalar field for exponential potential

		  > import scalpy.scalar as s1
		  > s1.scalarexp(2.,0.2,0.1).Omega_m_z(0)
		"""
		return self.Omega_m_n(np.log(1./(1.+z)))



	def equation_of_state_n(self,N):
		"""
		Equation of state as a function of log(a)
		Input:
		  N: log(a)
		Output:
		  w(N): equation of state for scalar field component as a function of e-folding N= log(a)
		------------------
		Examples:
		1) Calculating w(N) with power law potential V=phi^n with n=1 at e-folding = -2
		  > import scalpy.scalar as s1
		  > s1.scalarpow(2.,0.2,1.,0.1).equation_of_state_n(-2)

		2) Calculating present day equation of state of scalar field with exponential potential

		  > import scalpy.scalar as s1
		  > s1.scalarexp(2.,0.2,0.1).equation_of_state_n(0)
		"""
		eqnst1 = self.sol()[:,0]-1.
		eqnst2=UnivariateSpline(self.n1,eqnst1,k=3,s=0)
		return eqnst2(N)

	def equation_of_state_z(self,z):
		"""
		Equation of state as a function of z
		Input:
		  z: redshift
		Output:
		  w(z): equation of state for scalar field component as a function of redshift
		------------------
		Examples:
		1) Calculating w(z) with power law potential V=phi^n with n=1 at redshift z = 2
		  > import scalpy.scalar as s1
		  > s1.scalarpow(2.,0.2,1.,0.1).equation_of_state_z(2)

		2) Calculating present day equation of state of scalar field with exponential potential

		  > import scalpy.scalar as s1
		  > s1.scalarexp(2.,0.2,0.1).equation_of_state_z(0)
		"""
		return self.equation_of_state_n(np.log(1./(1.+z)))	


	def hubble_normalized_n(self,N):
		"""
		Dimensionless Hubble constant as a function of N = log(a)
		"""
		H=((1-self.sol()[999,1]-self.sol()[999,2])*np.exp(-3*self.n1)/(1-self.sol()[:,1]-self.sol()[:,2]))**(0.5)
		if N > np.log(1./(1+self.z_decoupling())):
			hubb=UnivariateSpline(self.n1,H,k=3,s=0)
			return hubb(N)
		else:
			return np.sqrt(self.Omega_m_n(0)*np.exp(-3*N)+self.Omega_r_n(0)*np.exp(-4*N))


	def hubble_normalized_z(self,z):
		"""
		Dimensionless Hubble parameter as a function of redshift z
		"""
		return self.hubble_normalized_n(np.log(1./(1.+z)))

	def inverse_hubble_normalized_z(self,z):
		"""
		Inverse of the normalized hubble parameter: 1/h(z)
		where h(z) = H(z)/H0
		Input:
		  z: redshift
		Output:
		  1/h(z): inverse of normalized hubble parameter
		"""
		return 1./self.hubble_normalized_z(z)


	def hubble_normalized_a(self,a):
		return self.hubble_normalized_z(1./a-1)


	def hubble_prime_normalized_n(self,N):
		"""
		Derivative of dimensionless hubble constant w.r.t. log(a)
		"""
		return derivative(self.hubble_normalized_n,N,dx=1e-6)

	def D_H(self):
		"""
		Hubble distance in units of Mpc (David Hogg, arxiv: astro-ph/9905116v4)
		"""
		return 2997.98/self.h

	def comoving_distance_n(self,N):
		"""
		Line of sight comoving distance as a function of log(a)
		as described in David Hogg paper
		in units of Mpc
		"""
		H=((1-self.sol()[999,1])*np.exp(-3*self.n1)/(1-self.sol()[:,1]))**(0.5)
		H1=np.exp(-self.n1)/H
		tck3=interpolate.splrep(self.n1,H1,s=0)
		rs=interpolate.splint(N,0,tck3)
		return self.D_H()*rs
	
	def comoving_distance_z(self,z):
		"""
		Line of sight comoving distance as a function of redshift z
		as described in David Hogg paper
		in units of Mpc
		"""
		return self.comoving_distance_n(np.log(1./(1.+z)))

	#angular diameter distance
	def angular_diameter_distance_z(self,z):
		"""
		Angular diameter distance as function of redshift z
		in units of Mpc
		"""
		d_c = self.comoving_distance_z(z)/(1.+z)
		return d_c

	#luminosity distance
	def luminosity_distance_z(self,z):
		"""
		Luminosity distance in as function of redshift z
		in units of Mpc
		"""
		d_l = self.comoving_distance_z(z)*(1.+z)
		return d_l

	def z_from_luminosity_distance(self,distance):
		"""
		This function gives redshift z for a given luminosity 
		distance (Mpc)
		"""
		def dis(z):
			return self.luminosity_distance_z(z) - distance
		return fsolve(dis,0.5)[0]

	def time_delay_distance(self,zd,zs):
		"""
		Time delay distance used in strong lensing cosmography
		(See Suyu et. al., Astrophys. J. 766, 70, 2013; arxiv:1208.6010)
		Input:
		  zd = redshift at which the deflector (or lens) is present
		  zs = redshift at which the source is placed
		Output:
		  Time_delay_distance
		"""
		D_ds = self.D_H()/(1+zs)*quad(self.inverse_hubble_normalized_z,zd,zs)[0]
		return (1+zd)*self.angular_diameter_distance_z(zd)*self.angular_diameter_distance_z(zs)/D_ds

	def z_decoupling(self):
		"""
		redshift of decoupling or reionization
		Input:
		  None
		Output:
		  z_decoupling
		-------------------------
		Example:
		1) Calculating z_decoupling with power law potential V=phi^n with n=1
		  > import scalpy.scalar as s1
		  > s1.scalarpow(2.,0.2,1.,0.1).z_decoupling()

		2) Calculating z_decoupling in scalar field model with exponential potential

		  > import scalpy.scalar as s1
		  > s1.scalarexp(2.,0.2,0.1).z_decoupling()
		"""
		g1 = 0.0783*(self.Ob0*self.h**2)**(-0.238)/(1+39.5*(self.Ob0*self.h**2)**(0.763))
		g2 = 0.560/(1+21.1*(self.Ob0*self.h**2)**1.81)
		Om0 = 1-self.Omega_phi_z(0)-self.Omega_r_z(0)
		return 1048*(1.+0.00124*(self.Ob0*self.h**2)**(-0.738))*(1+g1*(Om0*self.h**2)**g2)

	def z_drag_epoch(self):
		"""
		Gives the redshit of baryon drag epoch
		Eisenstein and Hu; arxiv:astro-ph/9709112
		Input:
		  None
		Output:
		  z_drag_epoch
		"""
		b1 = 0.313*(self.Omega_m_z(0)*self.h**2)**(-0.419)*(1+0.607*(self.Omega_m_z(0)*self.h**2)**0.674)
		b2 = 0.238*(self.Omega_m_z(0)*self.h**2)**0.223
		return 1291.*(self.Omega_m_z(0)*self.h**2)**0.251*(1+b1*(self.Omega_m_z(0)*self.h**2)**b2)/(1+0.659*(self.Omega_m_z(0)*self.h**2)**0.828)

	def sound_horizon1(self,z):
		R1 = 31500*(self.Ob0*self.h**2)*(2.728/2.7)**(-4)*1/(1.+z)
		def integrand1(a):
			return 1/(np.sqrt(3*(1+R1))*self.hubble_normalized_a(a)*a**2)
		return self.D_H()*quad(integrand1,0,1./(1+z))[0]

	def sound_horizon(self,z):
		if z<self.z_decoupling():
			raise ValueError('Redshift should be greater than decoupling redshift')
		om0,or0 = self.Omega_m_n(0), self.Omega_r_n(0)
		R1 = 31500*(self.Ob0*self.h**2)*(2.728/2.7)**(-4)*1/(1.+z)
		def integrand1(a):
			h1 = np.sqrt(om0*a**(-3)+or0*a**(-4))
			return 1/(np.sqrt(3*(1+R1))*h1*a**2)
		return self.D_H()*quad(integrand1,0,1./(1+z))[0]	


	# CMB Shift Parameter
	def cmb_shift_parameter(self):
		"""
		CMB Shift parameter
		"""
		return np.sqrt(self.Omega_m_z(0.))*(1+self.z_decoupling())*self.angular_diameter_distance_z(self.z_decoupling())/self.D_H()

	def acoustic_length(self):
		"""
		It calculates acoustis length scale at decoupling redshift
		"""
		z_star = self.z_decoupling()
		return np.pi*self.angular_diameter_distance_z(z_star)*(1+z_star)/self.sound_horizon(z_star)

	def t_H(self):
		"""
		Hubble time in units of Billion years
		"""
		return 9.78/self.h

	def lookback_time_n(self,N):
		"""
		Lookback time as a function of log(a)
		in units of billion years
		"""
		H=((1-self.sol()[999,1])*np.exp(-3*self.n1)/(1-self.sol()[:,1]))**(0.5)
		tck3=interpolate.splrep(self.n1,1/H,s=0)
		lt=interpolate.splint(N,0,tck3)
		return self.t_H()*lt

	def lookback_time_z(self,z):
		"""
		Lookback time as a function of redshift z
		in units of billion years
		"""
		return self.lookback_time_n(np.log(1./(1.+z)))

	def D_p(self,N):
	        yn11=self.sol()[:,4]
	        ynn11=UnivariateSpline(self.n1,yn11,k=3,s=0)
	        return ynn11(N)

	#Normalized solution for the growing mode
	def D_plus_n(self,N):
		"""
		Normalized solution for the growing mode as a function of efolding 
		"""
		return self.D_p(N)/self.D_p(0.)

	def D_plus_z(self,z):
		"""
		Normalized solution for the growing mode as a function of redshift
		"""
		return self.D_plus_n(np.log(1./(1.+z)))

	def growth_rate_n(self,N):
		d = self.sol()[:,:]
		d1 = d[:,4]
		dp = d[:,5]
		gr1 = UnivariateSpline(self.n1,dp/d1,k=3,s=0)
		return gr1(N)

	def growth_rate_z(self,z):
		"""
		Growth Rate f = D log(Dplus)/ D Log(a) as a function of redshift
		"""
		return self.growth_rate_n(np.log(1./(1.+z)))

	def fsigma8z(self,z):
		"""
		fsigma_{8} as a function of redshift
		"""
		return self.growth_rate_z(z)*self.D_plus_z(z)*self.sigma_8
		# Defining window function

	def Wf(self,k):
		return 3.*(np.sin(k*8.) - (k*8.)*np.cos(k*8.))/(k*8.)**3.

	def integrand_bbks(self,k):
		return k**(self.ns + 2.)*tf.Tbbks(k,self.Omega_m_z(0),self.Ob0,self.h)**2.*(self.Wf(k))**2.

	def integrand_wh(self,k):
		return k**(self.ns + 2.)*tf.Twh(k,self.Omega_m_z(0),self.Ob0,self.h)**2.*(self.Wf(k))**2.

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
		return self.A0bbks()*k**self.ns*tf.Tbbks(k,self.Omega_m_z(0),self.Ob0,self.h)**2.*self.D_plus_z(z)**2.

	def Pk_wh(self,k,z):
		"""
		Matter Power Spectra Pk in units if h^{-3} Mpc^{3} as a function of k in units of [h Mpc^{-1}]
		and z;
		Transfer function is taken to be Eisenstein & Hu  
		(ref(Eisenstein and Hu, Astrophys. J., 496, 605 (1998)))
		"""
		return self.A0wh()*k**self.ns*tf.Twh(k,self.Omega_m_z(0),self.Ob0,self.h)**2.*self.D_plus_z(z)**2.

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
		(ref(Eisenstein and Hu, Astrophys. J., 496, 605 (1998)))
		"""
		return k**3.*self.Pk_wh(k,z)/(2.0*np.pi**2.)


class scalarexp(scalarpow):
	"""
	A Class to calculate cosmological observables for quintessence field
	with exponential potential.

	parameters are:
	Ophi_i : initial value (at a = 0.001) of density parameter for scalar field
	li : initial value (at a = 0.001) for the slope	of potential (-V'(phi)/V(phi))
	Ob0 : present day density parameter for baryons (or visible matter)
	ns : spectral index for primordial power spectrum
	h : dimensionless parameter related to Hubble constant H0 = 100*h km s^-1 Mpc^-1
	sigma_8 : r.m.s. mass fluctuation on 8h^-1 Mpc scale
	"""
	def __init__(self,Ophi_i,li,Orad_i,h=0.7,Ob0=0.045,ns=0.96,sigma_8 = 0.8):
		self.Ophi_i = float(Ophi_i)
		self.li = float(li)
		self.Orad_i = float(Orad_i)
		self.h = float(h)
		self.Ob0 = float(Ob0)
		self.ns = float(ns)
		self.sigma_8 = float(sigma_8)
	
	n1 = np.linspace(np.log(1./1101),0.0,1000)

	@hope.jit
	def f(self,x,efold):
		g = 1.
		y = np.empty(6)
		y[0] = 3.0*x[0]*(x[0]-2)+x[3]*(3*x[1]*x[0])**0.5*(2-x[0])
		y[1] = 3*x[1]*(1-x[0])*(1-x[1])+x[1]*x[2]
		y[2] = x[2]*(x[2]-1)+3*x[2]*x[1]*(x[0]-1)
		y[3] = -(3*x[0]*x[1])**(0.5)*x[3]**2.*(g-1.0)
		y[4] = x[5]
		y[5] = -(0.5-0.5*x[2]-1.5*x[1]*(x[0]-1))*x[5]+1.5*(1-x[1]-x[2])*x[4]
		return y

class tachyonpow(scalarpow):
	"""
	A Class to calculate cosmological observables for tachyon field
	with power law potential.

	parameters are:
	Ophi_i : initial value (at a = 0.001) of density parameter for tachyon field
	li : initial value (at a = 0.001) for the slope	of potential (-V'(phi)/V(phi))
	n : order of power law potential V(phi)=phi**n
	Ob0 : present day density parameter for baryons (or visible matter)
	ns : spectral index for primordial power spectrum
	h : dimensionless parameter related to Hubble constant H0 = 100*h km s^-1 Mpc^-1
	sigma_8 : r.m.s. mass fluctuation on 8h^-1 Mpc scale
	"""
	def __init__(self,Ophi_i,li,n,Orad_i,h=0.7,Ob0=0.045,ns=0.96,sigma_8 = 0.8):
		self.Ophi_i = float(Ophi_i)
		self.li = float(li)
		self.n = float(n)
		self.Orad_i = float(Orad_i)
		self.Ob0 = float(Ob0)
		self.ns = float(ns)
		self.h = float(h)
		self.sigma_8 = float(sigma_8)
#	@hope.jit
	def f(self,x,efold):
		g = (self.n-1.)/self.n
		y = np.empty(6)
		y[0] = -6.*x[0]*(1.-x[0])+2.*x[3]*((1.-x[0])**(5./4.))*((3.*x[0]*x[1])**0.5)
		y[1] = 3.0*x[1]*((1.-x[0])*(1.-x[1]) + x[2])
		y[2] = -x[2]*(3.*x[1]*(1.-x[0]) + 1. - x[2])
		y[3] = -((3*x[0]*x[1])**(0.5))*(x[3]**2)*((1.-x[0])**(0.25))*(g-1.5)
		y[4] = x[5]
		y[5] = -(0.5-0.5*x[2]-1.5*x[1]*(x[0]-1))*x[5]+1.5*(1-x[1]-x[2])*x[4]
		return y


class tachyonexp(tachyonpow):
	"""
	A Class to calculate cosmological observables for tachyon field
	with exponential potential.

	parameters are:
	Ophi_i : initial value (at a = 0.001) of density parameter for tachyon field
	li : initial value (at a = 0.001) for the slope	of potential (-V'(phi)/V(phi))
	Ob0 : present day density parameter for baryons (or visible matter)
	ns : spectral index for primordial power spectrum
	h : dimensionless parameter related to Hubble constant H0 = 100*h km s^-1 Mpc^-1
	sigma_8 : r.m.s. mass fluctuation on 8h^-1 Mpc scale
	"""
	def __init__(self,Ophi_i,li,Orad_i,h=0.7,Ob0=0.045,ns=0.96,sigma_8 = 0.8):
		self.Ophi_i = float(Ophi_i)
		self.li = float(li)
		self.Orad_i = float(Orad_i)
		self.Ob0 = float(Ob0)
		self.ns = float(ns)
		self.h = float(h)
		self.sigma_8 = float(sigma_8)
#	@hope.jit
	def f(self,x,efold):
		g = 1.
		y = np.empty(6)
		y[0] = -6.*x[0]*(1.-x[0])+2.*x[3]*((1.-x[0])**(5./4.))*((3.*x[0]*x[1])**0.5)
		y[1] = 3.0*x[1]*((1.-x[0])*(1.-x[1]) + x[2])
		y[2] = -x[2]*(3.*x[1]*(1.-x[0]) + 1. - x[2])
		y[3] = -((3*x[0]*x[1])**(0.5))*(x[3]**2)*((1.-x[0])**(0.25))*(g-1.5)
		y[4] = x[5]
		y[5] = -(0.5-0.5*x[2]-1.5*x[1]*(x[0]-1))*x[5]+1.5*(1-x[1]-x[2])*x[4]
		return y


class galileonpow(scalarpow):
	"""
	A Class to calculate cosmological observables for Galileon field
	with exponential potential.

	parameters are:
	Ophi_i : initial value (at a = 0.001) of density parameter for Galileon field
	epsilon_i : initial value (at a = 0.001) for epsilon (see readme.rst)
	li : initial value (at a = 0.001) for the slope	of potential (-V'(phi)/V(phi))
	Ob0 : present day density parameter for baryons (or visible matter)
	ns : spectral index for primordial power spectrum
	h : dimensionless parameter related to Hubble constant H0 = 100*h km s^-1 Mpc^-1
	sigma_8 : r.m.s. mass fluctuation on 8h^-1 Mpc scale
	"""
	def __init__(self,Ophi_i,epsilon_i,li,n,Orad_i,h=0.7,Ob0=0.045,ns=0.96,sigma_8 = 0.8):
		self.Ophi_i = float(Ophi_i)
		self.epsilon_i = float(epsilon_i)
		self.li = float(li)
		self.n = float(n)
		self.Orad_i = float(Orad_i)
		self.Ob0 = float(Ob0)
		self.ns = float(ns)
		self.h = float(h)
		self.sigma_8 = float(sigma_8)
	@hope.jit
	def gal(self,x,efold):
		g = (self.n-1.)/self.n
		T1 = 4.+4.*x[2]+x[0]**2.*x[2]**2.
		T2 = (2.*(1+x[2])*(-3.+3*x[1]**2.-x[3])-3*x[0]**2*(2+4*x[2]+x[2]**2)+np.sqrt(6.)*x[0]*x[2]*x[1]**2*x[4])/T1
		T3 = (3*x[0]**3*x[2]-x[0]*(12.+x[2]*(3+3*x[1]**2-x[3]))+2.*np.sqrt(6.)*x[1]**2*x[4])/(x[0]*T1)
		w_pi = (-12.*(1.+x[2])*x[1]**2 + 3.*(4.+8.*x[2]+x[2]**2)*x[0]**2 - (24.**0.5)*x[0]*x[2]*x[4]*x[1]**2-x[2]**2*x[0]**2*x[3])/(3.*(4.+4.*x[2]+(x[2]**2)*(x[0]**2))*(x[1]**2+(1.+x[2])*x[0]**2))
		y = np.empty(7)
		y[0] = x[0]*(T3-T2)
		y[1] = -x[1]*(np.sqrt(3./2)*x[4]*x[0]+T2)
		y[2] = x[2]*(T3+T2)
		y[3] = -2.*x[3]*(2.+T2)
		y[4] = np.sqrt(6.)*x[0]*x[4]**2*(1-g)
		y[5] = x[6]
		y[6] = -(0.5-0.5*x[3]-1.5*(x[0]**2*(1+x[2])+x[1]**2)*w_pi)*x[6]+1.5*(1-x[0]**2*(1+x[2])-x[1]**2-x[3])*x[5]
		return y
		#return [(3.*x[0]**3*(2.+5.*x[2] + x[2]**2)-3.*x[0]*(2.-x[2]+(x[1]**2)*(2.+3.*x[2]))+(24.**0.5)*(x[1]**2)*x[4]-(6.**0.5)*(x[0]**2)*(x[1]**2)*x[2]*x[4]+(2.+3.*x[2])*x[0]*x[3])/(4.+4.*x[2]+(x[0]**2)*(x[2]**2)),-(x[1]*(12.*((x[1]**2)-1.)*(1.+x[2])-6.*(x[0]**2)*(2.+4.*x[2]+(x[2]**2))+(6.**0.5)*(x[0]**3)*(x[2]**2)*x[4]+(24.**0.5)*x[0]*(2.+(2.+x[1]**2)*x[2])*x[4] - 4*x[3]*(1+x[2])))/(2*(4.+4.*x[2]+(x[0]**2)*(x[2]**2))),-(x[2]*(-3.*x[0]*(-3.+x[1]**2)*(2.+x[2])+3.*(x[0]**3)*(2.+3.*x[2]+x[2]**2)-(24.**0.5)*(x[1]**2)*x[4]-(6.**0.5)*(x[0]**2)*(x[1]**2)*x[2]*x[4] - x[0]*x[3]*(2.+x[2])))/(x[0]*(4.+4.*x[2]+(x[0]**2)*(x[2]**2))),orad_equation,(6.**0.5)*x[0]*(x[3]**2)*(1.-g)]

	def sol(self):
		xi = 1e-7
		yi = np.sqrt(self.Ophi_i*1e-9 - xi**2*(1+self.epsilon_i))
		inic = [xi,yi,self.epsilon_i,self.Orad_i,self.li,1./1101,1./1101]
		res = odeint(self.gal, inic, self.n1)
		return res

	def equation_of_state_n(self,N):
		"""
		Equation of state as a function of log(a)
		"""
		eqnst1 = (-12.*(1.+self.sol()[:,2])*self.sol()[:,1]**2 + 3.*(4.+8.*self.sol()[:,2]+self.sol()[:,2]**2)*self.sol()[:,0]**2 - (24.**0.5)*self.sol()[:,0]*self.sol()[:,2]*self.sol()[:,4]*self.sol()[:,1]**2-self.sol()[:,2]**2*self.sol()[:,0]**2*self.sol()[:,3])/(3.*(4.+4.*self.sol()[:,2]+(self.sol()[:,2]**2)*(self.sol()[:,0]**2))*(self.sol()[:,1]**2+(1.+self.sol()[:,2])*self.sol()[:,0]**2))
		eqnst2=UnivariateSpline(self.n1,eqnst1,k=3,s=0)
		return eqnst2(N)

	def Omega_phi_n(self,N):
		"""
		Density parameter for scalar field as a function of log(a)
		where 'a' is the scale factor of the Universe.
		Input:
		  N: log(a)
		Output:
		  Omega_phi(N): density parameter for scalar field as a function of e-folding
		------------------
		Examples:
		1) Calculating Omega_phi for scalar field with power law potential V=phi^n with n=1
		  > import scalpy.scalar as s1
		  > s1.scalarpow(2.,0.2,1.,0.1).Omega_phi_n(-2)

		2) Calculating present day density parameter scalar field with exponential potential

		  > import scalpy.scalar as s1
		  > s1.scalarexp(2.,0.2,0.1).Omega_phi_n(0)
		"""
        	Op = self.sol()[:,1]**2 + (1.+self.sol()[:,2])*self.sol()[:,0]**2
		ophi2=UnivariateSpline(self.n1,Op,k=3,s=0)
		return ophi2(N)

	def Omega_m_n(self,N):
		"""
		Density parameter for matter component as a function of log(a)
		Input:
		  N: log(a)
		Output:
		  Omega_m_n(z): density parameter for matter component as a function of e-folding
		------------------
		Examples:
		1) Calculating Omega_m with scalar field with power law potential V=phi^n: n=1 at log(a) = -2
		  > import scalpy.scalar as s1
		  > s1.scalarpow(2.,0.2,1.,0.1).Omega_m_n(-2)

		2) Calculating present day density parameter Omega_m with scalar field for exponential potential

		  > import scalpy.scalar as s1
		  > s1.scalarexp(2.,0.2,0.1).Omega_m_n(0)
		"""
		om1 = 1.-self.sol()[:,0]**2.*(1+self.sol()[:,2]) - self.sol()[:,1]**2. - self.sol()[:,3]
		omg = UnivariateSpline(self.n1,om1,k=3,s=0)
		return omg(N)

	def Omega_r_n(self,N):
		"""
		Density parameter for radiation as a function of log(a)
		Input:
		  N: log(a)
		Output:
		  Omega_r_n(z): density parameter for radiation component as a function of e-folding
		------------------
		Examples:
		1) Calculating Omega_r for radiation component with power law potential V=phi^n with n=1
		  > import scalpy.scalar as s1
		  > s1.scalarpow(2.,0.2,1.,0.1).Omega_r_n(-2)

		2) Calculating present day density parameter of radiation component with scalar field
		  and exponential potential

		  > import scalpy.scalar as s1
		  > s1.scalarexp(2.,0.2,0.1).Omega_r_n(0)
		"""
        	orad1 = self.sol()[:,3]
        	orad2=UnivariateSpline(self.n1,orad1,k=3,s=0)
        	return orad2(N)

	def hubble_normalized_n(self,N):
		"""
		Dimensionless Hubble constant as a function of N = log(a)
		"""
		H=((1.-self.sol()[999,0]**2.*(1+self.sol()[999,2]) - self.sol()[999,1]**2- self.sol()[999,3])*np.exp(-3*self.n1)/((1.-self.sol()[:,0]**2.*(1+self.sol()[:,2]) - self.sol()[:,1]**2.- self.sol()[:,3])))**(0.5)
		if N > np.log(1./(1+self.z_decoupling())):
			hubb=UnivariateSpline(self.n1,H,k=3,s=0)
			return hubb(N)
		else:
			return np.sqrt(self.Omega_m_n(0)*np.exp(-3*N)+self.Omega_r_n(0)*np.exp(-4*N))


	def comoving_distance_n(self,N):
		"""
		Line of sight comoving distance as a function of log(a)
		as described in David Hogg paper
		in units of Mpc
		"""
		H=((1.-self.sol()[999,0]**2.*(1+self.sol()[999,2]) - self.sol()[999,1]**2. - self.sol()[999,3])*np.exp(-3*self.n1)/((1.-self.sol()[:,0]**2.*(1+self.sol()[:,2]) - self.sol()[:,1]**2. - self.sol()[:,3])))**(0.5)
		H1=np.exp(-self.n1)/H
		tck3=interpolate.splrep(self.n1,H1,s=0)
		rs=interpolate.splint(N,0,tck3)
		return self.D_H()*rs

	def D_p(self,N):
	        yn11=self.sol()[:,5]
	        ynn11=UnivariateSpline(self.n1,yn11,k=3,s=0)
	        return ynn11(N)

	#Normalized solution for the growing mode
	def D_plus_n(self,N):
		"""
		Normalized solution for the growing mode as a function of efolding 
		"""
		return self.D_p(N)/self.D_p(0.)

	def D_plus_z(self,z):
		"""
		Normalized solution for the growing mode as a function of redshift
		"""
		return self.D_plus_n(np.log(1./(1.+z)))

	def growth_rate_n(self,N):
		d = self.sol()[:,:]
		d1 = d[:,5]
		dp = d[:,6]
		gr1 = UnivariateSpline(self.n1,dp/d1,k=3,s=0)
		return gr1(N)


class galileonexp(galileonpow):
	"""
	A Class to calculate cosmological observables for Galileon field
	with exponential potential.

	parameters are:
	Ophi_i : initial value (at a = 0.001) of density parameter for Galileon field
	epsilon_i : initial value (at a = 0.001) for epsilon (see readme.rst)
	li : initial value (at a = 0.001) for the slope	of potential (-V'(phi)/V(phi))
	Ob0 : present day density parameter for baryons (or visible matter)
	ns : spectral index for primordial power spectrum
	h : dimensionless parameter related to Hubble constant H0 = 100*h km s^-1 Mpc^-1
	sigma_8 : r.m.s. mass fluctuation on 8h^-1 Mpc scale
	"""
	def __init__(self,Ophi_i,epsilon_i,li,Orad_i,h=0.7,Ob0=0.045,ns=0.96,sigma_8 = 0.8):
		self.Ophi_i = float(Ophi_i)
		self.epsilon_i = float(epsilon_i)
		self.li = float(li)
		self.Orad_i = float(Orad_i)
		self.Ob0 = float(Ob0)
		self.ns = float(ns)
		self.h = float(h)
		self.sigma_8 = float(sigma_8)

	@hope.jit
	def gal(self,x,efold):
		g = 1.0
		T1 = 4.+4.*x[2]+x[0]**2.*x[2]**2.
		T2 = (2.*(1+x[2])*(-3.+3*x[1]**2.-x[3])-3*x[0]**2*(2+4*x[2]+x[2]**2)+np.sqrt(6.)*x[0]*x[2]*x[1]**2*x[4])/T1
		T3 = (3*x[0]**3*x[2]-x[0]*(12.+x[2]*(3+3*x[1]**2-x[3]))+2.*np.sqrt(6.)*x[1]**2*x[4])/(x[0]*T1)

		w_pi=(-12.*(1.+x[2])*x[1]**2 + 3.*(4.+8.*x[2]+x[2]**2)*x[0]**2 - (24.**0.5)*x[0]*x[2]*x[4]*x[1]**2-x[2]**2*x[0]**2*x[3])/(3.*(4.+4.*x[2]+(x[2]**2)*(x[0]**2))*(x[1]**2+(1.+x[2])*x[0]**2))
		y = np.empty(7)
		y[0] = x[0]*(T3-T2)
		y[1] = -x[1]*(np.sqrt(3./2)*x[4]*x[0]+T2)
		y[2] = x[2]*(T3+T2)
		y[3] = -2.*x[3]*(2.+T2)
		y[4] = np.sqrt(6.)*x[0]*x[4]**2*(1-g)
		y[5] = x[6]
		y[6] = -(0.5-0.5*x[3]-1.5*(x[0]**2*(1+x[2])+x[1]**2)*w_pi)*x[6]+1.5*(1-x[0]**2*(1+x[2])-x[1]**2-x[3])*x[5]
		return y

