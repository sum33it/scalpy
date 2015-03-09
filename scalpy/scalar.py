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
	def __init__(self,Ophi_i,li,n,h=0.7,Ob0=0.045,ns=0.96,sigma_8 = 0.8):
		self.Ophi_i = float(Ophi_i)
		self.li = float(li)
		self.n = float(n)
		self.Ob0 = float(Ob0)
		self.ns = float(ns)
		self.h = float(h)
		self.sigma_8 = float(sigma_8)

	n1 = np.linspace(np.log(1./1101),0.0,1000)

	def f(self,x,efold):
		g = (self.n-1.)/self.n
		return [-3.0*x[0]*(2.0-x[0])+((3*x[0]*x[1])**(0.5))*(2.0-x[0])*x[2], 3.0*x[1]*(1-x[0])*(1-x[1]), -(((3*x[0]*x[1])**(0.5))*x[2]**2.)*(g-1.0)]

	def sol(self):
		inic = [1e-4,self.Ophi_i*1e-9,self.li]
		res = odeint(self.f, inic, self.n1)
		return res

	def ff(self):
		def ff1(x):
			return [x[0]**2.+x[1]**2.-self.Ophi_i*1e-9, 1. + 2.*x[0]**2/(x[0]**2.+x[1]**2.)-1e-4]
		return fsolve(ff1,[1e-7,1e-7])
	def ophi_pres(self):
		"""
		present day value of density parameter for scalar field $\Omega_{phi}$
		"""
		ophi0 = self.sol()[999,1]
		return ophi0
	def ophi(self,N):
		"""
		Density parameter for scalar field as a function of log(a)
		"""
        	ophi1 = self.sol()[:,1]
        	ophi2=UnivariateSpline(self.n1,ophi1,k=3,s=0)
        	return ophi2(N)

	def ophi_z(self,z):
		"""
		Density parameter for scalar field as a function of redshift z
		"""
		return self.ophi(np.log(1./(1.+z)))


	def eqn_state_pres(self):
		"""
		Present day value of equation of state w
		"""
		eqn0 = self.sol()[999,0]-1.
		return eqn0

	def eqn_state_n(self,N):
		"""
		Equation of state as a function of log(a)
		"""
		eqnst1 = self.sol()[:,0]-1.
		eqnst2=UnivariateSpline(self.n1,eqnst1,k=3,s=0)
		return eqnst2(N)

	def eqn_state_z(self,z):
		"""
		Equation of state as a function of z
		"""
		return self.eqn_state_n(np.log(1./(1.+z)))	

	def hub(self,N):
		"""
		Dimensionless Hubble constant as a function of log(a)
		"""
		H=((1-self.sol()[999,1])*np.exp(-3*self.n1)/(1-self.sol()[:,1]))**(0.5)
		hubb=UnivariateSpline(self.n1,H,k=3,s=0)
		return hubb(N)

	def hubp(self,N):
		"""
		Derivative of dimensionless hubble constant w.r.t. log(a)
		"""
		return derivative(self.hub,N,dx=1e-6)

	def hubz(self,z):
		"""
		Dimensionless Hubble constant as a function of redshift z
		"""
		return self.hub(np.log(1./(1.+z)))

	def D_H(self):
		"""
		Hubble distance in units of Mpc (David Hogg, arxiv: astro-ph/9905116v4)
		"""
		return 2997.98/self.h

	def co_dis_n(self,N):
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
	
	def co_dis_z(self,z):
		"""
		Line of sight comoving distance as a function of redshift z
		as described in David Hogg paper
		in units of Mpc
		"""
		return self.co_dis_n(np.log(1./(1.+z)))

	#angular diameter distance
	def ang_dis_z(self,z):
		"""
		Angular diameter distance as function of redshift z
		in units of Mpc
		"""
		d_c = self.co_dis_z(z)/(1.+z)
		return d_c

	#luminosity distance
	def lum_dis_z(self,z):
		"""
		Luminosity distance in as function of redshift z
		in units of Mpc
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
		D_ds = self.D_H()*quad(self.invhub,zd,zs)[0]
		return (1+zd)*self.ang_dis_z(zd)*self.ang_dis_z(zs)/D_ds

	# CMB Shift Parameter
	def Rth(self,z):
		"""
		CMB Shift parameter
		"""
		return np.sqrt(self.om_z(0.))*(1+z)*self.ang_dis_z(z)/self.D_H()

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

	def age_by(self):
		"""
		Age of the Universe in units of billion years
		"""
		def integrand(z1):
			return self.invhub(z1)/(1.+z1)
		t0 = quad(integrand,0,np.inf)[0]
		return self.t_H()*t0

	def om(self,N):
		"""
		Density parameter for total matter \Omega_{m} as a function of efolding log(a)
		"""
		om1 = (1.-self.sol()[:,1])
		omg = UnivariateSpline(self.n1,om1,k=3,s=0)
		return omg(N)

	def om_a(self,a):
		"""
		Density parameter for total matter \Omega_{m} as a function of scale factor a
		"""
		return self.om(np.log(a))

	def om_z(self,z):
		"""
		Density parameter for total matter \Omega_{m0} as a function of redshift z
		"""
		return self.om(np.log(1./(1.+z)))
		
	def Om_diag(self,z):
		x = 1+z
		return (self.hubz(x)**2. - 1)/(x**3. - 1.)


	def deriv(self,y1,N):
		return [y1[1],-(0.5-1.5*self.ophi(N)*self.eqn_state_n(N))*y1[1]+1.5*self.om(N)*y1[0]]

	def sol1(self):
		yinit = (0.001,0.001)
		return odeint(self.deriv,yinit,self.n1)

	def D_p(self,N):
	        yn11=self.sol1()[:,0]
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
		d = self.sol1()[:,:]
		d1 = d[:,0]
		dp = d[:,1]
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
		return k**(self.ns + 2.)*tf.Tbbks(k,self.om_z(0),self.Ob0,self.h)**2.*(self.Wf(k))**2.

	def integrand_wh(self,k):
		return k**(self.ns + 2.)*tf.Twh(k,self.om_z(0),self.Ob0,self.h)**2.*(self.Wf(k))**2.

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
		return self.A0bbks()*k**self.ns*tf.Tbbks(k,self.om_z(0),self.Ob0,self.h)**2.*self.D_plus_z(z)**2.

	def Pk_wh(self,k,z):
		"""
		Matter Power Spectra Pk in units if h^{-3} Mpc^{3} as a function of k in units of [h Mpc^{-1}]
		and z;
		Transfer function is taken to be Eisenstein & Hu  
		(ref(Eisenstein and Hu, Astrophys. J., 496, 605 (1998)))
		"""
		return self.A0wh()*k**self.ns*tf.Twh(k,self.om_z(0),self.Ob0,self.h)**2.*self.D_plus_z(z)**2.

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

	def __init__(self,Ophi_i,li,h=0.7,Ob0=0.045,ns=0.96,sigma_8 = 0.8):
		self.Ophi_i = float(Ophi_i)
		self.li = float(li)
		self.Ob0 = float(Ob0)
		self.ns = float(ns)
		self.h = float(h)
		self.sigma_8 = float(sigma_8)

	def f(self,x,efold):
		g = 1.0
		return [-3.0*x[0]*(2.0-x[0])+((3*x[0]*x[1])**(0.5))*(2.0-x[0])*x[2], 3.0*x[1]*(1-x[0])*(1-x[1]), -(((3*x[0]*x[1])**(0.5))*x[2]**2.)*(g-1.0)]

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
	def __init__(self,Ophi_i,li,n,h=0.7,Ob0=0.045,ns=0.96,sigma_8 = 0.8):
		self.Ophi_i = float(Ophi_i)
		self.li = float(li)
		self.n = float(n)
		self.Ob0 = float(Ob0)
		self.ns = float(ns)
		self.h = float(h)
		self.sigma_8 = float(sigma_8)

	def f(self,x,efold):
		g = (self.n-1.)/self.n
		return [-6.*x[0]*(1.-x[0])+2.*x[2]*((1.-x[0])**(5./4.))*((3.*x[0]*x[1])**0.5), 3.0*x[1]*(1.-x[0])*(1.-x[1]), -((3*x[0]*x[1])**(0.5))*(x[2]**2)*((1.-x[0])**(0.25))*(g-1.5)]
	
class tachyonexp(tachyonpow):
	"""
	A Class to calculate cosmological observables for quintessence field
	with exponential potential.

	parameters are:
	Ophi_i : initial value (at a = 0.001) of density parameter for tachyon field
	li : initial value (at a = 0.001) for the slope	of potential (-V'(phi)/V(phi))
	Ob0 : present day density parameter for baryons (or visible matter)
	ns : spectral index for primordial power spectrum
	h : dimensionless parameter related to Hubble constant H0 = 100*h km s^-1 Mpc^-1
	sigma_8 : r.m.s. mass fluctuation on 8h^-1 Mpc scale
	"""
	def __init__(self,Ophi_i,li,h=0.7,Ob0=0.045,ns=0.96,sigma_8 = 0.8):
		self.Ophi_i = float(Ophi_i)
		self.li = float(li)
		self.Ob0 = float(Ob0)
		self.ns = float(ns)
		self.h = float(h)
		self.sigma_8 = float(sigma_8)

	def f(self,x,efold):
		g = 1.0
		return [-6.*x[0]*(1.-x[0])+2.*x[2]*((1.-x[0])**(5./4.))*((3.*x[0]*x[1])**0.5), 3.0*x[1]*(1.-x[0])*(1.-x[1]), -((3*x[0]*x[1])**(0.5))*(x[2]**2)*((1.-x[0])**(0.25))*(g-1.5)]

class galileonpow(scalarpow):
	"""
	A Class to calculate cosmological observables for Galileon field
	with power law potential.

	parameters are:
	Ophi_i : initial value (at a = 0.001) of density parameter for Galileon field
	epsilon_i : initial value (at a = 0.001) for epsilon (see readme.rst)
	li : initial value (at a = 0.001) for the slope	of potential (-V'(phi)/V(phi))
	n : order of power law potential V(phi)=phi**n
	Ob0 : present day density parameter for baryons (or visible matter)
	ns : spectral index for primordial power spectrum
	h : dimensionless parameter related to Hubble constant H0 = 100*h km s^-1 Mpc^-1
	sigma_8 : r.m.s. mass fluctuation on 8h^-1 Mpc scale
	"""
	def __init__(self,Ophi_i,epsilon_i,li,n,h=0.7,Ob0=0.045,ns=0.96,sigma_8 = 0.8):
		self.Ophi_i = float(Ophi_i)
		self.epsilon_i = float(epsilon_i)
		self.li = float(li)
		self.n = float(n)
		self.Ob0 = float(Ob0)
		self.ns = float(ns)
		self.h = float(h)
		self.sigma_8 = float(sigma_8)


	def gal(self,x,efold):
		g = (self.n-1.)/self.n
		return [(3.*(x[0]**3)*(2.+5.*x[2] + x[2]**2)-3.*x[0]*(2.-x[2]+(x[1]**2)*(2.+3.*x[2]))+(24.**0.5)*(x[1]**2)*x[3]-(6.**0.5)*(x[0]**2)*(x[1]**2)*x[2]*x[3])/(4.+4.*x[2]+(x[0]**2)*(x[2]**2)),-(x[1]*(12.*((x[1]**2)-1.)*(1.+x[2])-6.*(x[0]**2)*(2.+4.*x[2]+(x[2]**2))+(6.**0.5)*(x[0]**3)*(x[2]**2)*x[3]+(24.**0.5)*x[0]*(2.+(2.+x[1]**2)*x[2])*x[3]))/(2*(4.+4.*x[2]+(x[0]**2)*(x[2]**2))),-(x[2]*(-3.*x[0]*(-3.+x[1]**2)*(2.+x[2])+3.*(x[0]**3)*(2.+3.*x[2]+x[2]**2)-(24.**0.5)*(x[1]**2)*x[3]-(6.**0.5)*(x[0]**2)*(x[1]**2)*x[2]*x[3]))/(x[0]*(4.+4.*x[2]+(x[0]**2)*(x[2]**2))),(6.**0.5)*x[0]*(x[3]**2)*(1.-g)]

	def sol(self):
		xi = 1e-7
		yi = np.sqrt(self.Ophi_i*1e-9 - xi**2*(1+self.epsilon_i))
		inic = [xi,yi,self.epsilon_i,self.li]
		res = odeint(self.gal, inic, self.n1)
		return res

	def y_pres(self):  # omega phi present is written in terms of x,y,epsilon_i
		y0 = ((self.sol()[999,0]**2)*(1.+self.epsilon_i) + self.sol()[999,1]**2)
		return y0

	def eqn_state_pres(self):
		return (-12.*(1.+self.sol()[999,2])*self.sol()[999,1]**2 + 3.*(4.+8.*self.sol()[999,2]+self.sol()[999,2]**2)*self.sol()[999,0]**2 - (24.**0.5)*self.sol()[999,0]*self.sol()[999,2]*self.sol()[999,3]*self.sol()[999,1]**2)/(3.*(4.+4.*self.sol()[999,2]+(self.sol()[999,2]**2)*(self.sol()[999,0]**2))*(self.sol()[999,1]**2+(1.+self.sol()[999,2])*self.sol()[999,0]**2))

	def eqn_state_n(self,N):
		"""
		Equation of state as a function of log(a)
		"""
		eqnst1 = (-12.*(1.+self.sol()[:,2])*self.sol()[:,1]**2 + 3.*(4.+8.*self.sol()[:,2]+self.sol()[:,2]**2)*self.sol()[:,0]**2 - (24.**0.5)*self.sol()[:,0]*self.sol()[:,2]*self.sol()[:,3]*self.sol()[:,1]**2)/(3.*(4.+4.*self.sol()[:,2]+(self.sol()[:,2]**2)*(self.sol()[:,0]**2))*(self.sol()[:,1]**2+(1.+self.sol()[:,2])*self.sol()[:,0]**2))
		eqnst2=UnivariateSpline(self.n1,eqnst1,k=3,s=0)
		return eqnst2(N)

	def eqn_state_z(self,z):
		"""
		Equation of state as a function of z
		"""
		return self.eqn_state_n(np.log(1./(1.+z)))	


	def ophi(self,N):
		O = self.sol()[:,1]**2 + (1.+self.sol()[:,2])*self.sol()[:,0]**2
		ophi2=UnivariateSpline(self.n1,O,k=3,s=0)
		return ophi2(N)


	def ophi_pres(self):
		return self.sol()[999,1]**2 + (1.+self.sol()[999,2])*self.sol()[999,0]**2
	
	def hub(self,N):
		"""
		Dimensionless Hubble constant as a function of log(a)
		"""
		H=((1.-self.sol()[999,0]**2.*(1+self.sol()[999,2]) - self.sol()[999,1]**2.)*np.exp(-3*self.n1)/((1.-self.sol()[:,0]**2.*(1+self.sol()[:,2]) - self.sol()[:,1]**2.)))**(0.5)
		hubb=UnivariateSpline(self.n1,H,k=3,s=0)
		return hubb(N)

	def co_dis_n(self,N):
		H=((1.-self.sol()[999,0]**2.*(1+self.sol()[999,2]) - self.sol()[999,1]**2.)*np.exp(-3*self.n1)/((1.-self.sol()[:,0]**2.*(1+self.sol()[:,2]) - self.sol()[:,1]**2.)))**(0.5)
		H1=np.exp(-self.n1)/H
		tck3=interpolate.splrep(self.n1,H1,s=0)
		rs=interpolate.splint(N,0,tck3)
		return self.D_H()*rs

	def om(self,N):
		om1 = 1.-self.sol()[:,0]**2.*(1+self.sol()[:,2]) - self.sol()[:,1]**2.
		omg = UnivariateSpline(self.n1,om1,k=3,s=0)
		return omg(N)



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
	def __init__(self,Ophi_i,epsilon_i,li,h=0.7,Ob0=0.045,ns=0.96,sigma_8 = 0.8):
		self.Ophi_i = float(Ophi_i)
		self.epsilon_i = float(epsilon_i)
		self.li = float(li)
		self.Ob0 = float(Ob0)
		self.ns = float(ns)
		self.h = float(h)
		self.sigma_8 = float(sigma_8)


	def gal(self,x,efold):
		g = 1.
		return [(3.*(x[0]**3)*(2.+5.*x[2] + x[2]**2)-3.*x[0]*(2.-x[2]+(x[1]**2)*(2.+3.*x[2]))+(24.**0.5)*(x[1]**2)*x[3]-(6.**0.5)*(x[0]**2)*(x[1]**2)*x[2]*x[3])/(4.+4.*x[2]+(x[0]**2)*(x[2]**2)),-(x[1]*(12.*((x[1]**2)-1.)*(1.+x[2])-6.*(x[0]**2)*(2.+4.*x[2]+(x[2]**2))+(6.**0.5)*(x[0]**3)*(x[2]**2)*x[3]+(24.**0.5)*x[0]*(2.+(2.+x[1]**2)*x[2])*x[3]))/(2*(4.+4.*x[2]+(x[0]**2)*(x[2]**2))),-(x[2]*(-3.*x[0]*(-3.+x[1]**2)*(2.+x[2])+3.*(x[0]**3)*(2.+3.*x[2]+x[2]**2)-(24.**0.5)*(x[1]**2)*x[3]-(6.**0.5)*(x[0]**2)*(x[1]**2)*x[2]*x[3]))/(x[0]*(4.+4.*x[2]+(x[0]**2)*(x[2]**2))),(6.**0.5)*x[0]*(x[3]**2)*(1.-g)]

