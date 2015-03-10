==================================================================
ScalPy: A python tool to study dynamical systems in cosmology
==================================================================
This code solves dynamical system for scalar fields such as minimally coupled 
quintessence, tachyon and Galileon fields. We have included two types of potential 
for each scalar field: power law V(phi)=phi^n and exponential 
V(phi) = exp(K*phi) where n and K are real constants.

#############################

Installation:
=================
In Linux system, install scalpy using pip

	% sudo pip install scalpy

or to install it using source, download the tar file from the url

	https://github.com/sum33it/scalpy

	% unzip scalpy-master.zip
	% cd scalpy-master
	% sudo python setup.py install


Structure:
==========

There are following important files in the code:


1) scalar.py

This file solves the dynamical system for scalar field where we give initial 
conditions (at decoupling) on quantities like field evolution, slope of the 
potential etc. as input (see below for details). In this file, different 
observables such as luminosity distance, Hubble parameter, angular diameter distance, 
growth rate, growth function, power spectrum are defined.

2) solver.py

For a given set of w0(present value of e.o.s) and Ophi_0 (present value of energy 
density parameter of scalar field), this solves for the initial conditions needed at 
decoupling.

3) fluids.py

This gives different observables for standard cosmological models such as LCDM, wCDM, w0waCDM and GCG.
In this file also, different observables such as luminosity distance, Hubble parameter, 
angular diameter distance, growth rate, growth function, power spectrum are defined.

4) transfer_func.py

This provides transfer functions given by Eisenstein and Hu as well as BBKS.


========================
pre-installed softwares:
========================
You should have following packages installed in your system:

-numpy

-scipy


===============================
Using the package
===============================

start your favourite python interpretor

	>>> python
or

	>>> ipython

then import scalpy

	>>> import scalpy

If you want to work with scalar field with power law potential, import the class named scalarpow from scalar	

	>>> from scalpy.scalar import *
	>>> x = scalarpow(2.0,0.2,2)

First argument is Ophi_i (initial value of density parameter for scalar field) second 
argument is lambda_i (initial value of parameter lambda_i) and third argument is order 
of power to scalar field power law potential.
One can put fourth, fifth, sixth and seventh arguments which are h, Ob0, ns and 
sigma8 if you don't want to use default values which are 
(h = 0.67, Ob0 = 0.045, ns = 0.96, sigma_8 = 0.8)

If you want to use tachyonic scalar field with exponential potential, you should 
import the class named "tachyonexp". Below is the present list of classes included in 
the package with the arguments they need:

    scalarpow(Ophi_i,lambda_i,n): scalar field with power law potential
    scalarexp(Ophi_i,lambda_i): scalar field with exponential potential
    tachyonpow(Ophi_i,lambda_i,n): tachyonic scalar field with power law potential
    tachyonexp(Ophi_i,lambda_i): tachyonic scalar field with exponential potential
    galileonpow(Ophi_i,epsilon,lambda_i,n): galileon field with power law potential
    galileonexp(Ophi_i,epsilon,lambda_i): galileon field with exponential potential

arguments are:

    ophi_i: initial value (at a = 0.001 or z \approax 1000) of density parameter for scalar field
    lambda_i: initial value for lambda parameter (see "Sen and Scherrer")
    epsilon: initial value for "epsilon" parameter in Galileon field 
            (see W. Hossain and A.A. Sen, Phys. Lett. B713 (2012), 140-144, arxiv: 1201.6192)
    n: order of power law potential V(phi)=phi^n


There are many attributes of the corresponding class. These are the standard 
cosmological functions such as normalized Hubble parameter, density parameters for 
matter and scalar field, luminosity distance, etc. Below is the list of attributes to 
a given class with arguments:

	hub(N): dimensionless Hubble parameter as a function of N=log(a) normalized today at 1
	ophi_pres(): present day density parameter for scalar field
	ophi_z(z): density parameter for scalar field as function of redshift z
	eqn_state_pres(): present day equation of state for scalar field
	eqn_state_n(N): equation of state for scalar field as a function of N=log(a)
	eqn_state_z(z): equation of state for scalar field as a function of redshift z
	hubz(z): dimensionless Hubble parameter as a function of z normalized today at 1
	co_dis_z(z): comoving distance (Mpc) as a function of redshift z
	ang_dis_z(z): angular diameter distance (Mpc) as a function of z
	lum_dis_z(z): luminosity distance (Mpc) as a function of z
	time_delay_dis(zd,zs): time delay distance; zd = redshift of lens, zs = redshift of source
	lookback_time_z(z): lookback time (in billion years) as a funtion of z
	age_by(): age of the universe in units of billion years
	om_z(z): density parameter for matter as a function of z
	Om_diag(z): Om diagnostic (given by Sahni et al) as a function of redshift z
	Rth(z): CMB shift parameter as a function of redshift z
	D_plus_z(z): growing mode solution of growth equation
	fsigma8z(z): f*sigma8 as a function of z
	Pk_bbks(k,z): linear matter power spectrum (Using BBKS transfer function) as a 
				function of wave number k and redshift z
	Pk_wh(k,z): linear matter power spectrum (Using Eisenstein-Hu transfer function) 
				as a function of wave number k and redshift z
	DPk_bbks(k,z): dimensionless linear power spectrum(Using BBKS transfer function) 
				as a function of wave number k and redshift z
	DPk_wh(k,z): dimensionless linear power spectrum (Using Eisenstein-Hu transfer 
				function) as a function of wave number k and redshift z

To calculate an atribute to a class,

	>>> x.hubz(0)
	>>> array(1.0)

to calculate power spectrum

	>>> x.Pk_wh(0.01,0.1)

Alternatively, you can also use it like

	>>> scalarpow(2.0,0.3,2).hubz(0)
	>>> scalarpow(2.0,0.3,2).Pk_wh(0.01,0.1)


One can give four more arguments while calling classes, namely h, Ob0, ns and sigma_8 
if one don't want to use default values of h = 0.7, Ob0 = 0.045, ns = 0.96  and 
sigma_8 = 0.8 e.g.

	>>> galileonpow(2.0,0.5,0.1,2,0.67,0.045,0.9634,0.83).Pk_bbks(1.0,0)

In arguments of attributes to a class, one can give a value as well as an array. 
For this functions np.vectorize should be used. e.g.

	>>> x = scalarexp(2.0,0.1)
	>>> z = np.linspace(0,1,10)
	>>> f = np.vectorize(x.dis_z)
	>>> distance = f(z)

For LCDM, wCDM, w0waCDM and GCG models, one should import fluids.py file

	>>> from scalpy.fluids import *
	>>> x = LCDM(0.3)
	>>> x.hubz(0.1)

Here we have four classes: LCDM for concordance Lambda CDM model, wCDM for constant w 
model with w not equat to -1, w0waCDM for varying dark energy model with CPL 
parametrization and GCG for generalized chaplygin gas. The arguments for different 
classes are:

    LCDM(Om0): standard Lambda CDM model with cosmological constant. Om0 is present value of density parameter for total matter.
    wCDM(Om0,w): Dark energy model with dark energy density goes as (1+z)^(3(1+w))
    w0waCDM(Om0,w0,wa): Varying dark energy model with CPL parametrization w(a) = w0 + wa*(1-a)
    GCG(Om0,As,alpha): Dark energy is parametrized by equation of state for GCG

the atributes to these classes are:

	hubz(z): dimensionless Hubble parameter as a function of redshift z
	dis_z(z): comoving distance as a function of redshift z
	ang_dis_z(z): angular diameter distance as a function of z
	lum_dis_z(z): luminosity distance as a function of z
	time_delay_dis(zd,zs): time delay distance; zd = redshift of lens, zs = redshift of source
	lookback_time_z(z): lookback time (in billion years) as a funtion of z
	age_by(): age of the universe in units of billion years
	om_z(z): density parameter for matter as a function of z
	Om_diag(z): Om diagnostic (given by Sahni et al) as a function of redshift z
	Rth(z): CMB shift parameter as a function of redshift z
	D_plus_z(z): growing mode solution of growth equation
	fsigma8z(z): f*sigma8 as a function of z
	Pk_bbks(k,z): linear matter power spectrum (Using BBKS transfer function) as a 
				function of wave number k and redshift z
	Pk_wh(k,z): linear matter power spectrum (Using Eisenstein-Hu transfer function) 
				as a function of wave number k and redshift z
	DPk_bbks(k,z): dimensionless linear power spectrum(Using BBKS transfer function) 
				as a function of wave number k and redshift z
	DPk_wh(k,z): dimensionless linear power spectrum (Using Eisenstein-Hu transfer 
				function) as a function of wave number k and redshift z

To calculate linear power spectrum at redshift z=0 and for k = 0.01 for w0waCDM model,

	>>> w0waCDM(0.3,-1.02,0.5).Pk_bbks(0.01,0)

Here we have taken Om0=0.3, w0=-1.02 and w0 = 0.5

Here also, one can give four more arguments while importing classes, namely Ob0, ns, h 
and sigma_8 if one doesn't want to use default values of h = 0.7, Ob0 = 0.045, ns = 0.96 
and sigma_8 = 0.8 e.g.

	>>> w0waCDM(0.3,-1.02,0.5,0.67,0.045,0.9634,0.83).Pk_bbks(1.0,0)

######################################################
Some example (for plotting)
######################################################

	>>> python
	>>> import pylab as pl
	>>> from scalpy.scalar import *
	>>> x = galileonpow(2.1,10.0,0.2,2)
	>>> z1 = pl.linspace(0,2,100) 
	>>> y = x.hubz(z1)
	>>> pl.plot(z1,y)
	>>> pl.show()


If you want to plot linear matter power spectrum on log scale from k = 0.001 to k = 1.0 at redshift z = 0

	>>> k = pl.logspace(-3,0,100)
	>>> P = x.Pk_wh(k,0)
	>>> pl.loglog(k,P)
	>>> pl.show()
