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



