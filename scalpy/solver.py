import scalar
from scipy.optimize import fsolve

def solve_sp(x1,w0,op,n):
	out = [scalar.scalarpow(x1[0],x1[1],n).ophi_pres()-op]
	out.append(scalar.scalarpow(x1[0],x1[1],n).eqn_pres()-w0)
	return out
def fsp(w0,op0,n):
	return fsolve(solve_sp,[1,1],args=(w0,op0,n,))

######
def solve_se(x1,w0,op):
	out = [scalar.scalarexp(x1[0],x1[1]).ophi_pres()-op]
	out.append(scalar.scalarexp(x1[0],x1[1]).eqn_pres()-w0)
	return out

def fse(w0,op0):
	return fsolve(solve_se,[1,1],args=(w0,op0,))
######
def solve_tp(x1,w0,op,n):
	out = [scalar.tachyonpow(x1[0],x1[1],n).ophi_pres()-op]
	out.append(scalar.tachyonpow(x1[0],x1[1],n).eqn_pres()-w0)
	return out
def ftp(w0,op0,n):
	return fsolve(solve_tp,[1,1],args=(w0,op0,n,))

######
def solve_te(x1,w0,op):
	out = [scalar.tachyonexp(x1[0],x1[1]).ophi_pres()-op]
	out.append(scalar.tachyonexp(x1[0],x1[1]).eqn_pres()-w0)
	return out
def fte(w0,op0):
	return fsolve(solve_te,[1,1],args=(w0,op0,))

######
def solve_gp(x1,w0,op,epsilon,n):
	out = [scalar.galileonpow(x1[0],epsilon,x1[1],n).ophi_pres()-op]
	out.append(scalar.galileonpow(x1[0],epsilon,x1[1],n).eqn_pres()-w0)
	return out
def fgp(w0,op0,epsilon,n):
	return fsolve(solve_gp,[4,.4],args=(w0,op0,epsilon,n,))
######
def solve_ge(x1,w0,op,epsilon):
	out = [scalar.galileonexp(x1[0],epsilon,x1[1]).ophi_pres()-op]
	out.append(scalar.galileonexp(x1[0],epsilon,x1[1]).eqn_pres()-w0)
	return out
def fge(w0,op0,epsilon):
	return fsolve(solve_ge,[4,.4],args=(w0,op0,epsilon,))
