import numpy as np
#from scipy.special import sph_jn

# put best fit values of As and alpha from chi square analysis


or0 = 0.0000475
thetacmb = 2.728/2.7


def zeq(om0, h):
    return 2.5*10**4.*om0*h**2.*thetacmb**(-4.)


def bb1(om0, h):
    return 0.313*(om0*h**2.)**(-0.419)*(1.+0.607*(om0*h**2.)**0.674)


def bb2(om0, h):
    return 0.238*(om0*h**2.)**0.223


def zd(om0, ob0, h):
    return 1291.*((om0*h**2.)**0.251*(1.+bb1(om0, h)*(ob0*h**2.)**bb2(om0, h)))/(1.+0.659*(om0*h**2.)**0.828)


def keq(om0, h):
    return 0.0746*om0*h**2*thetacmb**(-2.)


def Req(om0, ob0, h):
    return 31.5*ob0*h**2.*thetacmb**(-4.)*(1000/zeq(om0, h))


def Rd(om0, ob0, h):
    return 31.5*ob0*h**2.*thetacmb**(-4.)*(1000/zd(om0, ob0, h))


def s(om0, ob0, h):
    return (2./(3.*keq(om0, h)))*np.sqrt(6/Req(om0, ob0, h))*np.log((np.sqrt(1.+Rd(om0, ob0, h))+np.sqrt(Rd(om0, ob0, h)+Req(om0, ob0, h)))/(1+np.sqrt(Req(om0, ob0, h))))


def ksilk(om0, ob0, h):
    return 1.6*(ob0*h**2.)**0.52*(om0*h**2.)**0.73*(1.+(10.4*om0*h**2.)**(-0.95))


def a1(om0, h):
    return (46.9*om0*h**2.)**(0.670)*(1.+(32.1*om0*h**2.)**(-0.532))


def a2(om0, h):
    return (12.0*om0*h**2.)**(0.424)*(1.+(45.0*om0*h**2.)**(-0.582))


def alphac(om0, ob0, h):
    return a1(om0, h)**(-ob0/om0)*a2(om0, h)**(-(ob0/om0)**3.)


def b1(om0, h):
    return 0.944*(1.+(458.*om0*h**2.)**(-0.708))**(-1.0)


def b2(om0, h):
    return (0.395*om0*h**2.)**(-0.0266)


def betac(om0, ob0, h):
    return 1./(1.+b1(om0, h)*(((om0-ob0)/om0)**b2(om0, h)-1.))


def q(k, om0, h):
    return k/(13.41*keq(om0, h))


def C1(x, k, om0, h):
    return 14.2/x + 386./(1.+69.9*q(k, om0, h)**1.08)


def T0(k, x, y, om0, ob0, h):
    return np.log(np.e+1.8*y*q(k, om0, h))/(np.log(np.e+1.8*y*q(k, om0, h))+C1(x, k, om0, h)*q(k, om0, h)**2.)


def f(k, om0, ob0, h):
    return 1./(1.+(k*s(om0, ob0, h)/5.4)**4.)


def Tc(k, x, y, om0, ob0, h):
    return f(k, om0, ob0, h)*T0(k, 1.0, y, om0, ob0, h) + (1.-f(k, om0, ob0, h))*T0(k, x, y, om0, ob0, h)


# Baryon Transfer Function
def G(x):
    return x*(-6.*np.sqrt(1.+x)+(2.+3.*x)*np.log((np.sqrt(1.+x)+1.)/(np.sqrt(1.+x)-1.)))


def alphab(om0, ob0, h):
    return 2.07*keq(om0, h)*s(om0, ob0, h)*(1.+Rd(om0, ob0, h))**(-3./4)*G((1.+zeq(om0, h))/(1+zd(om0, ob0, h)))


def betanode(om0, h):
    return 8.41*(om0*h**2.)**0.435


def betab(om0, ob0, h):
    return 0.5+ob0/om0+(3.-2.*ob0/om0)*np.sqrt((17.2*om0*h**2.)**2.+1.)


def s1(k, om0, ob0, h):
    return s(om0, ob0, h)/((1.+(betanode(om0, h)/(k*s(om0, ob0, h)))**3.)**(1./3))


def Tb(k, x1, y1, om0, ob0, h):
    return (T0(k, 1.0, 1.0, om0, ob0, h)/(1.+(k*s(om0, ob0, h)/5.2)**2.)+x1/(1.+(y1/(k*s(om0, ob0, h)))**3.)*np.exp(-(k/ksilk(om0, ob0, h))**1.4))*np.sin(k*s1(k, om0, ob0, h))/(k*s1(k, om0, ob0, h))


# Total Power Spectrum
def Twh(k, om0, ob0, h):
    kk = k*h
    if ob0 == 0:
        ans = T0(kk, alphac(om0, ob0, h), betac(om0, ob0, h), om0, ob0, h)
    else:
        ans = ob0/om0*Tb(kk, alphab(om0, ob0, h), betab(om0, ob0, h), om0, ob0, h)+(
            om0-ob0)/om0*Tc(kk, alphac(om0, ob0, h), betac(om0, ob0, h), om0, ob0, h)
    return ans

# print s,zeq,zd
# BBKM Transfer function


def gm(om0, ob0, h):
    return om0*h*np.exp(-ob0*(1. + np.sqrt(2.*h)/om0))


def q1(k, om0, ob0, h):
    return k/(gm(om0, ob0, h))


def Tbbks(k, om0, ob0, h):
    return (np.log(1. + 2.34*q1(k, om0, ob0, h))/(2.34*q1(k, om0, ob0, h)))*(1.+3.89*q1(k, om0, ob0, h) + (16.1*q1(k, om0, ob0, h))**2. + (5.46*q1(k, om0, ob0, h))**3. + (6.71*q1(k, om0, ob0, h))**4)**(-0.25)
