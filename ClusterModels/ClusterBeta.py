import numpy as np
import scipy.special as special
import scipy.constants as cnst
import scipy.integrate as integrate


class Cluster:

    #instantiate
    def __init__(self,theta_c_arcsec, M200_Msun, beta, z, fgas=0.116):
        self.theta_c = arcsec2rad(theta_c_arcsec)
        self.M200 = M200_Msun * 1.989e30           #kg
        self.beta = beta
        self.z = z
        self.fgas=fgas

        self.r200 = self.calcR200()
        self.theta200 = self.calcTheta200()
        self.rc = self.calcRc()
        self.n0 = self.calcN0()
        self.tau0 = self.calcTau0()
        self.Te = self.calcTe()
        self.y0 = self.calcy0()

        #useful units
        self.r200_Mpc = m2pc(self.r200)*1.e-6
        self.theta200_arcsec = rad2arcsec(self.theta200)
        self.rc_kpc = m2pc(self.rc)*1.e-3
        self.Te_keV = kelvin2kev(self.Te)
        self.n0_iccm = self.n0/100./100./100.
        
        
    #the projected Compton parameter as a function of angle theta under
    #the assumption of an isothermal beta profile
    def y(self,theta_arcsec):
        theta = arcsec2rad(theta_arcsec)
        return self.y0*(1+(theta/self.theta_c)**2)**((1.-3.*self.beta)/2.)

    #the central projected Compton parameter
    def calcy0(self):
        return self.tau0*cnst.k*self.Te/(cnst.m_e*cnst.c**2)

    #r200
    def calcR200(self):
        return (2.*cnst.G/200.*self.M200/H(self.z)**2)**(1./3.)

    #theta200
    def calcTheta200(self):
        return self.r200/DA(self.z)
    
    #core radius in terms of theta_c
    def calcRc(self):
        return DA(self.z)*self.theta_c
    
    #n0 - the central density
    def calcN0(self):
        num = self.fgas*self.M200
        den = (4.*np.pi*cnst.m_p*Ib(self.r200,self.rc,self.beta))
        return num/den

    #the projected central optical depth
    def calcTau0(self):
        sigmaT = 6.6524e-29   #m^2
        fact = special.gamma(3.*self.beta/2.-0.5)/special.gamma(3.*self.beta/2.)
        return self.n0*sigmaT*np.sqrt(np.pi)*(self.theta_c*DA(self.z))*fact

    #electron temperature in K
    def calcTe(self):
        mu = 0.88
        num = self.M200*(self.r200**2+self.rc**2)*cnst.G*mu*cnst.m_p
        den = 3.*self.beta*cnst.k*self.r200**3
        return num/den

    #the radial dependence of the electron density under the isothermal
    #beta profile
    def calcN_e(self,r):
        return self.n0*(1.+(r/self.rc)**2)**(-3.*self.beta/2.)

#Ib - the volume integral of the radial dependence of the beta-model
def Ibarg(r,rc,beta):
    return r**2*(1.+(r/rc)**2)**(-3.*beta/2.)
def Ib(r200,rc,beta):
    return integrate.romberg(Ibarg,0.,r200,args=(rc,beta))
    
#the hubble constant (note the cosmology)
def H(z):
    H0 = 2.2e-18          #SI units [1/s]
    OmegaM = 0.3
    OmegaLambda = 0.7
    return H0*np.sqrt(OmegaM*(1.+z)**3 + OmegaLambda)

#the critical density
def rhocrit(z):
    return 3./8.*H(z)**2/(8.*np.pi*cnst.G)

#angular diameter distance
def DAarg(z):
    return 1./H(z)
def DA(z):
    return cnst.c/(1.+z)*integrate.romberg(DAarg,0.,z)

#converts meters to parsecs
def m2pc(dist):
    return dist/3.086e16

#converts parsecs to meters
def pc2m(dist):
    return dist*3.086e16

def kelvin2kev(T):
    return 8.617e-8*T

def rad2arcsec(theta):
    return theta*180./np.pi*3600.

def arcsec2rad(theta):
    return theta/3600.*np.pi/180.
