import numpy as np
import scipy.special as special
import scipy.constants as cnst
import scipy.integrate as integrate
import scipy.interpolate as interpol
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt
from matplotlib import patches
from pathlib import Path


#this class calculates the tSZ signal assuming the Arnaud10 universal
#pressure profile
#for now, ignore correction for relativistic effects
#the calculation follows Section 2.2 of Hasselfield+ 2013 
class ClusterUPP:
    """
    Parameters
    ----------
    M500solar : float
        Mass at R_500 for the cluster. Units are in solar masses.
    z : float
        Redshift of the cluster. Unitless.
    vP : float, optional
        Peculiar velocity of the cluster. The default is 1000 km/s.
        Units are km/s. Will be converted to m/s in the code.
    h70 : float, optional
        Dimensionless Hubble constant. The default is 1. Unitless.

    Returns
    -------
    Cluster object.

    """
    #instantiate
    def __init__(self, M500solar, z, velocity = 1000, h70=1.):
      
        self.M500 = M500solar # cluster mass internal to R500 [solar mass]
        self.M500kg = M500solar*1.989e30 # [kg]
        self.vP = velocity * 1e3 # peculiar velocity [m/s]
        self.h70 = h70
        self.z = z
        self.P500 = self.calcP500()
        self.R500 = self.calcR500() #[ m]
        self.R500Mpc = m2pc(self.R500) * 1e-6 #[Mpc] 
        self.theta500 = self.calcTheta500()
        self.theta500arcmin = self.theta500*(180./np.pi)*60.
        self.c500 = 1.177        
        self.Tcmb = 2.726 #CMB temperature [K]
        self.Tisothermal = self.calcTisothermal()


        
        # TolTEC bandpass files
        dirpath = Path(__file__).parent
        with np.load(dirpath.joinpath('passbands/FTS_avg1p1.npz')) as bp_1p1x:
            self.bp_1p1x = {k: bp_1p1x[k] for k in bp_1p1x.files}
        with np.load(dirpath.joinpath('passbands/FTS_avg1p4.npz')) as bp_1p4x:
            self.bp_1p4x = {k: bp_1p4x[k] for k in bp_1p4x.files}
        with np.load(dirpath.joinpath('passbands/FTS_avg2p0.npz')) as bp_2p0x:
            self.bp_2p0x = {k: bp_2p0x[k] for k in bp_2p0x.files}
        
        # get the transmission and frequencies (in GHz)
        self.bp1p1_freq = self.bp_1p1x['fc']
        self.bp1p1_band = norm(self.bp_1p1x['sc'],self.bp1p1_freq)

        self.bp1p4_freq = self.bp_1p4x['fc']
        self.bp1p4_band = norm(self.bp_1p4x['sc'],self.bp1p4_freq, wave=1.4)
        
        self.bp2p0_freq = self.bp_2p0x['fc']
        self.bp2p0_band = norm(self.bp_2p0x['sc'],self.bp2p0_freq, wave=2.0)

    #the generalized NFW profile pressure
    #Nagai et al. 2007
    #with best-fit parameters from Arnaud+2010 Equations 11 and 12
    #the returned quantity is unitless
    def gnfwp(self, x):
        P0 = 8.403*self.h70**(-1.5)
        c500 = self.c500        
        gamma = 0.3081
        alpha = 1.0510
        beta = 5.4905
        term1 = P0
        term2 = (c500*x)**gamma
        term3 = (1.+(c500*x)**alpha)**((beta-gamma)/alpha)
        return term1/(term2*term3)


    #the physical pressure profile P(x)
    #Arnaud+2010, Eq. 13
    #Note x = r/R500
    #units of output are keV/cm^3
    def P(self, x):
        alpha_p = 0.12  #eq. 7
        alpha_pp = 0.1-(alpha_p+0.1)*((x/0.5)**3)/(1.+(x/0.5)**3) #eq. 8
        term1 = 1.65e-3*(H(self.z)/H(0))**(8./3.)
        term2 = (self.M500/(3.e14/self.h70))**(2./3.+alpha_p+alpha_pp)
        term3 = self.gnfwp(x)*self.h70**2
        return term1*term2*term3

    
    #definition of P500 from Arnaud+2010, Eq. 5
    #units of output are keV/cm^3
    def calcP500(self):
        term1 = 1.65e-3*(H(self.z)/H(0))**(8./3.)
        term2 = (self.M500/(3.e14/self.h70))**(2./3.)
        return term1*term2


    #R500
    #units in m
    def calcR500(self):
        return (2.*cnst.G/500.*self.M500kg/H(self.z)**2)**(1./3.)


    #theta500
    #units in rad
    def calcTheta500(self):
        return self.R500/DA(self.z)


    #the UPP along a chord through the cluster
    #do units conversion from keV/cm^3 to J/m^3
    #there's something not quite right here ... shows up in y_0 calc.
    def Pchord(self, x, thetaArcmin):
        roff = self.R500*(thetaArcmin/self.theta500arcmin)
        xoff = roff/self.R500
        s = np.sqrt(x**2 + xoff**2)
        return self.P(s)*1.60218e16*1.e-6

    
    #tau as a function of angular distance, theta
    #do integral of Pchord along line of sight from 0.01*R500 to 5*R500 then mult by 2
    #theta in arcminutes is angle offset from cluster center
    #finally normalize to 1 at tau(0)
    #following H13
    def tau(self, thetaArcmin):
        I = integrate.quad(self.Pchord, 0.01, 5., args=(thetaArcmin,))
        I0 = integrate.quad(self.Pchord, 0.01, 5., args=(0.,))
        return I[0]/I0[0]

    
    #compton parameter y(theta)
    #theta in arcminutes
    #following H13, equation 7
    def y(self, thetaArcmin):
        m = self.M500/(3.e14/self.h70)
        tenA0 = 4.95e-5*np.sqrt(self.h70)
        B0 = 0.08
        C0 = -0.025
        Ez = H(self.z)/H(0)
        #need to differentiate a scalar from an np.array
        if(isinstance(thetaArcmin, (list, tuple, np.ndarray))):
            tau = np.zeros(len(thetaArcmin))
            for i in np.arange(len(thetaArcmin)):
                tau[i] = self.tau(thetaArcmin[i])
        else:
            tau = self.tau(thetaArcmin)
        return tenA0*(Ez**2)*(m**(1.+B0))*tau
        

    #returns a rotationally symmetric and smoothed image of the
    #cluster y parameter
    def image(self, radiusArcmin, beamFWHMArcsec, dT=0, nu_obs_GHz=150.):
        #do the radial calculation first
        npts = 100
        r = np.linspace(0.,1.5*radiusArcmin,npts)
        y = self.y(r)

        if(dT):
            print("Output map in deltaT units at {} GHz.".format(nu_obs_GHz))
            y = self.deltaT(y,nu_obs_GHz)
        
        imy = interpol.interp1d(r,y,kind='quadratic')
        #create the image
        nx = int(np.ceil(radiusArcmin*60*2))        
        x = np.linspace(-radiusArcmin, radiusArcmin, nx)
        y = np.linspace(-radiusArcmin, radiusArcmin, nx)
        pixScale = x[1]-x[0]
        xx,yy = np.meshgrid(x,y)
        img = imy(np.sqrt(xx**2+yy**2))
        #now smooth it with a gaussian
        beamSigmaPixels = (beamFWHMArcsec/60.)/2.35482004503/pixScale
        img = ndimage.gaussian_filter(img, beamSigmaPixels)
        return img, x, y


    #the images of the thermal and kinetic effects. The thermal effect
    # is y [unitless].  The kinetic effect is delta T [K].  An
    # isothermal sphere model of the cluster is used for the electron
    # temperature calculation
    def SZImages(self, radiusArcmin):
        #do the radial calculation first
        npts = 100
        r = np.linspace(0.,1.5*radiusArcmin,npts)
        y = self.y(r)
        imy = interpol.interp1d(r,y,kind='quadratic')
        #create the image
        nx = int(np.ceil(radiusArcmin*60*2))        
        x = np.linspace(-radiusArcmin, radiusArcmin, nx)
        y = np.linspace(-radiusArcmin, radiusArcmin, nx)
        pixScale = x[1]-x[0]
        xx,yy = np.meshgrid(x,y)
        img = imy(np.sqrt(xx**2+yy**2))
        #now the kinetic image in temperature units
        beta = self.vP/cnst.c
        fact = cnst.m_e*cnst.c**2/(cnst.k*self.Tisothermal)
        kimg = -img*beta*fact*2.726
        return img, kimg, x, y


    #returns delta_T given y at an obs freq.
    def deltaT(self, y, nu_obs_GHz):
        #dimensionless frequency
        x = cnst.h*nu_obs_GHz*1.e9/(cnst.k*2.726)
        #from Sean's sensitivity-millimeter.pdf memo
        term2 = x*(np.exp(x)+1.)/(np.exp(x)-1.)-4.
        return y*term2*2.726
    

    #returns delta_I/I given y at an obs freq.
    def deltaI(self, y, nu_obs_GHz):
        #dimensionless frequency
        x = cnst.h*nu_obs_GHz*1.e9/(cnst.k*2.726)
        #the thermal SZ eq. from Carlstrom, Holder and Reese 2002
        #Eq. 4
        term1 = x**4*np.exp(x)/(np.exp(x)-1.)**2
        term2 = x*(np.exp(x)+1.)/(np.exp(x)-1.)-4.
        
        #commented out the term that makes this dI_nu and not dI_nu/I_0
        return y*term1*term2 #*Planck(nu_obs_GHz*1.e9,2.726)*1.e26/1.e6


    #electron optical depth vs radius in arcmins
    def tau_e(self, thetaArcmin):
        term1 = (cnst.m_e * cnst.c ** 2.) 
        term2 = kT_e(self.M500, self.R500Mpc) * 1e3 * 1.60218e-19 #keV to J
        return (term1 / term2) * self.y(thetaArcmin)


    def deltaIkSZE(self, thetaArcmin, v, nu_obs_GHz):
        beta = v / cnst.c # peculiar velocity / speed of light
        #dimensionless freuqency
        x = cnst.h * nu_obs_GHz * 1.e9 / (cnst.k * 2.726)
        #kSZ eq from Birkinshaw 1999
        # Eqn. 85
        xterm = (x ** 4 * np.exp(x)) / (np.exp(x) - 1) ** 2
        return (-beta * self.tau_e(thetaArcmin) * xterm)


    #calculate T_e assuming cluster is isothermal and in hydrostatic
    #equilibrium.  This is thought to be a good estimate for the
    #electron temperature near the cluster core.  This will be used in
    #the kSZ calculation.
    def calcTisothermal(self):
        mp = cnst.m_p
        G = cnst.G
        kb = cnst.k
        return G*mp*self.M500kg/(2.*self.R500*kb)





    
    #deltaI/I versus nu plot for a given y
    def plotSZE(self, y, nu_obs_GHz):
        
        # calculate intensity for CMB and         
        cmbInu = Planck(nu_obs_GHz * 1e9, self.Tcmb) * 1e26 / 1e6 #in MJy/sr
        dusty = dustGal(nu_obs_GHz * 1e9) #trouble with units on this guy #* 1e26 / 1e6 #in MJy/sr
        
        # calculate dI/I for the SZEs
        dI_I = self.deltaI(y, nu_obs_GHz) #tSZE
        dI_IkSZ = self.deltaIkSZE(0., self.vP, nu_obs_GHz)
        
        fig, ax = plt.subplots()

        # the dI/I plots. the first line is the scaled CMB spectrum
        # ax.plot(nu_obs_GHz, cmbInu / 5.5e4, 'b--', label = r'$B_\nu(T_{CMB})$')
        ax.plot(nu_obs_GHz, dusty / 5.5e4, 'r', linestyle = 'dotted', label = r'$I_\nu(T_{dust} = 30 K)$')
        ax.plot(nu_obs_GHz, dI_I, color = 'b', label = 'tSZE, ' + r'$y_0$'+' = {0:.4f}'.format(y))
        ax.plot(nu_obs_GHz, dI_IkSZ, color = 'k', linestyle = 'dashed', label = 'kSZE, ' + r'$\tau_0$' + ' = {0:.4f}'.format(self.tau_e(0.)))

        #TolTEC bands
        # ax.axvspan(128., 170., alpha = 0.5, color = 'tab:red')
        # ax.axvspan(195., 245., alpha = 0.5, color = 'tab:green')
        # ax.axvspan(245., 310., alpha = 0.5, color = 'tab:blue')        
        ax.plot(self.bp1p1_freq, self.bp1p1_band * 3.5e-3, color='tab:blue', alpha = 0.45)
        ax.fill_between(self.bp1p1_freq, self.bp1p1_band * 3.5e-3, color = 'tab:blue', alpha = 0.5)

        ax.plot(self.bp1p4_freq, self.bp1p4_band * 3.5e-3, color='tab:green', alpha = 0.45)
        ax.fill_between(self.bp1p4_freq, self.bp1p4_band * 3.5e-3, color = 'tab:green', alpha = 0.5)

        ax.plot(self.bp2p0_freq,self.bp2p0_band * 3.5e-3, color='tab:red', alpha = 0.45)   
        ax.fill_between(self.bp2p0_freq, self.bp2p0_band * 3.5e-3, color = 'tab:red', alpha = 0.5)
        
        #Planck bands -- highest bands out of range
        hatchstyle = '///' # style for the hatching on the bandpass indicators 
        # ax.axvspan(83, 120, alpha = 0.25, facecolor = 'none', edgecolor = 'tab:red', hatch = hatchstyle)
        # ax.axvspan(120, 170, alpha = 0.25, facecolor = 'none', edgecolor = 'tab:green', hatch = hatchstyle)
        # ax.axvspan(180, 270, alpha = 0.25, facecolor = 'none', edgecolor = 'tab:cyan', hatch = hatchstyle)
        # ax.axvspan(300, 430, alpha = 0.25, facecolor = 'none', edgecolor = 'tab:blue', hatch = hatchstyle)
        
        #Herschel bands - SPIRE
        # ax.axvspan(0.45e3, 0.75e3, alpha = 0.25, facecolor = 'none',edgecolor = 'navy', hatch = ' \\ ')
        
        # xaxis 
        xmin = min(nu_obs_GHz)
        xmax = max(nu_obs_GHz)
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel('Frequency [GHz]') 
        
        # yaxis
        ymin = min(dI_I) + 0.2 * min(dI_I)
        ymax = max(dI_I) + 0.2 * max(dI_I)
        
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(r'$\Delta I/I_{0}$')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))        
 
        # Planck bands
        ax.add_patch(patches.Rectangle((83,ymax), 120 - 83, 2.5e-4, clip_on=False, facecolor='none',edgecolor='tab:red',hatch=hatchstyle))
        ax.add_patch(patches.Rectangle((120,ymax), 170 - 120, 2.5e-4, clip_on=False, facecolor='none',edgecolor='tab:green',hatch=hatchstyle))
        ax.add_patch(patches.Rectangle((180,ymax), 270 - 180, 2.5e-4, clip_on=False, facecolor='none',edgecolor='tab:cyan',hatch=hatchstyle))
        ax.add_patch(patches.Rectangle((300,ymax), 430 - 300, 2.5e-4, clip_on=False, facecolor='none',edgecolor='tab:blue',hatch=hatchstyle))

        # Herschel bands
        ax.add_patch(patches.Rectangle((450,ymax), xmax - 450, 2.5e-4, clip_on=False, facecolor='none',edgecolor='navy',hatch=' \ '))
        
        # legend
        ax.legend(loc = 'lower right')  
        
        # line to show where 0 is on the yaxis
        # ax.hlines(0., min(nu_obs_GHz), max(nu_obs_GHz), color = 'k')
        
        return
    
    #returns central density (rho_0) of NFW profile
    def getRho0(self):
        return self.M500/(4.*np.pi*self.R500**3*(np.log(2.)-0.5))


    #given input source locations, provides updated locations and
    #magnifications.
    #this follows Hurtado+2014 in Applied Mathematics (1308.5271.pdf)
    #thetaXArcsec - array of source x locations in arcseconds in source plane
    #thetaYArcsec - array of source y locations in arcseconds in source plane
    #z - redshifts of sources
    def lensSources(self, thetaXArcsec, thetaYArcsec, z):
        nSources = len(thetaXArcsec)
        rho0 = self.getRho0()
        r_s = self.R500/self.c500
        theta_s = r_s/self.R500*self.theta500

        #unitless distances from cluster center in source (y) and lens (x) planes
        y = np.deg2rad(np.sqrt(thetaXArcsec**2 + thetaYArcsec**2)/3600.)/theta_s
        x = np.zeros(nSources)
        
        #calculate C for each source
        Dsource = np.zeros(nSources)
        for i in np.arange(nSources):
            Dsource[i] = DA(z[i])
        Dlens = DA(self.z)
        Dls = Dsource-Dlens
        Sigma_crit = cnst.c**2/(4.*np.pi*cnst.G)*Dsource/(Dlens*Dls)
        C = Sigma_crit/(4.*rho0*r_s)

        #generate interpolations for f(x) and g(x) out to 3*theta500
        #note these are smooth so don't go nuts
        xp = np.linspace(0.,3.*self.theta500,100)/self.theta500
        fxp = f(xp)
        
        #go source by source now
        for i in np.arange(nSources):
            alphaxp = xp/C[i]*fxp
            yp = xp-alphaxp
            yx = interpol.interp1d(yp,xp,kind='quadratic')
            x[i] = yx(y[i])

        #back to vector math for these calcs
        mu = C**2/np.abs(C-f(x))/np.abs(C-g(x))
        t = np.arctan2(thetaYArcsec, thetaXArcsec)
        thetaYoutArcsec = x*self.theta500arcmin*60*np.sin(t)
        thetaXoutArcsec = x*self.theta500arcmin*60*np.cos(t)
        return thetaXoutArcsec, thetaYoutArcsec

# ------------------------------------------------------
# external functions of convinience 
# ------------------------------------------------------

#lens equation 1.58 from Hurtado+2014
def f(x):
    t1 = 1./x**2
    t2 = np.log(x/2.)
    t3 = 2./np.sqrt(1.-x**2)
    t4 = np.arctanh(np.sqrt(1.-x)/np.sqrt(1.+x))
    return t1*(t2+t3*t4)

#lens equation 1.59 from Hurtado+2014
def g(x):
    t1 = -1./x**2
    t2 = np.log(x/2.)
    t3 = x**2/(1.-x**2)
    t4 = 2.*(1.-2.*x**2)/(1.-x**2)**1.5
    t5 = np.arctanh(np.sqrt(1.-x)/np.sqrt(1.+x))
    return t1*(t2+t3+t4*t5)

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

def kev2kelvin(kT):
    return kT  / (8.61733e-5 * 1e-3)

def rad2arcsec(theta):
    return theta*180./np.pi*3600.

def rad2arcmin(theta):
    return theta*180./np.pi*60.

def arcsec2rad(theta):
    return theta/3600.*np.pi/180.

# calculate electron temp from Eq 12 in B99 
# assumes HSE. Units = keV
def kT_e(M, R_eff):
    return 7 * (M / (3e14)) * (R_eff) ** -1

def Planck(nu,T):
    x = cnst.h*nu/(cnst.k*T)
    term1 = cnst.h*nu**3/cnst.c**2
    term2 = 1./(np.exp(x)-1.)
    return term1*term2

def dustGal(nu, beta = 1.5, T = 30):
    term1 = nu ** beta
    term2 = Planck(nu, T)
    
    return term1 * term2

# function to make passband array zero outside of band
def norm(val, nu, wave = 1.1, model = 'no'):
    """
    Inputs:
    val = bandpass value
    nu = frequency
    wave = wavelength of band (1.1, 1.4, or 2.0)
    
    Outputs:
    p_nu_norm = normalized bandpass
    """    
    
    p_nu = np.copy(val)

    if wave == 1.1:
        minVal = 230
        maxVal = 340
        
    elif wave == 1.4:
        minVal = 180
        maxVal = 260    
        
    elif wave == 2.0:
        minVal = 120
        maxVal = 180
        
    if model == 'yes':
        if wave == 1.1:
            minVal = 245
            maxVal = 310

        elif wave == 1.4:
            minVal = 195
            maxVal = 245     

        elif wave == 2.0:
            minVal = 128
            maxVal = 170
        
    mask = np.where((nu <= minVal) | (nu >= maxVal) | (p_nu < 0))[0]
    # set freq values outside of band to 0
    p_nu[mask] = 0.
    
    
    # normalizes the band to the highest value
    p_nu_norm = p_nu / max(p_nu)
    
    # returns array going from 0 to 1
    return p_nu_norm
