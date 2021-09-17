from astropy.table import Table
import scipy.ndimage as ndimage
import scipy.constants as cnst
import numpy as np
from pathlib import Path

#map depth, area, or time
class TolTEC:

    #instantiate
    def __init__(self, band, atmFactor=1.,
                 tablePath=Path(__file__).parent):
        self.atmFactor = atmFactor
        self.FOV_arcmin2 = np.pi*2.**2
        self.ynorm = 0.5e-4
        self.vnorm = 500.      #km/s
        self.taunorm = 0.005
        self.band=band
        if(band==1.1):
            self.nu = cnst.c/(1.1e-3)
            self.fwhm = 5.
            self.MS = 22./atmFactor
            self.arrayTable = tablePath.joinpath("a1100_prop.ecsv")
        elif(band==1.4):
            self.nu = cnst.c/(1.4e-3)
            self.fwhm = 6.3
            self.MS = 39./atmFactor
            self.arrayTable = tablePath.joinpath("a1400_prop.ecsv")
        elif(band==2.0):
            self.nu = cnst.c/(2.0e-3)
            self.fwhm = 10.
            self.MS = 94./atmFactor
            self.arrayTable = tablePath.joinpath("a2000_prop.ecsv")
        else:
            print("band must be 1.1, 1.4, or 2.0")


    #time to map to a given depth 
    def time_hours(self, depth_mJy, area_deg2):
        """Returns time in Hours
           Inputs: depth in mJy, area in deg^2"""
        return area_deg2/self.MS/(depth_mJy**2)

    #area one could map to a given depth in a given time
    def area_deg2(self, depth_mJy, time_hours):
        return time_hours*self.MS*(depth_mJy**2)
    
    #1-sigma depth in given time over given area
    def depth_mJy(self, area_deg2, time_hours):
        return np.sqrt(area_deg2/(time_hours*self.MS))

    #time to map given y with depth signamPerPixel at 150GHz (2mm)
    def time_tSZ_mins(self,y,area_arcmin2,sigmaPerPix):
        if(self.band != 2.0):
            print("This function only relevant for 2.0mm band.")
            return 0
        else:
            return 138.*self.atmFactor/7.*\
                area_arcmin2/(2.*self.FOV_arcmin2)*\
                (self.ynorm/y)**2*\
                (sigmaPerPix/5.)**2


    #time to map given v with depth sigmaPerPix at 220GHz (1.4mm)
    def time_kSZ_hours(self,v,area_arcmin2,taue,sigmaPerPix):
        if(self.band != 1.4):
            print("This function only relevant for 1.4mm band.")
            return 0
        else:
            return 115.*self.atmFactor/7.*\
                area_arcmin2/(2.*self.FOV_arcmin2)*\
                (self.taunorm/taue)**2 *\
                (self.vnorm/v)**2 *\
                (sigmaPerPix/2.)**2


    #the 1-sigma depth in y (2.0mm only)
    def depth_y(self, area_arcmin2, time_hours):
        if(self.band != 2.0):
            print("This function only relevant for 2.0mm band.")
            return 0
        else:
            return self.ynorm*np.sqrt(138.*self.atmFactor/7.*\
                                      area_arcmin2/(2.*self.FOV_arcmin2)*\
                                      (1./(time_hours*60.))*\
                                      (1/5.)**2)


    #Compton y to K conversion
    def y2K(self, y):
        #dimensionless frequency
        x = cnst.h*self.nu/(cnst.k*2.726)
        #the thermal SZ eq. from Carlstrom, Holder and Reese 2002
        #Eq. 4
        term2 = x*(np.exp(x)+1.)/(np.exp(x)-1.)-4.
        return y*term2*2.726

    
    #Jy to Kelvin conversion for single polarization
    #checked with values from Bryan et al. 2018
    def Jy2K(self, fluxJy):
        fwhm = np.deg2rad(self.fwhm/3600.)
        sigma = fwhm2sigma(fwhm)
        omega = 2.*np.pi*sigma**2
        dBdT1 = dBdT(self.nu)
        jypk = omega*dBdT1*2.*1.e26 #note extra 2 for Jy def.
        return fluxJy/jypk

    #Kelvin to Jy conversion taking values from Bryan et al. 2018
    def K2Jy(self, T):
        kpj = self.Jy2K(1.)
        return T/kpj
    

    #make a map of instrument noise
    #nPixX - map size in pixels in x-coordinate
    #nPixY - map size in pixels in y-coordinate
    #pixSizeArcsec - pixel size in arcseconds
    #depth_mJy - the map depth in mJy/beam
    def noiseMap_mJy(self, nPixX, nPixY, pixSizeArcsec, depth_mJy):
        sigma = fwhm2sigma(self.fwhm)
        white = np.random.normal(0.,1.,(nPixX,nPixY))
        #convolve this with the PSF
        sigmaPix = sigma/pixSizeArcsec
        filtered = ndimage.gaussian_filter(white,sigma=sigmaPix)
        filtered *= depth_mJy/filtered.std()
        return filtered


    #return the theta_x, theta_y values of the detector locations
    #given 4' field of view of LMT
    #output values in arcseconds
    def getArrayLocArcsec(self):
        t = Table.read(self.arrayTable)
        x = t['x']
        y = t['y']
        sf = 4.*60./(x.max()-x.min())
        thetaX = x*sf
        thetaY = y*sf
        return thetaX, thetaY

    
    #make a list of Plotly shapes for the footprint
    def makeArrayShapes(self, color="LightSeaGreen"):
        tx, ty = self.getArrayLocArcsec()
        tx /= 60.
        ty /= 60.
        r = self.fwhm/2./60.
        shapes = []
        for i in np.arange(len(tx)):
            shapes.append(
                dict(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=tx[i]-r,
                    y0=ty[i]-r,
                    x1=tx[i]+r,
                    y1=ty[i]+r,
                    line_color=color,
                ))
        return shapes
        

            
#convert from FWHM to sigma
def fwhm2sigma(fwhm):
    return fwhm/(2.*np.sqrt(2.*np.log(2.)))

#calculate dB/dTcmb for single polarization
def dBdT(nu):
    #constants
    h = cnst.h
    k = cnst.k
    c = cnst.c
    T = 2.726
    x = h*nu/(k*T)
    term1 = h**2*nu**4/(c**2*k*T**2)
    term2 = np.exp(x)/((np.exp(x)-1)**2)
    return term1*term2

#the same but under the RJ approximation
def dBdTRJ(nu):
    #constants
    h = cnst.h
    k = cnst.k
    c = cnst.c
    T = 2.726
    return nu**2*cnst.k/cnst.c**2


        
