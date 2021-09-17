from scipy.interpolate import interp1d
import scipy.integrate as integrate
import scipy.ndimage as ndimage
import numpy as np


#this is a class designed to draw from a distribution of dusty star
#forming galaxies and produce a catalog or a map of sources

class DSFGs:

    #areaDeg2 - area of map to conisder [sq. degrees]
    #minFluxmJy - minimum flux of distribution to draw from
    #beta - spectral index of dust
    #instantiate
    def __init__(self, areaDeg2, minFluxmJy=0.01, beta=1.5):
        self.areaDeg2 = areaDeg2
        self.minFluxmJy = minFluxmJy
        self.beta = beta
        self.Ntot = int(np.floor(self.Ngreater(self.minFluxmJy)))
        #draw Ntot 1.1mm source fluxes
        self.S1p1 = self.drawFromdNdS(self.Ntot)
        self.S1p4 = self.S1p1*(1.1/1.4)**(self.beta+2.)
        self.S2p0 = self.S1p1*(1.1/2.0)**(self.beta+2.)
        

    #Schechter funtion from Scott et al. 2011
    #S - flux in mJy
    #N3 - cnst [mJy^-1 deg^-2]
    #Sp - cnst [mJy]
    #alpha - cnst
    def dNdS(self, S, N3=230., Sp=1.7, alpha=-2):
        term1 = (S/3.)**(alpha+1)
        term2 = np.exp(-(S-3.)/Sp)
        return N3*term1*term2

    
    #calculate number of sources greater than S, capped at 20mJy
    def Ngreater(self, S):
        I = integrate.quad(self.dNdS, S, 20.)[0]
        return self.areaDeg2*I

    #calculate number of sources less than S, down to minFluxmJy
    def Nless(self, S):
        I = integrate.quad(self.dNdS, self.minFluxmJy, S)[0]
        return self.areaDeg2*I


    #draw N sources from dNdS
    def drawFromdNdS(self, N):
        npts = 500
        S = np.linspace(self.minFluxmJy, 20., npts)
        I = np.zeros(npts,dtype='float')
        for i in np.arange(1,npts):
            I[i] = self.Nless(S[i])
        #normalize I to turn it into a CDF
        I = I/I.max()
        #build reverse interpolation
        R = interp1d(I,S,kind='quadratic')
        #draw uniform deviates and use R to get fluxes of sources
        u = np.random.uniform(size=N)
        return R(u)


    #generate a point source (delta functions) map
    #nPixX - map size in pixels in x-coordinate
    #nPixY - map size in pixels in y-coordinate
    #pixSizeArcsec - pixel size in arcseconds
    #returned image has amplitudes corresponding to S1p1 fluxes
    def pointSourceMapCoords(self, nPixX, nPixY):
        ux = np.random.random_integers(0, nPixX-1, size=self.Ntot)
        uy = np.random.random_integers(0, nPixY-1, size=self.Ntot)
        return ux, uy
    
    #make a map of sources for each of TolTEC's three bands
    #nPixX - map size in pixels in x-coordinate
    #nPixY - map size in pixels in y-coordinate
    #pixSizeArcsec - pixel size in arcseconds
    def dsfgMap_mJy(self, nPixX, nPixY, pixSizeArcsec):
        fwhm = np.array([5., 6.5, 10.])
        sigma = fwhm/(2.*np.sqrt(2.*np.log(2.)))
        sigmaPix = sigma/pixSizeArcsec
        #generate point source map
        ux,uy = self.pointSourceMapCoords(nPixX, nPixY)
        img1p1 = np.zeros((nPixX,nPixY),dtype='float')
        for i in np.arange(self.Ntot):
            img1p1[ux[i],uy[i]] = self.S1p1[i]
        img1p4 = img1p1*(1.1/1.4)**(self.beta+2.)
        img2p0 = img1p1*(1.1/2.0)**(self.beta+2.)
        #convolve with the PSF
        img1p1 = ndimage.gaussian_filter(img1p1,sigma=sigmaPix[0])
        img1p4 = ndimage.gaussian_filter(img1p4,sigma=sigmaPix[1])
        img2p0 = ndimage.gaussian_filter(img2p0,sigma=sigmaPix[2])
        return img1p1, img1p4, img2p0
