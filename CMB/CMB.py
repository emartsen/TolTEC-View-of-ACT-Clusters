import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import astropy.io.fits as fits
import matplotlib.cm as cm
import numpy as np
import matplotlib
import math
import sys
plt.ion()

from pathlib import Path

#this class is based on Jeff McMahon's CMB summer school github code
class CMB:

    #instantiate
    def __init__(self, nPixX, nPixY, pixSizeArcsec,
                 pathToClFile=Path(__file__).parent,
                 clFile = "CAMB_fiducial_cosmo_scalCls.dat"):
        
        #grab the c_ls
        self.ell, self.DlTT = np.loadtxt(pathToClFile.joinpath(clFile), usecols=(0, 1), unpack=True)

        #the size of the map to be generated, make it a power of 2 of max dim
        nPix = max(nPixX, nPixY)
        pos = np.ceil(math.log(nPix,2))
        self.nPix = int(2**pos)
        self.nPixX = nPixX
        self.nPixY = nPixY
        self.pixSizeArcmin = pixSizeArcsec/60. 
        self.pixSizeArcsec = pixSizeArcsec

        #make the big map and slice it to get returned map
        bigMap = self.make_CMB_T_map()
        self.CMB_T = bigMap[int(nPix/2)-int(nPixX/2):int(nPix/2)-int(nPixX/2)+nPixX,
                            int(nPix/2)-int(nPixY/2):int(nPix/2)-int(nPixY/2)+nPixY]

        

    #makes a realization of a simulated CMB sky map given an input
    #DlTT as a function of ell, the pixel size (pix_size) required and
    #the number N of pixels in the linear dimension.
    def make_CMB_T_map(self):
        N = self.nPix
        pix_size = self.pixSizeArcmin
        ell = self.ell
        DlTT = self.DlTT
        
        # convert Dl to Cl
        ClTT = DlTT * 2 * np.pi / (ell*(ell+1.))
        ClTT[0] = 0. # set the monopole and the dipole of the Cl spectrum to zero
        ClTT[1] = 0.
            
        # make a 2D real space coordinate system
        onesvec = np.ones(N)
        inds  = (np.arange(N)+.5 - N/2.) /(N-1.)
            
        # compute the outer product matrix: X[i, j] = onesvec[i] * inds[j] for i,j 
        # in range(N), which is just N rows copies of inds - for the x dimension
        X = np.outer(onesvec,inds) 
        Y = np.transpose(X)
        # radial component R
        R = np.sqrt(X**2. + Y**2.)
    
        # now make a 2D CMB power spectrum
        pix_to_rad = (pix_size/60. * np.pi/180.) 
        ell_scale_factor = 2. * np.pi /pix_to_rad  
        ell2d = R * ell_scale_factor 
        ClTT_expanded = np.zeros(int(ell2d.max())+1) 
        ClTT_expanded[0:(ClTT.size)] = ClTT 

        # the 2D Cl spectrum is defined on the multiple vector set by the pixel scale
        CLTT2d = ClTT_expanded[ell2d.astype(int)] 
            
        # now make a realization of the CMB with the given power spectrum in real space
        random_array_for_T = np.random.normal(0,1,(N,N))
        FT_random_array_for_T = np.fft.fft2(random_array_for_T)    
        FT_2d = np.sqrt(CLTT2d) * FT_random_array_for_T 
    
        # move back from ell space to real space
        CMB_T = np.fft.ifft2(np.fft.fftshift(FT_2d)) 
        # move back to pixel space for the map
        CMB_T = CMB_T/(pix_size /60.* np.pi/180.)
        # we only want to plot the real component
        CMB_T = np.real(CMB_T)

        ## return the map
        return CMB_T


    def plotMap(self):
        Map_to_Plot = self.CMB_T
        c_min = self.CMB_T.min()
        c_max = self.CMB_T.max()
        X_width = self.nPixX*self.pixSizeArcmin
        Y_width = self.nPixY*self.pixSizeArcmin
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        print("map mean:",np.mean(Map_to_Plot),"map rms:",np.std(Map_to_Plot))
        plt.ion()
        plt.clf()
        im = plt.imshow(Map_to_Plot, interpolation='bilinear', origin='lower',cmap=cm.RdBu_r)
        im.set_clim(c_min,c_max)
        ax=plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = plt.colorbar(im, cax=cax)
        im.set_extent([0,X_width,0,Y_width])
        plt.ylabel('angle [arcmins]')
        plt.xlabel('angle [arcmins]')
        cbar.set_label('tempearture [uK]', rotation=270)
        plt.show()

