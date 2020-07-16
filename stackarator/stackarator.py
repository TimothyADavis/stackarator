import numpy as np
import scipy.interpolate as interpolate
from astropy.io import fits
from stackarator.dist_ellipse import dist_ellipse
import astropy.units as u
from tqdm import tqdm
    
class stackarator:
    
    def __init__(self,cube,rms=None):
        self.vsys=None
        self.moment1=None
        self.mom1_interpol_func = None
        self.rms=rms
        self.badvel=-10000
        
        
        ### read in cube ###
        hdulist=fits.open(cube)
        self.cubehdr=hdulist[0].header
        self.datacube = hdulist[0].data.T
        self.region=np.ones(self.datacube.shape[0:2])
        
        
        try:
            self.bmaj=self.cubehdr['BMAJ']*3600.
            self.bmin=self.cubehdr['BMIN']*3600.
            self.bpa=self.cubehdr['BPA']*3600.
        except:
            self.bmaj=np.median(hdulist[1].data['BMAJ'])
            self.bmin=np.median(hdulist[1].data['BMIN'])
            self.bpa=np.median(hdulist[1].data['BPA'])
            
        self.xcoord,self.ycoord,self.vcoord = self.get_header_coord_arrays(self.cubehdr,"cube")
        self.dv=np.median(np.diff(self.vcoord))
        if self.dv < 0:
            self.datacube = np.flip(hdulist[0].data.T,axis=2)
            self.dv*=(-1)
            self.vcoord = np.flip(self.vcoord)

        if self.rms == None:
            quarterx=np.array(self.xcoord.size/4.).astype(np.int)
            quartery=np.array(self.ycoord.size/4.).astype(np.int)
            self.rms=np.nanstd(self.datacube[quarterx*1:3*quarterx,1*quartery:3*quartery,2:5])
            print("Estimated RMS from channels 2-5:",self.rms)

        self.datacube[~np.isfinite(self.datacube)]=0.0
        self.beam_fac=self.makebeam(self.xcoord.size, self.ycoord.size, [self.bmaj,self.bmin,self.bpa], cellSize=self.cubehdr['CDELT2']*3600).sum()
        #
        # convert to j/kms? or to kelvin? or just use whats put in...
        # if self.cubehdr['BUNIT'] == 'K':
        #     fwhm_to_sigma = 1./(8*np.log(2))**0.5
        #     beam_area = 2.*np.pi*(self.bmaj*u.arcsec*u.arcsec*self.bmin*fwhm_to_sigma**2)
        #     freq = self.cubehdr['RESTFRQ']*u.Hz
        #     equiv = u.brightness_temperature(freq)
        #     self.jy_to_k=(u.Jy/beam_area).to(u.K, equivalencies=equiv)
        #     self.datacube/=self.jy_to_k.value
        #     self.rms/=self.jy_to_k.value
        #     self.cubehdr['BUNIT'] = 'Jy/beam'
        


            

            
        
    def get_header_coord_arrays(self,hdr,cube_or_mom):
        try:
            cd1=hdr['CDELT1']
            cd2=hdr['CDELT2'] 
            
        except:
            cd1=hdr['CD1_1']
            cd2=hdr['CD2_2']        
        
        x1=((np.arange(0,hdr['NAXIS1'])-hdr['CRPIX1'])*cd1) + hdr['CRVAL1']
        y1=((np.arange(0,hdr['NAXIS2'])-hdr['CRPIX2'])*cd2) + hdr['CRVAL2']
        
        if cube_or_mom == "cube":
            try:    
                cd3=hdr['CDELT3']
            except:    
                cd3=hdr['CD3_3']
                
            if hdr['CTYPE3'] =='VRAD':     
                v1=((np.arange(0,hdr['NAXIS3'])-hdr['CRPIX3'])*cd3) + hdr['CRVAL3']
                if hdr['CUNIT3']=='m/s':
                    v1/=1e3
                    
            else:
                f1=(((np.arange(0,hdr['NAXIS3'])-hdr['CRPIX3'])*cd3) + hdr['CRVAL3'])*u.Hz
                restfreq = hdr['RESTFRQ']*u.Hz  # rest frequency of 12 CO 1-0 in GHz   
                v1=f1.to(u.km/u.s, equivalencies=u.doppler_radio(restfreq))
                v1=v1.value
                  
            return x1,y1,v1
            
        return x1,y1
            
    def makebeam(self, xpixels, ypixels, beamSize, cellSize=1, cent=None):
        """
        Creates a psf with which one can convolve their cube based on the beam provided.
        
        :param xpixels:
                (float or int) Number of pixels in the x-axis
        :param ypixels:
                (float or int) Number of pixels in the y-axis
        :param beamSize:
                (float or int, or list or array of float or int) Scalar or three element list for size of convolving beam (in arcseconds). If a scalar then beam is
                assumed to be circular. If a list/array of length two. these are the sizes of the major and minor axes,
                and the position angle is assumed to be 0. If a list/array of length 3, the first 2 elements are the
                major and minor beam sizes, and the last the position angle (i.e. [bmaj, bmin, bpa]).
        :param cellSize:
                (float or int) Pixel size required (arcsec/pixel)
        :param cent: 
            (array or list of float or int) Optional, default value is [xpixels / 2, ypixels / 2].
                Central location of the beam in units of pixels.
        :return psf or trimmed_psf:
                (float array) psf required for convlution in self.model_cube(). trimmed_psf returned if self.huge_beam=False, 
                otherwise default return is the untrimmed psf.              
        """

        if not cent: cent = [xpixels / 2, ypixels / 2]

        beamSize = np.array(beamSize)

        try:
            if len(beamSize) == 2:
                beamSize = np.append(beamSize, 0)
            if beamSize[1] > beamSize[0]:
                beamSize[1], beamSize[0] = beamSize[0], beamSize[1]
            if beamSize[2] >= 180:
                beamSize[2] -= 180
        except:
            beamSize = np.array([beamSize, beamSize, 0])

        st_dev = beamSize[0:2] / cellSize / 2.355

        rot = beamSize[2]

        if np.tan(np.radians(rot)) == 0:
            dirfac = 1
        else:
            dirfac = np.sign(np.tan(np.radians(rot)))

        x, y = np.indices((int(xpixels), int(ypixels)), dtype='float')

        x -= cent[0]
        y -= cent[1]

        a = (np.cos(np.radians(rot)) ** 2) / (2 * st_dev[1] ** 2) + (np.sin(np.radians(rot)) ** 2) / \
            (2 * (st_dev[0] ** 2))

        b = (dirfac * (np.sin(2 * np.radians(rot)) ** 2) / (4 * st_dev[1] ** 2)) + ((-1 * dirfac) * \
            (np.sin(2 * np.radians(rot)) ** 2) / (4 * st_dev[0] ** 2))

        c = (np.sin(np.radians(rot)) ** 2) / (2 * st_dev[1] ** 2) + (np.cos(np.radians(rot)) ** 2) / \
            (2 * st_dev[0] ** 2)

        psf = np.exp(-1 * (a * x ** 2 - 2 * b * (x * y) + c * y ** 2))

        return psf
        
        
    def read_mom1(self,mom1,vsys=0):
        hdulist1=fits.open(mom1)
        self.vsys=vsys
        if self.vsys == 0:
            try:
                self.vsys=hdulist1[0].header['SYSVEL']
            except:
                self.vsys=0
                
        self.moment1 = hdulist1[0].data + self.vsys
        self.moment1[~np.isfinite(self.moment1)]=(-10000)
        x1,y1 = self.get_header_coord_arrays(hdulist1[0].header,"mom")
        self.mom1_interpol_func = interpolate.interp2d(x1, y1, self.moment1, fill_value=self.badvel)
        
        
    def define_region_ellipse(self,gal_centre,inc,pa,rad_inner=0,rad_outer=np.inf):
        self.region=np.zeros(self.datacube.shape[0:2])
        xc,=np.where(np.abs(self.xcoord-gal_centre[0]) == np.min(np.abs(self.xcoord-gal_centre[0])))
        yc,=np.where(np.abs(self.ycoord-gal_centre[1]) == np.min(np.abs(self.ycoord-gal_centre[1])))
        distim=dist_ellipse(self.datacube.shape[0:2], xc[0], yc[0], 1/np.cos(np.deg2rad(inc)), pa=pa)*np.abs(np.median(np.diff(self.xcoord))*3600.)
        self.region[(distim >= rad_inner)&(distim<rad_outer)] = 1
        
    def stack(self):
        spec=np.zeros(3*self.vcoord.size)
        num=np.zeros(3*self.vcoord.size)
         
        
        
        vout=((np.arange(0,3*self.vcoord.size)-(1.5*self.vcoord.size))*self.dv) 
        
        x,y=np.where((self.region == 1))     

        for i in  tqdm(range(0,x.size)):
            vcent=self.mom1_interpol_func(self.xcoord[x[i]],self.ycoord[y[i]])
            if vcent != self.badvel:
                # f=interpolate.interp1d(self.vcoord-vcent, self.datacube[x[i],y[i],:],fill_value=0, kind = 'nearest',bounds_error=False)
                
                spec+=np.interp(vout, self.vcoord-vcent, self.datacube[x[i],y[i],:],left=0,right=0)
                num+=np.interp(vout, self.vcoord-vcent, np.ones(self.vcoord.size),left=0,right=0)

        outspec=spec[num >= 1]
        outn=num[num >= 1]
        outspec=outspec#/outn
        outrms=self.rms*np.sqrt(outn)
        return vout[num >= 1],outspec,outrms,outn
        
    

