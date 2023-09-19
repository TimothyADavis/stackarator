import warnings

import astropy.units as u
import numpy as np
import scipy.interpolate as interpolate
from astropy.io import fits
from spectral_cube import SpectralCube
from spectral_cube.utils import SpectralCubeWarning, NoBeamError
from stackarator.dist_ellipse import dist_ellipse
from tqdm import tqdm

warnings.filterwarnings(action="ignore", category=SpectralCubeWarning, append=True)

class stackarator:
    def __init__(self):
        self.vsys = None
        self.moment1 = None
        self.mom1_interpol_func = None
        self.switch_to_flat=80. ## switch to flat w/radius after this many degrees
        self.rms = None
        self.badvel = -10000
        self.datacube = None
        self.region = None
        self.bmaj = None
        self.bmin = None
        self.bpa = None
        self.xcoord, self.ycoord, self.vcoord = None, None, None
        self.dv = None
        self.cellsize = None
        self.silent = False  # rig for silent running if true
        self.rmsimg=None
        self.distimg = None
        
    def input_cube(self, cube, xcoord, ycoord, vcoord, rms=None):
        self.datacube = cube
        self.region = np.ones(self.datacube.shape[0:2])
        self.datacube[~np.isfinite(self.datacube)] = 0.0
        self.xcoord, self.ycoord, self.vcoord = xcoord, ycoord, vcoord
        self.dv = np.median(np.diff(self.vcoord))
        self.cellsize = np.abs(np.median(np.diff(self.xcoord))) * 3600.0
        self.rms = rms
        if self.dv < 0:
            self.dv *= -1
            self.vcoord = np.flip(self.vcoord)
            self.datacube = np.flip(hdulist[0].data.T, axis=2)
        if self.rms == None:
            self.rms_estimate()

    def rms_estimate(self):
        quarterx = np.array(self.xcoord.size / 4.0).astype(int)
        quartery = np.array(self.ycoord.size / 4.0).astype(int)
        (wnotzero,) = np.where(np.nansum(np.nansum(self.datacube, axis=0), axis=0) > 0)
        self.rms = np.nanstd(
            self.datacube[
                quarterx * 1 : 3 * quarterx,
                1 * quartery : 3 * quartery,
                wnotzero[2] : wnotzero[5],
            ]
        )
        if not self.silent:
            print("Estimated RMS from channels 2-5:", self.rms)

    def read_fits_cube(self, cube, pbcube=None, rms=None, velocity_convention="radio"):
        ### read in cube ###
        # hdulist=fits.open(cube)
        # hdr=hdulist[0].header
        # self.datacube = np.squeeze(hdulist[0].data.T) #squeeze to remove singular stokes axis if present
        self.spectralcube = SpectralCube.read(cube).with_spectral_unit(
            u.km / u.s, velocity_convention=velocity_convention
        )  # .to(u.Jy/u.beam) #, rest_value=self.restfreq)


        hdr = self.spectralcube.header

        self.datacube = np.squeeze(
            self.spectralcube.filled_data[:, :, :].T
        ).value  # squeeze to remove singular stokes axis if present
        # cube[np.isfinite(cube) == False] = 0.0

        self.region = np.ones(self.datacube.shape[0:2])
        self.rms = rms
        
        # only save beam info to attr if it exists in cube file
        try:
            hasbeam = hasattr(self.spectralcube, "beam")
        except NoBeamError as e:
            print(e)
            hasbeam = False
        
        if not hasbeam: # allow for varying resolution (multi-beam) cubes
            hasbeam = hasattr(self.spectralcube, "beams")
        
        if hasbeam:
        
            self.bunit = self.spectralcube.unit.to_string()
        

            try:
                beamtab = self.spectralcube.beam
            except:
                beamtab = self.spectralcube.beams[
                    np.floor(self.spectralcube.beams.size / 2).astype(int)
                ]

            self.bmaj = beamtab.major.to(u.arcsec).value
            self.bmin = beamtab.minor.to(u.arcsec).value
            self.bpa = beamtab.pa.to(u.degree).value

        # try:
        #    self.bmaj=hdr['BMAJ']*3600.
        #    self.bmin=hdr['BMIN']*3600.
        #    self.bpa=hdr['BPA']*3600.
        # except:
        #    self.bmaj=np.median(hdulist[1].data['BMAJ'])
        #    self.bmin=np.median(hdulist[1].data['BMIN'])
        #    self.bpa=np.median(hdulist[1].data['BPA'])

        (
            self.xcoord,
            self.ycoord,
            _,
            self.cellsize,
            self.dv,
        ) = self.get_header_coord_arrays(
            hdr, "cube", velocity_convention=velocity_convention
        )

        self.vcoord = self.spectralcube.spectral_axis.value

        if self.dv < 0:
            self.datacube = np.flip(self.datacube, axis=2)
            self.dv *= -1
            self.vcoord = np.flip(self.vcoord)

        if self.rms == None:
            self.rms_estimate()

        self.datacube[~np.isfinite(self.datacube)] = 0.0

        if pbcube != None:
            pbhdulist = fits.open(pbcube)
            self.rmsimg = self.rms / np.median(np.squeeze(pbhdulist[0].data.T), 2)
        else:
            self.rmsimg = self.rms / np.ones(self.datacube.shape[0:2])
        self.rmsimg[np.isfinite(self.rmsimg) == 0] = self.rms

    def get_header_coord_arrays(self, hdr, cube_or_mom, velocity_convention="radio"):
        try:
            cd1 = hdr["CDELT1"]
            cd2 = hdr["CDELT2"]

        except:
            cd1 = hdr["CD1_1"]
            cd2 = hdr["CD2_2"]

        x1 = ((np.arange(0, hdr["NAXIS1"]) - (hdr["CRPIX1"] - 1)) * cd1) + hdr["CRVAL1"]
        y1 = ((np.arange(0, hdr["NAXIS2"]) - (hdr["CRPIX2"] - 1)) * cd2) + hdr["CRVAL2"]

        if cube_or_mom == "cube":
            try:
                cd3 = hdr["CDELT3"]
            except:
                cd3 = hdr["CD3_3"]

            if velocity_convention == "optical":
                ctype = "VOPT"
            else:
                ctype = "VRAD"

            if ctype in hdr["CTYPE3"]:  # allow more flexibility with CTYPE3
                v1 = ((np.arange(0, hdr["NAXIS3"]) - (hdr["CRPIX3"] - 1)) * cd3) + hdr[
                    "CRVAL3"
                ]
                if (hdr["CUNIT3"] == "m/s") or (hdr["CUNIT3"] == "m s-1"):
                    v1 /= 1e3
                    cd3 /= 1e3

            else:
                f1 = (
                    ((np.arange(0, hdr["NAXIS3"]) - (hdr["CRPIX3"] - 1)) * cd3)
                    + hdr["CRVAL3"]
                ) * u.Hz
                restfreq = hdr["RESTFRQ"] * u.Hz  # rest frequency of 12 CO 1-0 in GHz
                if velocity_convention == "optical":
                    equiv = u.doppler_optical(restfreq)
                else:
                    u.doppler_radio(restfreq)

                v1 = f1.to(u.km / u.s, equivalencies=equiv)
                v1 = v1.value

            return x1, y1, v1, np.abs(cd1 * 3600), cd3

        return x1, y1

    def read_fits_mom1(self, mom1, vsys=None):
        hdulist1 = fits.open(mom1)
        if vsys == None:
            try:
                vsys = hdulist1[0].header["SYSVEL"]
            except:
                vsys = 0

        mom1 = hdulist1[0].data
        if (hdulist1[0].header["BUNIT"] == "m s-1") or (
            hdulist1[0].header["BUNIT"] == "m/s"
        ):
            mom1 /= 1e3

        x1, y1 = self.get_header_coord_arrays(hdulist1[0].header, "mom")
        self.input_mom1(x1, y1, mom1, vsys=vsys)

    def input_mom1(self, x1, y1, mom1, vsys=0):
        self.vsys = vsys
        self.moment1 = mom1 + self.vsys
        self.moment1[~np.isfinite(self.moment1)] = self.badvel
        self.mom1_interpol_func = interpolate.interp2d(
            x1, y1, self.moment1, fill_value=self.badvel
        )

    def define_region_ellipse(self,gal_centre,inc,pa,rad_inner=0,rad_outer=np.inf):
        self.region=np.zeros(self.datacube.shape[0:2])
        xc,=np.where(np.abs(self.xcoord-gal_centre[0]) == np.min(np.abs(self.xcoord-gal_centre[0])))
        yc,=np.where(np.abs(self.ycoord-gal_centre[1]) == np.min(np.abs(self.ycoord-gal_centre[1])))
        
        if inc<self.switch_to_flat:
            distim=dist_ellipse(self.datacube.shape[0:2], yc[0], xc[0], 1/np.cos(np.deg2rad(inc)), pa=(90-pa)+90)*self.cellsize
        else:
            distim=self.dist_flat(self.datacube.shape[0:2],pa+90,[xc[0],yc[0]])*self.cellsize
        
        self.region[(distim >= rad_inner)&(distim<rad_outer)] = 1
        # save dist ellipse object as stack attribute
        self.distimg = distim
        
    def dist_flat(self,size,pa,cent):
        outarr = np.zeros((size[0],size[1]))
        m = np.tan(np.deg2rad(90+pa))

        for i in range(0,size[0]):
           for j in range(0,size[1]):
              outarr[i,j] = abs(((i-cent[0])-((j-cent[1])/m))*np.sin((((90+pa)*np.pi)/180)))
           
        return outarr    


    def stack(self):
        spec = np.zeros(3 * self.vcoord.size)
        num = np.zeros(3 * self.vcoord.size)
        rms = np.zeros(3 * self.vcoord.size) + self.rms

        vout = (np.arange(0, 3 * self.vcoord.size) - (1.5 * self.vcoord.size)) * self.dv

        x, y = np.where((self.region == 1))

        for i in tqdm(range(0, x.size), disable=self.silent):
            vcent = self.mom1_interpol_func(self.xcoord[x[i]], self.ycoord[y[i]])
            if vcent != self.badvel:
                newspec = np.interp(
                    vout,
                    self.vcoord - vcent,
                    self.datacube[x[i], y[i], :],
                    left=0,
                    right=0,
                )
                spec += newspec
                nadded = np.interp(
                    vout,
                    self.vcoord - vcent,
                    np.ones(self.vcoord.size),
                    left=0,
                    right=0,
                )
                num += nadded
                rms[nadded > 0] = np.sqrt(
                    rms[nadded > 0] ** 2 + self.rmsimg[x[i], y[i]] ** 2
                )

        outspec = spec[num >= 1]
        outn = num[num >= 1]
        outrms = rms[num >= 1]  # self.rms*np.sqrt(outn)
        return vout[num >= 1], outspec, outrms, outn
