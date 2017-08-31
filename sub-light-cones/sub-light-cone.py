# Part of TianGongSLWG
""""
Routine aimed to extract real galaxy images from COSMOS catalog and
randomly locate them to a user-specified sky region.

To make this routine work, you need to install Galsim package:
https://github.com/GalSim-developers/GalSim

And also you need to download the COSMOS catalog:
COSMOS_25.2_training_sample.tar.gz or
COSMOS_23.5_training_sample.tar.gz
Link: http://great3.jb.man.ac.uk/leaderboard/data

This routine is a highly simplified version of TianGong image simulation

-- Liu Dezi, 2017/08/22
In this routine, we assume a constant Gaussian PSF model and null shear
"""

import galsim
import numpy as np
import pylab as pl
import logging
import time
import os, sys
from astropy.table import Table

# the working directory
dirname, filename = os.path.split(os.path.abspath(__file__))

rxkey = filename[:-3]
loggn = rxkey + ".log"
logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logFile = logging.FileHandler(loggn)
logFile.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger(rxkey).addHandler(logFile)
logger = logging.getLogger(rxkey)

# COSMOS catalog directory
cosdir = "/Users/uranus/Desktop/COSMOS_GALS_DATABASE/COSMOS_25.2_training_sample"
cosCatn = "real_galaxy_catalog_25.2.fits"

# Define some parameters for the simulation
# These parameters are from Euclid science book.
# For more convenient, these parameters can be stored in a configuration file
pixelScale = 0.1             # pixel scale        [aresec/pixel]
xsize, ysize = 3600, 3600    # ccd pixel size     [pixel]
telesDiam = 1.2              # telescope diameter [meter]
expTime = 590.0              # exposure time      [second]
fwhmPSF = 0.2                # psf size           [arcsec]

# simulation setting
# denGal = 30.0                 # galaxy density     [/arcmin^2]
denGal = 1.0                 # galaxy density     [/arcmin^2]
raCen, decCen = 0.0, 0.0      # Ra&Dec center of the simulated image [deg]
noiseVariance = 150.0         # Gaussian noise variance [ADU^2]
outImgName = "TianGongCosmosImg.fits"

randomSeed = 11111

# basically, the following two parameters can be fixed
hst_eff_area = 2.4**2.0*(1.-0.33**2)
fluxScaling = (telesDiam**2/hst_eff_area) * expTime

logger.info("^_^ Working directory: %s", dirname)
logger.info("^_^ Start the simulation ...")

## start time
t0 = time.time()

# number of galaxies to be simulated
skySize = xsize*ysize*pixelScale**2/3600.0
ngal = int(denGal*skySize)
logger.info("    Total %d galaxies to be simulated.", ngal)

# Read in galaxy catalog
cosmosCat = galsim.COSMOSCatalog(cosCatn, dir=cosdir)
logger.info("    Import %d galaxies from COSMOS catalog", cosmosCat.nobjects)

gals_ids = cosmosCat.__dict__['param_cat']['IDENT']
gals_mags = cosmosCat.__dict__['param_cat']['mag_auto']
gals_flux = cosmosCat.__dict__['param_cat']['flux']
gals_zphot = cosmosCat.__dict__['param_cat']['zphot']

# Setup the image:
fullImage = galsim.ImageF(xsize, ysize)
fullImage.setOrigin(0,0)
rng = galsim.BaseDeviate(randomSeed)

noiseImage = galsim.ImageF(xsize, ysize)
noiseImage.setOrigin(0,0)

# image projection
theta = 0.0 * galsim.degrees
dudx =  np.cos(theta.rad()) * pixelScale
dudy = -np.sin(theta.rad()) * pixelScale
dvdx =  np.sin(theta.rad()) * pixelScale
dvdy =  np.cos(theta.rad()) * pixelScale
imageCenter = fullImage.trueCenter()
affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=imageCenter)

skyCenter = galsim.CelestialCoord(ra=raCen*galsim.degrees, dec=decCen*galsim.degrees)
wcs = galsim.TanWCS(affine, skyCenter, units=galsim.arcsec)
fullImage.wcs = wcs

# assume a constant Gaussian PSF and null shear for simulated galaxies
psf = galsim.Gaussian(flux=1., fwhm=fwhmPSF)
# psf = galsim.TopHat(flux=1., radius=0.2)
# g1, g2, mu = 0.0, 0.0, 1.0

# Loop on every galaxy
nS = str(ngal)

for k in range(5):
    ud = galsim.UniformDeviate(randomSeed+k+1)

    # generate a random position in the image
    x = ud()*(xsize-1)
    y = ud()*(ysize-1)
    imagePos = galsim.PositionD(x,y)
    worldPos = affine.toWorld(imagePos)

    # Next determine which index in the catalog we will use for this object.
    index = int(ud() * cosmosCat.nobjects)
    gal_tmp = cosmosCat.makeGalaxy(index=index, gal_type='real', rng=ud, noise_pad_size=5)

    gal = gal_tmp

    # Apply a random flipping
    flips = ud()
    if flips > 0.5:
        gal.gal_image = galsim.Image(np.fliplr(gal_tmp.gal_image.array),dtype=np.float32)

    # # Apply a random rotation
    # theta = ud()*2.0*np.pi*galsim.radians
    # gal = gal.rotate(theta)

    # stamp = gal_tmp.drawImage(method='no_pixel')

    pl.figure()
    pl.contourf(gal.gal_image.array)
    pl.colorbar()

    print k, index, gals_mags[index], gals_flux[index][0], gals_zphot[index]

pl.show()
