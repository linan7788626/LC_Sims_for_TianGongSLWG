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
import logging
import time
import os, sys

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
cosdir = dirname + "/data/COSMOS_25.2_training_sample"
cosCatn = "real_galaxy_catalog_25.2.fits"

# Define some parameters for the simulation
# These parameters are from Euclid science book.
# For more convenient, these parameters can be stored in a configuration file
pixelScale = 0.1             # pixel scale        [aresec/pixel]
xsize, ysize = 4096, 4096    # ccd pixel size     [pixel]
telesDiam = 1.2              # telescope diameter [meter]
expTime = 590.0              # exposure time      [second]
fwhmPSF = 0.2                # psf size           [arcsec]

# simulation setting
denGal = 30.0                 # galaxy density     [/arcmin^2]
raCen, decCen = 0.0, 0.0      # Ra&Dec center of the simulated image [deg]
noiseVariance = 150.0         # Gaussian noise variance [ADU^2]
outImgName = "testCosmosImg.fits"

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
g1, g2, mu = 0.0, 0.0, 1.0

# Loop on every galaxy
nS = str(ngal)
for k in range(ngal):
    t1 = time.time()
    ud = galsim.UniformDeviate(randomSeed+k+1)

    # generate a random position in the image
    x = ud()*(xsize-1)
    y = ud()*(ysize-1)
    imagePos = galsim.PositionD(x,y)
    worldPos = affine.toWorld(imagePos)

    # Next determine which index in the catalog we will use for this object.
    index = int(ud() * cosmosCat.nobjects)
    gal = cosmosCat.makeGalaxy(index=index, gal_type='real', rng=ud, noise_pad_size=11.3)

    # Apply a random rotation
    theta = ud()*2.0*np.pi*galsim.radians
    gal = gal.rotate(theta)

    gal *= fluxScaling
    gal  = gal.lens(g1, g2, mu)

    # Convolve with the PSF.
    final = galsim.Convolve(psf, gal)

    # Account for the fractional part of the position
    xNominal = imagePos.x + 0.5
    yNominal = imagePos.y + 0.5
    ixNominal = int(np.floor(xNominal+0.5))
    iyNominal = int(np.floor(yNominal+0.5))
    dx = xNominal - ixNominal
    dy = yNominal - iyNominal
    offset = galsim.PositionD(dx,dy)

    # draw the galaxy to real image
    stamp = final.drawImage(wcs=wcs.local(imagePos), offset=offset)
    #stamp = final.drawImage(wcs=wcs.local(imagePos), offset=offset, method='no_pixel')
    stamp.setCenter(ixNominal,iyNominal)
    bounds = stamp.bounds & fullImage.bounds

    newVariance = stamp.whitenNoise(final.noise)
    noiseImage[bounds] += newVariance

    # Finally, add the stamp to the full image.
    fullImage[bounds] += stamp[bounds]

    t2 = time.time()
    tgal = t2 - t1

    xx = "%"+str(len(nS))+"d"
    inS = xx%(k+1)
    logger.info("    GalID %s/%s: draw COSMOS galaxy %6d; %5.2fs are used", inS,nS,index,tgal)

logger.info("")
# add noise
logger.info("    Add noise to final large image")
maxCurrentVariance = np.max(noiseImage.array)
noiseImage = maxCurrentVariance - noiseImage
vn = galsim.VariableGaussianNoise(rng, noiseImage)
fullImage.addNoise(vn)

noiseVariance -= maxCurrentVariance
noise = galsim.GaussianNoise(rng, sigma=np.sqrt(noiseVariance))
fullImage.addNoise(noise)

# write the image to real world
fullImage.write(outImgName)
logger.info("    Write image to %s", outImgName)

t3 = time.time()
tt = (t3 - t0)/60.0
logger.info("^_^ Total %5.2f minutes are elapsed.", tt)

