# Copyright (c) 2016 by Mike Jarvis and the other collaborators on GitHub at
# https://github.com/rmjarvis/Piff  All rights reserved.
#
# Piff is free software: Redistribution and use in source and binary forms
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

import numpy as np
import pickle
import galsim
import time
import piff
import treecorr
import multiprocessing as mp
from matplotlib import pyplot as plt
from itertools import repeat
from astropy import units
from astropy.io import fits

piff_dev_dir = "./Piff/devel/input/"

decaminfo = piff.des.DECamInfo()

def get_positions(nstars, rng):
    # Avoid positions within 20 pixels of the edge.
    pixedge = 20

    # Randomly cover the DES footprint and 61/62 CCDs
    chiplist =  [1] + list(range(3,62+1))  # omit chipnum=2
    chipnum = rng.np.choice(chiplist, nstars)
    x = rng.np.uniform(1+pixedge, 2048-pixedge, nstars)
    y = rng.np.uniform(1+pixedge, 4096-pixedge, nstars)

    # Get focal plane coords in mm
    u, v = decaminfo.getPosition(chipnum, x, y)

    # Convert to arcsec
    arcsecperpixel = 0.26
    u *= arcsecperpixel / decaminfo.mmperpixel
    v *= arcsecperpixel / decaminfo.mmperpixel

    # Reorient so N is up, W is right.
    u, v = -v, -u

    return chipnum, x, y, u, v
 
def get_initial_wavefront(ref_wf, u, v, chipnum):

    # Wavefront files use the focal plane position in mm.
    arcsecperpixel = 0.26
    x_fp = -v / arcsecperpixel * decaminfo.mmperpixel
    y_fp = -u / arcsecperpixel * decaminfo.mmperpixel

    wf_arr = np.zeros(ref_wf.maxnZ+1)
    for isource in range(ref_wf.nsources):
        for iZ in ref_wf.zlists[isource]:
            chipnum1 = chipnum if ref_wf.chipkeys[isource] else None
            tab = ref_wf.interp_objects[(isource, chipnum1, iZ)]
            wf_arr[iZ] = tab(x_fp, y_fp)

    wf_arr = piff.wavefront.convert_zernikes_des(wf_arr)
    return wf_arr

def make_mirror_figure(mirror_figure_im, mirror_figure_halfsize):
    im = galsim.fits.read(mirror_figure_im)
    u = np.linspace(-mirror_figure_halfsize, mirror_figure_halfsize, num=512)
    v = np.linspace(-mirror_figure_halfsize, mirror_figure_halfsize, num=512)
    tab = galsim.LookupTable2D(u, v, im.array)
    return galsim.UserScreen(tab)

def make_profiles(chipnum, u, v, params):
    """Returns Opt+Atm profiles for these positions.
    """
    # get reference wavefront
    wavefront_kwargs = dict(
        survey='des',
        source1=dict(
           file= piff_dev_dir + "GPInterp-20140212s2-v22i2.npz",
           zlist= [4,5,6,7,8,9,10,11,14,15],
           keys= {"x_fp":"xfp","y_fp":"yfp"},
           chip= "chipnum",
           wavelength= 700.0),
        source2=dict(
           file= piff_dev_dir + "decam_2012-iband-700nm.npz",
           zlist= [12,13,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37],
           keys= {"x_fp":"xfp","y_fp":"yfp"},
           chip= "None",
           wavelength= 700.0)
        )
    ref_wf = piff.wavefront.Wavefront(wavefront_kwargs)

    fov_radius = 4500 # arcsec = 1.25 deg
    lam = 782.1  # Set in Aaron's config file

    # The following is "starby2" in gsparams_template:
    gsparams = galsim.GSParams(minimum_fft_size=64, folding_threshold=0.01)

    # This is the optical template used by Aaron's code.
    des_128 = {
            'diam': 4.010,  # meters
            'lam': 700, # nm
            'pad_factor': 1 ,
            'oversampling': 1,
            'sigma': 8.0 * (0.263/15.0), # 8micron sigma CCD diffusion
            'mirror_figure_im': piff_dev_dir + 'DECam_236392_finegrid512_nm_uv.fits',
            'mirror_figure_halfsize': 2.22246,
            'pupil_plane_im': piff_dev_dir + 'DECam_pupil_128uv.fits'
    }
    diam = des_128['diam']
    sigma = des_128['sigma']
    aperture = galsim.Aperture(diam=diam,
                               pad_factor=1, oversampling=1,
                               #pupil_plane_im=des_128['pupil_plane_im'])
                               pupil_plane_im=galsim.fits.read(des_128['pupil_plane_im']).array)
    aperture._load_pupil_plane()
    # print('scale = ',aperture._pupil_plane_scale)

    mirror_figure_screen = make_mirror_figure(des_128['mirror_figure_im'],
                                              des_128['mirror_figure_halfsize'])

    # Create the profiles
    profs = []
    nstars = len(u)
    for i in range(nstars):

        r0 = 0.15/params.get('opt_size')  # size is defined as 0.15/r0
        L0 = params.get('opt_L0')
        g1 = params.get('opt_g1')
        g2 = params.get('opt_g2')

        atm_prof = galsim.VonKarman(lam=lam, r0=r0, L0=L0, gsparams=gsparams).shear(g1=g1, g2=g2)

        # These are the order of variation for each Zernike
        # tuples are (pupil zernike number, spatial order)
        double_zernike_terms = ((4, 1), (5, 3), (6, 3), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1))

        # Convert field position from arcsec to fraction of focal radius
        ui = u[i] / fov_radius
        vi = v[i] / fov_radius

        # Start with reference wavefront for this position (u,v,chipnum)
        wf = get_initial_wavefront(ref_wf, u[i], v[i], chipnum[i])

        # add in changes to aberrations
        for iZ, nF in double_zernike_terms:
            wf[iZ] += params.get('z%df%d' % (iZ,1))
            if nF == 3:
                wf[iZ] += ui * params.get('z%df%d' % (iZ,2))
                wf[iZ] += vi * params.get('z%df%d' % (iZ,3))

        optical_screen = galsim.OpticalScreen(diam=diam, aberrations=wf, lam_0=lam)
        screens = [optical_screen, mirror_figure_screen]
        opt_prof = galsim.PhaseScreenPSF(screen_list=screens, aper=aperture, lam=lam)

        diff_prof = galsim.Gaussian(sigma=sigma, gsparams=gsparams)

        prof = galsim.Convolve(diff_prof, atm_prof, opt_prof, gsparams=gsparams)
        profs.append(prof)

    return profs

def set_params(optics_type, rng):
    """Set up the parameters for the double zernike optical PSF variation.
    """
    params = {}
    params['opt_size'] = rng.np.uniform(0.8,1.2,1)[0]
    params['opt_L0'] = rng.np.uniform(3.,10.,1)[0]
    params['opt_g1'] = 0.0
    params['opt_g2'] = 0.0

    # The labels here are the zernike number being varied and the coefficient of the variation.
    # e.g. z4f1 is the constant term for the Z4 aberration (aka defocus).
    #      z5f2 and z5f3 are the two linear terms for the Z5 aberration (astigmatism)
    params['z4f1'] = rng.np.uniform(-0.3,0.3,1)[0]
    params['z5f1'] = rng.np.uniform(-0.2,0.2,1)[0]
    params['z6f1'] = rng.np.uniform(-0.2,0.2,1)[0]
    params['z11f1'] = rng.np.uniform(-0.2,0.2,1)[0]

    if optics_type=='Nominal_with_g':
        # This mostly just adds an overall constant to all the shapes.
        params['opt_g1'] = rng.np.uniform(-0.05,0.05,1)[0]
        params['opt_g2'] = rng.np.uniform(-0.05,0.05,1)[0]

    if optics_type.startswith('Nominal'):
        for iz in range(7,10+1):
            params['z%df1' % (iz)] = rng.np.uniform(-0.2,0.2,1)[0]
        params['z5f2'] = rng.np.uniform(-0.3,0.3,1)[0]
        params['z5f3'] = rng.np.uniform(-0.3,0.3,1)[0]
        params['z6f2'] = rng.np.uniform(-0.3,0.3,1)[0]
        params['z6f3'] = rng.np.uniform(-0.3,0.3,1)[0]
    else:
        for iz in range(7,10+1):
            params['z%df1' % (iz)] = 0.
        params['z5f2'] = 0.
        params['z5f3'] = 0.
        params['z6f2'] = 0.
        params['z6f3'] = 0.

    return params

def calculate_star_shapes(seed=12345, nstars=8000, optics_type='Fast'):
    """
    Create lists of u, v, g1, g2, flag for optical + atmosphere stars.

    :param seed:          Random number seed [default: 12345]
    :param nstars:        Number of stars to make [default: 8000]
    :param optics_type:   Type of optical wavefront to generate, 'Fast' or 'Nominal' [default: Fast]
    """
    # random number generator
    rng = galsim.BaseDeviate(seed)

    # setup the random parameters for the double zernike optical variation.
    params = set_params(optics_type, rng)

    # Choose positions for stars
    chipnum, x, y, u, v = get_positions(nstars, rng)

    # Make galsim profiles of stars at these locations
    profs = make_profiles(chipnum, u, v, params)

    # Match Aaron's code.
    stamp_size = 19
    wcs = galsim.JacobianWCS(0, -0.26, -0.26, 0)

    hsm = [p.drawImage(nx=stamp_size, ny=stamp_size, center=(xx,yy),
                       wcs=wcs, dtype=float).FindAdaptiveMom(use_sky_coords=True)
           for p, xx, yy in zip(profs,x,y)]
    g1 = [h.observed_shape.g1 for h in hsm]
    g2 = [h.observed_shape.g2 for h in hsm]
    flag = [h.moments_status for h in hsm]

    return u, v, g1, g2, flag, params

def getLineParameters(xpos, ypos, gamma):
    length, angle = np.abs(gamma), np.angle(gamma)
    return [xpos - np.cos(angle)*length/2, xpos + np.cos(angle)*length/2], \
            [ypos - np.sin(angle)*length/2, ypos + np.sin(angle)*length/2]

def make_one_exposure(ind, seed=1100, out_dir="./exposures/", \
                      nstars=10000, optics_type='Nominal'):
    start = time.time()
    print(f"Making exposure {ind}", flush=True)
    u, v, g1, g2, flag, params = calculate_star_shapes(seed, nstars, optics_type)
    cat = treecorr.Catalog(x=u, y=v, g1=g1, g2=g2, flag=flag, x_units='arcsec', y_units='arcsec')
    
    print(f"Saving exposure {ind}: {time.time() - start}", flush=True)
    cat.write(out_dir + f'optic_shapes{ind}.fits')
    print(f"Saved exposure {ind}: {time.time() - start}", flush=True)
    
    return u, v, g1, g2

def make_one_exposure_wrapper(args):
    ind, = args
    try:
        # check if exposure already exists
        with fits.open(f"./exposures/optic_shapes{ind}.fits") as hdul:
            print(f"Exposure {ind} already exists", flush=True)
            return hdul[1].data['x'], hdul[1].data['y'], hdul[1].data['g1'], hdul[1].data['g2']
    except:
        return make_one_exposure(ind, 1100*ind)

if __name__ == '__main__':
    n_exposures = 25
    out_dir = "./exposures/"
    scaling_factor = 5
    skip_factor = 2
    plot_margin = 0.15

    args = zip(range(n_exposures))
    pool = mp.Pool(processes=25)
    results = pool.map(make_one_exposure_wrapper, args)
    pool.close()
    pool.join()

    print("Making plots", flush=True)

    fig, axs = plt.subplots(5, 5, figsize=(15, 15))
    for ind, ax in enumerate(axs.flatten()):
        u, v, g1, g2 = results[ind]
        gammas = np.array(g1) + 1j*np.array(g2)
        conv_factor = (1*units.arcsec/units.degree).to("").value
        coords = np.vstack((u*conv_factor, v*conv_factor, gammas)).T
        for ra, de, gam in coords[::skip_factor]:
            ax.plot(*getLineParameters(ra, de, gam*scaling_factor), \
                        color="black", lw=0.5, alpha = 0.5)
            ax.set_aspect('equal')
            ax.set_xlim([np.nanmin(u*conv_factor) - plot_margin, \
                         np.nanmax(u*conv_factor) + plot_margin])
            ax.set_ylim([np.nanmin(v*conv_factor) - plot_margin, \
                         np.nanmax(v*conv_factor) + plot_margin])

            ax.set_xlabel("RA (deg)")
            ax.set_ylabel("Dec (deg)")
            ax.set_title(f"{ind}")
            ax.set_aspect('equal')
            ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_dir + "exposures.png", dpi=300)
    plt.close()
    
