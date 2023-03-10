import numpy as np
from matplotlib import pyplot as plt
from astropy import constants as c
from astropy import units as u
import sys, os, glob
from astropy.io import fits
from matplotlib.patches import Ellipse
import treecorr
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
from pathlib import Path

################ README START
# set these up
print("setting up")

# angular range and density of psfs
rad = 1.2
RA_range = (-rad, rad)
Dec_range = (-rad, rad)
num_den = 100000

thin_gal = 1000
alpha = 0.5

num_samples = 10

# saving stuff
subfolder = "20230127_sample_highn/"
out_dir = f"./out/{subfolder}/"
plot_dir = f"./figs/{subfolder}/"
Path(plot_dir + "3point/").mkdir(parents=True, exist_ok=True)
Path(plot_dir + "2point/").mkdir(parents=True, exist_ok=True)
Path(plot_dir + "coeff_shear/").mkdir(parents=True, exist_ok=True)
Path(out_dir).mkdir(parents=True, exist_ok=True)

# 3pt stuff
rmin = 0.5
rmax = 20
nr = 10

# 2pt stuff
twoptrmin = 0.5
twoptrmax = 20
bin_size = 0.1
bin_slop = 0.1

#################
# Readme generator
readme_path = os.path.join(plot_dir, 'README.md')
with open(readme_path, 'w') as f:
    f.write(subfolder+'\n')
    with open(os.path.abspath(__file__), 'r') as py_file:
        removing = False
        for line in py_file.readlines():
            if 'README START' in line:
                removing = True
            if removing:
                f.write(line)
            if 'README END' in line:
                removing = False
                break
    print(f'README.md generated at {readme_path}')

###################

try:
    RAs_Decs = np.load(out_dir + f'/RAs_Decs.npz')
    RAs, Decs = RAs_Decs["RAs"], RAs_Decs["Decs"]
except:
    num_gal = int(np.abs(num_den * (Dec_range[1] - Dec_range[0]) * (RA_range[1] - RA_range[0])))
    print(num_gal)

    RAs = np.random.uniform(RA_range[0], RA_range[1], size=num_gal)
    Decs = np.random.uniform(Dec_range[0], Dec_range[1], size=num_gal)
    np.savez(out_dir + f'/RAs_Decs.npz', RAs=RAs, Decs=Decs)

coeff_file = open(out_dir + "coeff.txt", "w")
coeff_file.write("ind, d0, d1, a0, a1, c0, c1\n")

# combine all 2/3point values

phis, g_ttts, sig_ttts = np.zeros((num_samples, nr, 20 * 4)), \
            np.zeros((num_samples, nr, 20 * 4)), np.zeros((num_samples, nr, 20 * 4))
meanrs = np.zeros((num_samples, nr))

rnoms, xips, xims, varxips, varxims = np.zeros((num_samples, 37)), np.zeros((num_samples, 37)), \
    np.zeros((num_samples, 37)), np.zeros((num_samples, 37)), np.zeros((num_samples, 37))

for ind in range(num_samples):
    Path(out_dir + f"{ind}/").mkdir(parents=True, exist_ok=True)
    
    try:
        coeff = np.load(out_dir + f'/{ind}/coeff.npz')
        print("imported previously saved coefficients", coeff)
        d0, d1, a0, a1, c0, c1 = coeff["coeff"]
        d0, c0, c1 = np.real(d0), np.real(c0), np.real(c1)
    except:
        # Zernike polynomials
        # sample values based on Table 
        d0, d1 = np.random.normal(-0.006, 0.053), (np.random.normal(0.009, 0.038) - np.random.normal(-0.003, 0.040)*1j)
        a0, a1 = (np.random.normal(0.014, 0.099) - np.random.normal(-0.011, 0.089)*1j), \
                    (np.random.normal(0.001, 0.066) - np.random.normal(-0.002, 0.055)*1j)
        c0, c1 = np.random.normal(-0.039, 0.109), np.random.normal(-0.010, 0.124)
        np.savez(out_dir + f'/{ind}/coeff.npz', coeff=[d0, d1, a0, a1, c0, c1])
    
    print(ind, d0, d1, a0, a1, c0, c1)
    coeff_file.write(f"{ind}, {d0}, {d1}, {a0}, {a1}, {c0}, {c1}\n")
    
    d = np.vectorize(lambda x, y: d0 + np.real(d1 * (x - 1j * y) / rad))
    s = np.vectorize(lambda x, y: 0)

    a = np.vectorize(lambda x, y: a0 + a1 * (x + 1j*y) / rad)
    a_real, a_imag = np.vectorize(lambda x, y: np.real(a(x, y))), np.vectorize(lambda x, y: np.imag(a(x, y)))

    c_real, c_imag = np.vectorize(lambda x, y: c0), np.vectorize(lambda x, y: c1)
    c = np.vectorize(lambda x, y: c_real(x, y) + c_imag(x, y) * 1j)
    
    # plotting stuff
    coeffs = [d, a_real, a_imag, s, c_real, c_imag, \
             np.vectorize(lambda x, y : np.real(Q(x, y))), \
              np.vectorize(lambda x, y : np.imag(Q(x, y))), lambda x, y : S(x, y)]
    # labels = [r"$d$", r"$\Re(a)$", r"$\Im(a) \sim \sqrt{x^2+y^2}$", r"$s$", r"$\Re(c)$", r"$\Im(c)\sim \sqrt{x^2+y^2}$", \
    #           r"$\Re(Q)$", r"$\Im(Q)$", r"S"]
    labels = [r"$d$", r"$\Re(a)$", r"$\Im(a)$", r"$s$", r"$\Re(c)$", r"$\Im(c)$", \
              r"$\Re(Q)$", r"$\Im(Q)$", r"S"]
    
################# README END

    def Q(x, y):
        return 4 * (d(x, y) + 4/3 * s(x, y)) * a(x, y) + 1/3 * c(x, y)**2
    def S(x, y):
        return 2 * (d(x, y) + 4/3 * s(x, y)) ** 2 \
                + 2 * np.abs(a(x, y))**2 + 2/3 * np.abs(c(x, y))**2 + 4/9 * np.abs(s(x, y))**2
    def gamma(x, y):
        return Q(x, y) / S(x, y)

    def getLineParameters(xpos, ypos, gamma):
        length, angle = np.abs(gamma), np.angle(gamma)
        return [xpos - np.cos(angle)*length/2, xpos + np.cos(angle)*length/2], \
                [ypos - np.sin(angle)*length/2, ypos + np.sin(angle)*length/2]

    print("calculating shear field")
    gammas = gamma(RAs, Decs)
    e_s = np.abs(gammas*2)
    phi_s = np.angle(gammas*2)
    a_ep_s = np.sqrt(S(RAs, Decs))
    b_ep_s = a_ep_s * np.sqrt(1 - e_s**2)
    print(a_ep_s.shape, b_ep_s.shape, e_s.shape, phi_s.shape, np.degrees(phi_s).shape)

    rand_coords = np.vstack((RAs, Decs, a_ep_s, b_ep_s, phi_s, gammas)).T
    print(rand_coords.shape)

    RA_grid, Dec_grid = np.meshgrid(np.linspace(*RA_range), np.linspace(*Dec_range))

    print("plotting coefficents and shear field")
    # plot coefficients
    fig = plt.figure(figsize=(20, 8))
    plt.rcParams.update({'font.size': 8})
    gs = fig.add_gridspec(3, 24)
    axs = list(map(fig.add_subplot, [gs[0, 4*y:4*y+3] for y in range(3)] + \
                   [gs[1, 4*y:4*y+3] for y in range(3)] + [gs[2, 4*y:4*y+3] for y in range(3)]))
    for ax, coeff, label in zip(axs, coeffs, labels):
        f = ax.imshow(coeff(RA_grid, Dec_grid), origin="lower", extent=[*RA_range, *Dec_range])
        cbar = fig.colorbar(f, orientation='horizontal')
        cbar.set_label(label)
        ax.set_aspect('equal')
        ax.grid()

    # plot shear field
    ax = fig.add_subplot(gs[:, 11:])
    for ra, de, a_ep, b_ep, phi, gam in rand_coords[::thin_gal]:
        ra, de, a_ep, b_ep, phi = np.real([ra, de, a_ep, b_ep, phi])
        # Create an ellipse patch with a given size centered at the RA, Dec position
        plt.plot(*getLineParameters(ra, de, gam), color="black", lw=2)
        ellipse = Ellipse(xy=(ra, de), width=a_ep, height=b_ep, angle=np.degrees(phi), alpha=alpha)
        # Add the ellipse patch to the axes
        ax.add_patch(ellipse)

    # Show the plot
    ax.set_aspect('equal')
    ax.grid()
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    plt.xlim(RA_range)
    plt.ylim(Dec_range)
    plt.savefig(plot_dir+f"coeff_shear/{ind}.png")
    plt.savefig(plot_dir+f"coeff_shear/{ind}.pdf", dpi=300)
    plt.close()

    ###################################
    # 3 pt
    print("3pt time")
    cats = treecorr.Catalog(ra=RAs, dec=Decs, g1=np.real(gammas), g2=np.imag(gammas), \
                    ra_units='deg', dec_units='deg')

    narrow = dict(min_sep=rmin, max_sep=rmax, sep_units='arcmin', nbins=nr,
                  min_u=0.0, max_u=1, nubins=20,
                  min_v=0.0, max_v=0.1, nvbins=1, verbose=2, output_dots=True)
    wide = dict(min_sep=rmin, max_sep=rmax, sep_units='arcmin', nbins=nr,
                min_u=0.9, max_u=1, nubins=1,
                min_v=0.0, max_v=0.8, nvbins=20, verbose=2, output_dots=True)
    wider = dict(min_sep=rmin, max_sep=rmax, sep_units='arcmin', nbins=nr,
                 min_u=0.9, max_u=1, nubins=1,
                 min_v=0.8, max_v=0.95, nvbins=20, verbose=2, output_dots=True)
    widest = dict(min_sep=rmin, max_sep=rmax, sep_units='arcmin', nbins=nr,
                  min_u=0.9, max_u=1, nubins=1,
                  min_v=0.95, max_v=1.0, nvbins=20, verbose=2, output_dots=True)

    print("narrow")
    ggg1 = treecorr.GGGCorrelation(narrow)
    try:
        ggg1.read(out_dir + f'/{ind}/narrow.hdf')
    except:
        ggg1.process(cats, comm=comm)
        ggg1.write(out_dir + f'/{ind}/narrow.hdf', write_patch_results=True)

    print("wide")
    ggg2 = treecorr.GGGCorrelation(wide)
    try:
        ggg2.read(out_dir + f'/{ind}/wide.hdf')
    except:
        ggg2.process(cats, comm=comm)
        ggg2.write(out_dir + f'/{ind}/wide.hdf', write_patch_results=True)

    print("wider")
    ggg3 = treecorr.GGGCorrelation(wider)
    try:
        ggg3.read(out_dir + f'/{ind}/wider.hdf')
    except:
        ggg3.process(cats, comm=comm)
        ggg3.write(out_dir + f'/{ind}/wider.hdf', write_patch_results=True)

    print("widest")
    ggg4 = treecorr.GGGCorrelation(widest)
    try:
        ggg4.read(out_dir + f'/{ind}/widest.hdf')
    except:
        ggg4.process(cats, comm=comm)
        ggg4.write(out_dir + f'/{ind}/widest.hdf', write_patch_results=True)


    #########################
    # plotting
    print("plotting 3pt")
    all_g_ttt = []
    all_sig_ttt = []
    all_meanr = []
    all_phi = []

    for ggg in [ggg1, ggg2, ggg3, ggg4]:

        g_ttt = -0.25 * (ggg.gam0 + ggg.gam1 + ggg.gam2 + ggg.gam3).real
        var_ttt = 0.25**2 * (ggg.vargam0 + ggg.vargam1 + ggg.vargam2 + ggg.vargam3)

        _nr, nu, nv = g_ttt.shape
        # print(nr,nu,nv)
        assert _nr == nr
        assert nv % 2 == 0
        nv //= 2
        assert nu == 1 or nv == 1

        d1 = ggg.meand1
        d2 = ggg.meand2
        d3 = ggg.meand3
        if nu == 1:
            # if nu==1, then u=1, so d2 = d3, and phi is between d2 and d3
            phi = np.arccos( (d2**2 + d3**2 - d1**2) / (2*d2*d3) )
            meanr = np.array([np.mean([d2[ir], d3[ir]]) for ir in range(nr)])
        else:
            # if nv==1, then v=0, so d1 = d2, and phi is between d1 and d2
            phi = np.arccos( (d1**2 + d2**2 - d3**2) / (2*d1*d2) )
            meanr = np.array([np.mean([d1[ir], d2[ir]]) for ir in range(nr)])
        phi *= 180/np.pi

        # We don't care about v>0 vs v<0, so combine them.
        phi = (phi[:,:,nv-1::-1] + phi[:,:,nv:]) / 2
        g_ttt = (g_ttt[:,:,nv-1::-1] + g_ttt[:,:,nv:]) / 2
        var_ttt = (var_ttt[:,:,nv-1::-1] + var_ttt[:,:,nv:]) / 4
        sig_ttt = var_ttt**0.5

        # print('shapes:')
        print('phi: ',phi.shape)
        # print('g_ttt: ',g_ttt.shape)
        # print('sig_ttt: ',sig_ttt.shape)
        # print('meanr: ',meanr.shape)

        print('meanr =  ',meanr)

        if nu == 1:
            phi = phi[:,0,:]
            g_ttt = g_ttt[:,0,:]
            sig_ttt = sig_ttt[:,0,:]
        else:
            phi = phi[:,:,0]
            g_ttt = g_ttt[:,:,0]
            sig_ttt = sig_ttt[:,:,0]

        # print('shapes ->')
        # print('phi: ',phi.shape)
        # print('g_ttt: ',g_ttt.shape)
        # print('sig_ttt: ',sig_ttt.shape)

        all_phi.append(phi)
        all_g_ttt.append(g_ttt)
        all_sig_ttt.append(sig_ttt)
        all_meanr.append(meanr)

    phi = np.concatenate(all_phi, axis=1)
    g_ttt = np.concatenate(all_g_ttt, axis=1)
    sig_ttt = np.concatenate(all_sig_ttt, axis=1)
    meanr = np.concatenate(all_meanr, axis=0)

    fig, ax = plt.subplots()

    lines = []
    for ir in range(nr):
        # print('ir = ',ir)
        # print('meanr = ',meanr[ir])
        # print('phi = ',phi[ir])
        # print('g = ',g_ttt[ir])
        # print('sig = ',sig_ttt[ir])
        # print()

        line = ax.errorbar(phi[ir], g_ttt[ir], sig_ttt[ir])
        lines.append((line, 'd2 ~= %.1f'%meanr[ir]))
        
    phis[ind] = phi
    g_ttts[ind] = g_ttt
    sig_ttts[ind] = sig_ttt
    meanrs[ind] = meanr[:nr]

    # save to out file
    np.savez(out_dir + f'/{ind}/3point.npz', phi=phi, g_ttt=g_ttt, sig_ttt=sig_ttt, meanr=meanr)
    
    ax.legend(*(list(zip(*lines))), loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel(r'Opening angle $\phi$ [deg]')
    ax.set_ylabel(r'$\gamma_{\rm ttt}$ isoceles')

    fig.set_tight_layout(True)
    ax.grid()
    ax.axhline(y=0, ls="--", color="black")
    plt.savefig(plot_dir + f'/3point/{ind}.png')
    plt.savefig(plot_dir + f'/3point/{ind}.pdf', dpi=300)
    # plt.show()
    plt.close()
    
    ################
    # 2 pt
    cats = treecorr.Catalog(ra=RAs, dec=Decs, g1=np.real(gammas), g2=np.imag(gammas), \
                    ra_units='deg', dec_units='deg')

    rho = treecorr.GGCorrelation(min_sep=twoptrmin, max_sep=twoptrmax, sep_units='arcmin',
                                bin_size=bin_size, bin_slop=bin_slop, verbose=2)
    rho.process(cats)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.errorbar(rho.rnom, rho.xip, np.sqrt(rho.varxip), label=r"$\xi_+(r)$")
    ax2.errorbar(rho.rnom, rho.xim, np.sqrt(rho.varxim), label=r"$\xi_-(r)$")
    # ax.errorbar(rho.rnom, rho.xip_im, np.sqrt(rho.varxip), label=r"$\Im\xi_+(r)$")
    # ax.errorbar(rho.rnom, rho.xip_im, np.sqrt(rho.varxip), label=r"$\Im\xi_-(r)$")
    fig.set_tight_layout(True)
    ax1.grid()
    ax2.grid()
    # ax.set_yscale("log")
    ax1.legend()
    ax2.legend()
    ax2.set_xlabel("separation (arcmin)")
    plt.savefig(plot_dir + f'/2point/{ind}.png')
    plt.savefig(plot_dir + f'/2point/{ind}.pdf', dpi=300)
    # plt.show()
    plt.close()
    
    # save to out file
    np.savez(out_dir + f'/{ind}/2point.npz', rnom=rho.rnom, xip=rho.xip, varxip=rho.varxip,\
                            xim=rho.xim, varxim=rho.varxim)
    
    # check these 2 point files to set the size of these arrays
    rnoms[ind], xips[ind], xims[ind], varxips[ind], varxims[ind] = rho.rnom, rho.xip, rho.xim, rho.varxip, rho.varxim
    

coeff_file.close()
print(phis.shape, g_ttts.shape, sig_ttts.shape, meanrs.shape)

fig, ax = plt.subplots()
lines = []
phi_mean, phi_stds = np.nanmean(phis, axis=0), np.nanstd(phis, axis=0)
g_mean, g_stds = np.nanmean(g_ttts, axis=0), np.nanstd(g_ttts, axis=0)
for ir in range(nr):
    line = ax.errorbar(phi_mean[ir], g_mean[ir], yerr=g_stds[ir], xerr=phi_stds[ir])
    lines.append((line, 'd2 ~= %.1f'%meanr[ir]))
ax.legend(*(list(zip(*lines))), loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'Opening angle $\phi$ [deg]')
ax.set_ylabel(r'$\gamma_{\rm ttt}$ isoceles')
fig.set_tight_layout(True)
ax.grid()
ax.axhline(y=0, ls="--", color="black")
plt.savefig(plot_dir + f'/3point.png')
plt.savefig(plot_dir + f'/3point.pdf', dpi=300)
# plt.show()
plt.close()

# 2 point

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.errorbar(np.nanmean(rnoms, axis=0), np.nanmean(xips, axis=0), \
             np.nanstd(xips, axis=0), np.nanstd(rnoms, axis=0), label=r"$\xi_+(r)$")
ax2.errorbar(np.nanmean(rnoms, axis=0), np.nanmean(xims, axis=0), \
             np.nanstd(xims, axis=0), np.nanstd(rnoms, axis=0), label=r"$\xi_-(r)$")
# ax.errorbar(rho.rnom, rho.xip_im, np.sqrt(rho.varxip), label=r"$\Im\xi_+(r)$")
# ax.errorbar(rho.rnom, rho.xip_im, np.sqrt(rho.varxip), label=r"$\Im\xi_-(r)$")
fig.set_tight_layout(True)
ax1.grid()
ax2.grid()
# ax.set_yscale("log")
ax1.legend()
ax2.legend()
ax2.set_xlabel("separation (arcmin)")
plt.savefig(plot_dir + f'/2point.png')
plt.savefig(plot_dir + f'/2point.pdf', dpi=300)
# plt.show()
plt.close()