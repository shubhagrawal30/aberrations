import treecorr
import numpy as np
import matplotlib.pyplot as plt

cat = treecorr.Catalog('optic_shapes.fits', x_col='x', y_col='y', g1_col='g1', g2_col='g2',
                       x_units='arcsec', y_units='arcsec')
gg = treecorr.GGCorrelation(min_sep=1, max_sep=100, sep_units='arcmin', bin_size=0.1)
gg.process(cat)
print('xi+ = ',gg.xip)
print('xi- = ',gg.xim)

fix, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].scatter(cat.x * (180*60/np.pi), cat.y * (180*60/np.pi))
ax[0].set_xlabel('u (arcmin)')
ax[0].set_ylabel('v (arcmin)')

ax[1].plot(np.exp(gg.meanlogr), gg.xip)
ax[1].set_xlabel('r (arcmin)')
ax[1].set_ylabel('rho_0')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_ylim(1.e-6, 2.e-4)
ax[1].set_xlim(0.8, 120)
plt.show()

