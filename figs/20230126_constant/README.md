20230126_constant/
################ README START
# set these up
print("setting up")

# angular range and density of psfs
rad = 1.2
RA_range = (-rad, rad)
Dec_range = (-rad, rad)
num_den = 100000

# Zernike polynomials

# mike paper
# d = np.vectorize(lambda x, y: -0.006 + np.real((0.009 - 0.003j) * (x - 1j * y) / rad))
# s = np.vectorize(lambda x, y: 0)

# a = np.vectorize(lambda x, y: 0.014 - 0.011j + (0.001 - 0.002j) * (x + 1j*y) / rad)
# a_real, a_imag = np.vectorize(lambda x, y: np.real(a(x, y))), np.vectorize(lambda x, y: np.imag(a(x, y)))

# c_real, c_imag = np.vectorize(lambda x, y: -0.039), np.vectorize(lambda x, y: -0.010)
# c = np.vectorize(lambda x, y: c_real(x, y) + c_imag(x, y) * 1j)

d = np.vectorize(lambda x, y: 1e-1)
s = np.vectorize(lambda x, y: 1e-2)

a_real, a_imag = np.vectorize(lambda x, y: 1e-2), np.vectorize(lambda x, y: np.sqrt(x**2+y**2) * 3e-3) #
a = np.vectorize(lambda x, y: a_real(x, y) + a_imag(x, y) * 1j)

c_real, c_imag = np.vectorize(lambda x, y: 1e-2), np.vectorize(lambda x, y: np.sqrt(x**2+y**2) * 3e-3)
c = np.vectorize(lambda x, y: c_real(x, y) + c_imag(x, y) * 1j)

# saving stuff
subfolder = "20230126_constant/"
# subfolder = "20230112_mike_paper/"
# subfolder = "20230110_radial_imag/"
out_dir = f"./out/{subfolder}"
plot_dir = f"./figs/{subfolder}"
Path(out_dir).mkdir(parents=True, exist_ok=True)
Path(plot_dir).mkdir(parents=True, exist_ok=True)

# plotting stuff
coeffs = [d, a_real, a_imag, s, c_real, c_imag, \
         np.vectorize(lambda x, y : np.real(Q(x, y))), \
          np.vectorize(lambda x, y : np.imag(Q(x, y))), lambda x, y : S(x, y)]
# labels = [r"$d$", r"$\Re(a)$", r"$\Im(a) \sim \sqrt{x^2+y^2}$", r"$s$", r"$\Re(c)$", r"$\Im(c)\sim \sqrt{x^2+y^2}$", \
#           r"$\Re(Q)$", r"$\Im(Q)$", r"S"]
labels = [r"$d$", r"$\Re(a)$", r"$\Im(a)$", r"$s$", r"$\Re(c)$", r"$\Im(c)$", \
          r"$\Re(Q)$", r"$\Im(Q)$", r"S"]
thin_gal = 1000
alpha = 0.5

# 3pt stuff
rmin = 0.5
rmax = 20
nr = 10

# 2pt stuff
twoptrmin = 0.5
twoptrmax = 20
bin_size = 0.1
bin_slop = 0.1
################# README END
