20230122_sample_test/
################ README START
# set these up
print("setting up")

# angular range and density of psfs
rad = 1.2
RA_range = (-rad, rad)
Dec_range = (-rad, rad)
num_den = 1000

num_samples = 10

# saving stuff
subfolder = "20230122_sample_test/"
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
