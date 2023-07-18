import glob, os
from scipy.io import netcdf_file
import numpy as np
from time import time


def get_extension(folderpath):
    inputfile = glob.glob(f"{folderpath}/input.*")
    extension = os.path.basename(inputfile[0])[6:]
    return extension


def write_var_txt(woutname, varname="tmp", prints=False):
    with netcdf_file(woutname, "r") as wfile:
        with open(f"{varname}.txt", "w") as tfile:
            for key in sorted(wfile.variables.keys()):
                shape = np.array([wfile.variables[key].data]).shape
                tfile.write(f"{key}\t{shape}\n")
                tfile.write(f"{np.array([wfile.variables[key].data])}\n\n")
                if prints:
                    print(key)
                    print(wfile.variables[key].data)


def normalize(a):
    return (a - a.min()) / (a.max() - a.min())


def equal_aspect(ax):
    """
    For 3D plots to have the same aspect ratio.
    TODO: Move to a plot-specific library
    """
    x, y, z = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))


def timer(func, *args, **kwargs):
    # This function shows the execution time of
    # the function object passed
    t1 = time()
    result = func(*args, **kwargs)
    t2 = time()
    print(f"{func}:\t {(t2-t1):.6f}s")
    return result
