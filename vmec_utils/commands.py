import os
import subprocess
from scipy.io import netcdf_file
import numpy as np

##  VMEC


def call_vmec(folderpath, extension):
    wd = os.getcwd()
    os.chdir(folderpath)
    subprocess.call(["xvmec2000", extension, ">&", "log.tmp", "&"])
    os.chdir(wd)


def make_woutname_vmec(folderpath, extension):
    return f"{folderpath}wout_{extension}.nc"


##  BOOZ_XFORM


def make_xform_input(folderpath, ext, s=None):
    woutname = make_woutname_vmec(folderpath, ext)
    with netcdf_file(woutname, "r") as wout:
        ntor = wout.variables["ntor"].data.copy()
        mpol = wout.variables["mpol"].data.copy()
        ns = wout.variables["ns"].data.copy()
    if s is None:
        s = np.arange(1, ns - 1, 10)
        print(s)
    elif s == "all":
        s = np.arange(1, ns)
    with open(f"{folderpath}in_booz.{ext}", "w") as f:
        f.write(f"{6*mpol} {3*ntor}\n")
        f.write(f"{ext}\n")
        [f.write(f"{i} ") for i in s]


def call_xform(folderpath, extension, s=None):
    make_xform_input(folderpath, extension, s)
    wd = os.getcwd()
    os.chdir(folderpath)
    subprocess.call(["xbooz_xform", f"in_booz.{extension}"])
    os.chdir(wd)


def make_woutname_booz(folderpath, extension):
    return f"{folderpath}boozmn_{extension}.nc"
