import numpy as np
from scipy.io import netcdf_file
from scipy.optimize import basinhopping
from .helper import vh


class MyBounds:
    def __init__(self, xmax=[1.1, 1.1], xmin=[-1.1, -1.1]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        result = tmax and tmin
        return result


class Booz:
    def __init__(
        self,
        woutfile,
        theta=np.linspace(0, 2 * np.pi, 100),
        phi=np.linspace(0, 2 * np.pi, 100),
    ):
        self.woutdata = {}
        self.vars = {}
        self.ders = {}
        self.vecs = {}
        self.xyzs = {}
        self.read_woutdata(woutfile=woutfile)
        self.s_idx = self.woutdata["jlist"] - 1
        self.s_vmec = np.linspace(0, 1, self.woutdata["ns_b"])
        self.s = self.s_vmec[self.s_idx]
        self.iota = self.woutdata["iota_b"][self.s_idx]
        self._th = theta
        self._ph = phi
        self.init_vh()

    @property
    def th(self):
        return self._th

    @th.setter
    def th(self, th):
        self._th = th
        self.init_vh()

    @property
    def ph(self):
        return self._ph

    @ph.setter
    def ph(self, ph):
        self._ph = ph
        self.init_vh()

    def init_vh(self):
        vh.initialize(
            ms=self.woutdata["ixm_b"],
            ns=self.woutdata["ixn_b"],
            th=self.th,
            ph=self.ph,
        )

    def read_woutdata(self, woutfile):
        with netcdf_file(woutfile, "r") as wfile:
            for var in wfile.variables:
                self.woutdata[var] = wfile.variables[var].data.copy()

    def print_vars(self):
        for key in sorted(self.woutdata.keys()):
            print(key)

    def print_sizes(self):
        for key in self.woutdata:
            print(key, self.woutdata[key].shape)

    def get_all_single(self, s_idx, theta, phi):
        self.th = theta
        self.ph = phi
        self.init_vh()
        self.get_vars(s_idx=s_idx)
        self.get_xyzs(s_idx=s_idx)
        self.get_derivatives()
        self.calculate_vectors()

    def get_vars(self, s_idx=None):
        if s_idx == None:
            s_idx = slice(s_idx)
        else:
            s_idx = np.array([s_idx])
        ms = self.woutdata["ixm_b"]
        ns = self.woutdata["ixn_b"]
        R = vh.genvar_new(
            self.woutdata["rmnc_b"][s_idx].T, self.th, self.ph, ms, ns, "c"
        )
        Z = vh.genvar_new(
            self.woutdata["zmns_b"][s_idx].T, self.th, self.ph, ms, ns, "s"
        )
        P = vh.genvar_new(
            self.woutdata["pmns_b"][s_idx].T, self.th, self.ph, ms, ns, "s"
        )
        PHI = self.ph + P
        mod_b = vh.genvar_new(
            self.woutdata["bmnc_b"][s_idx].T, self.th, self.ph, ms, ns, "c"
        )
        sqrt_g = vh.genvar_new(
            self.woutdata["gmn_b"][s_idx].T, self.th, self.ph, ms, ns, "c"
        )
        self.vars = {
            "R": R,
            "Z": Z,
            "PHI": PHI,
            "P": P,
            "mod_b": mod_b,
            "sqrt_g": sqrt_g,
        }

    def posdifsq(self, theta, phi, s_idx, xyz):
        xyz = np.array(xyz)
        xyz_surf = np.array(
            [self.get_single_xyz(theta=theta, phi=phi, s_idx=s_idx)]
        )
        return np.sum((xyz - xyz_surf) ** 2)

    def get_single_xyz(self, theta, phi, s_idx=-1):
        ms = self.woutdata["ixm_b"]
        ns = self.woutdata["ixn_b"]
        R = vh.genvar(
            self.woutdata["rmnc_b"][s_idx].T, theta, phi, ms, ns, "c"
        )
        Z = vh.genvar(
            self.woutdata["zmns_b"][s_idx].T, theta, phi, ms, ns, "s"
        )
        P = vh.genvar(
            self.woutdata["pmns_b"][s_idx].T, theta, phi, ms, ns, "s"
        )
        PHI_CYL = phi + P
        X = R * np.cos(PHI_CYL)
        Y = R * np.sin(PHI_CYL)
        return np.array([X, Y, Z]).flatten()

    def get_closest_booz_ang_pos(self, s_idx, xyz_0, init=[0, 0]):
        """
        Finds closest point to coil on the surface _surf_ and gets its magnetic coordinates.
        """
        xyz_0 = np.array(xyz_0).flatten()
        tfun = lambda thphi: self.posdifsq(
            theta=thphi[0], phi=thphi[1], s_idx=s_idx, xyz=xyz_0
        )
        print(xyz_0, tfun(init))
        bounds = MyBounds(xmin=[-np.pi, 0], xmax=[np.pi, np.pi * 2])
        res = basinhopping(
            tfun,
            init,
            15,
            T=5.0,
            stepsize=1.0,
            accept_test=bounds,
            # callback=callback,
        )
        if not res.success:
            print("unsuccessful")
        theta, phi = res.x[0], res.x[1]
        print("result: ", tfun(res.x))
        print(res)
        return theta, phi

    def get_xyzs(self, s_idx=None):
        self.get_vars(s_idx=s_idx)
        xs = self.vars["R"] * np.cos(self.vars["PHI"])
        ys = self.vars["R"] * np.sin(self.vars["PHI"])
        zs = self.vars["Z"]
        self.xyzs = {"xs": xs, "ys": ys, "zs": zs}

    def get_derivatives(
        self,
    ):
        ms = self.woutdata["ixm_b"]
        ns = self.woutdata["ixn_b"]
        dr_ds = +vh.dgen_ds(
            self.woutdata["rmnc_b"].T,
            self.s,
            self.th,
            self.ph,
            ms,
            ns,
            typ="c",
        )
        dz_ds = +vh.dgen_ds(
            self.woutdata["zmns_b"].T,
            self.s,
            self.th,
            self.ph,
            ms,
            ns,
            typ="s",
        )
        dph_ds = +vh.dgen_ds(
            self.woutdata["pmns_b"].T,
            self.s,
            self.th,
            self.ph,
            ms,
            ns,
            typ="s",
        )
        dr_dth = -vh.genvar_modi_new(
            self.woutdata["rmnc_b"].T, self.th, self.ph, ms, ns, ms, typ="s"
        )
        dz_dth = +vh.genvar_modi_new(
            self.woutdata["zmns_b"].T, self.th, self.ph, ms, ns, ms, typ="c"
        )
        dph_dth = +vh.genvar_modi_new(
            self.woutdata["pmns_b"].T, self.th, self.ph, ms, ns, ms, typ="c"
        )
        dr_dph = +vh.genvar_modi_new(
            self.woutdata["rmnc_b"].T, self.th, self.ph, ms, ns, ns, typ="s"
        )
        dz_dph = -vh.genvar_modi_new(
            self.woutdata["zmns_b"].T, self.th, self.ph, ms, ns, ns, typ="c"
        )
        dph_dph = 1 - vh.genvar_modi_new(
            self.woutdata["pmns_b"].T, self.th, self.ph, ms, ns, ns, typ="c"
        )
        self.ders = {
            "dr_ds": dr_ds,
            "dz_ds": dz_ds,
            "dph_ds": dph_ds,
            "dr_dth": dr_dth,
            "dz_dth": dz_dth,
            "dph_dth": dph_dth,
            "dr_dph": dr_dph,
            "dz_dph": dz_dph,
            "dph_dph": dph_dph,
        }

    def get_vectors(self, get_grads=True):
        self.get_vars()
        self.get_xyzs()
        self.get_derivatives()
        self.calculate_vectors(get_grads=get_grads)

    def calculate_vectors(self, get_grads=True):
        exph = np.exp(1j * self.vars["PHI"])
        r_exph = self.vars["R"] * exph
        dr_ds_exph = self.ders["dr_ds"] * exph
        dr_dth_exph = self.ders["dr_dth"] * exph
        dr_dph_exph = self.ders["dr_dph"] * exph
        e_s = np.array(
            [
                dr_ds_exph.real - r_exph.imag * self.ders["dph_ds"],
                dr_ds_exph.imag + r_exph.real * self.ders["dph_ds"],
                self.ders["dz_ds"],
            ]
        )
        e_th = np.array(
            [
                dr_dth_exph.real - r_exph.imag * self.ders["dph_dth"],
                dr_dth_exph.imag + r_exph.real * self.ders["dph_dth"],
                self.ders["dz_dth"],
            ]
        )
        e_ph = np.array(
            [
                dr_dph_exph.real - r_exph.imag * self.ders["dph_dph"],
                dr_dph_exph.imag + r_exph.real * self.ders["dph_dph"],
                self.ders["dz_dph"],
            ]
        )
        jac = np.einsum(
            "ijkl,ijkl->jkl",
            e_s,
            np.cross(e_th, e_ph, axisa=0, axisb=0, axisc=0),
        )
        if get_grads is True:
            grad_s = np.cross(e_th, e_ph, axisa=0, axisb=0, axisc=0) / jac
            grad_th = np.cross(e_ph, e_s, axisa=0, axisb=0, axisc=0) / jac
            grad_ph = np.cross(e_s, e_th, axisa=0, axisb=0, axisc=0) / jac
        else:
            grad_s = None
            grad_th = None
            grad_ph = None
        self.vecs = {
            "e_s": e_s,
            "e_th": e_th,
            "e_ph": e_ph,
            "grad_s": grad_s,
            "grad_th": grad_th,
            "grad_ph": grad_ph,
        }
        self.vars["sqrtg_vecs"] = jac
