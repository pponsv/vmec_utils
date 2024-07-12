import numpy as np
from scipy.io import netcdf_file
from .helper import vh
from scipy.interpolate import interp1d, Akima1DInterpolator, PchipInterpolator
from .fft_utils import invert_fourier, make_coef_array


var_keys = {
    "R": ["rmn", False],
    "Z": ["zmn", False],
    "lambda": ["lmn", False],
    "sqrt_g": ["gmn", True],
    "mod_b": ["bmn", True],
    "b_sub_s": ["bsubsmn", True],
    "b_sub_u": ["bsubumn", True],
    "b_sub_v": ["bsubvmn", True],
    "b_sup_u": ["bsupumn", True],
    "b_sup_v": ["bsupvmn", True],
    "J_sub_u": ["currumn", True],
    "J_sub_v": ["currvmn", True],
}


class Vmec:
    def __init__(self, woutfile, n_th, n_ph):
        self.n_th = n_th
        self.n_ph = n_ph
        self.woutdata = {}
        self.vars = {}
        self.ders = {}
        self.vecs = {}
        self.xyzs = {}
        self.read_woutdata(woutfile=woutfile)
        self.nm_tot = self.woutdata["mnmax"].copy()
        self.nm_nyq_tot = self.woutdata["mnmax_nyq"].copy()
        self.s = np.linspace(0, 1, self.woutdata["ns"])
        self.ds = np.mean(np.diff(self.s))
        self.s_half = self.s[1:] - self.ds / 2
        self.get_coefs_full_mesh()
        self.th = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
        self.ph = np.linspace(0, 2 * np.pi, n_ph, endpoint=False)
        self.compute_surface_variables()
        self.compute_derivatives()
        self.compute_sub_base()

    def read_woutdata(self, woutfile):
        with netcdf_file(woutfile, "r") as wfile:
            for var in wfile.variables:
                self.woutdata[var] = wfile.variables[var].data.copy()

    def interp_to_full_mesh(self, xmn):
        """
        Not sure if this is done correctly
        """
        # f = interp1d(
        #     self.s_half,
        #     xmn[1:],
        #     fill_value="extrapolate",  # type: ignore
        #     axis=0,
        #     # kind="quadratic",
        #     kind="linear",
        # )
        f = Akima1DInterpolator(self.s_half, xmn[1:], axis=0, extrapolate=True)
        # f = PchipInterpolator(self.s_half, xmn[1:], axis=0, extrapolate=True)
        return f(self.s)

    def get_coefs_full_mesh(self):
        self.xm = self.woutdata["xm"].astype(int)
        self.xn = self.woutdata["xn"].astype(int)
        self.xm_nyq = self.woutdata["xm_nyq"].astype(int)
        self.xn_nyq = self.woutdata["xn_nyq"].astype(int)
        self.rmnc = self.woutdata["rmnc"]
        self.zmns = self.woutdata["zmns"]
        self.lmns = self.interp_to_full_mesh(self.woutdata["lmns"])
        self.gmnc = self.interp_to_full_mesh(self.woutdata["gmnc"])
        self.bmnc = self.interp_to_full_mesh(self.woutdata["bmnc"])
        self.bsubumnc = self.interp_to_full_mesh(self.woutdata["bsubumnc"])
        self.bsubvmnc = self.interp_to_full_mesh(self.woutdata["bsubvmnc"])
        self.bsubsmns = self.interp_to_full_mesh(self.woutdata["bsubsmns"])
        self.currumnc = self.interp_to_full_mesh(self.woutdata["currumnc"])
        self.currvmnc = self.interp_to_full_mesh(self.woutdata["currvmnc"])
        self.bsupumnc = self.interp_to_full_mesh(self.woutdata["bsupumnc"])
        self.bsupvmnc = self.interp_to_full_mesh(self.woutdata["bsupvmnc"])

    def radial_derivative(self, xmn):
        """
        The method can be changed
        """
        return np.gradient(xmn, self.s, axis=0, edge_order=2)

    def get_coef_rad_derivatives(self):
        self.d_rmnc_ds = self.radial_derivative(self.rmnc)
        self.d_zmns_ds = self.radial_derivative(self.zmns)
        self.d_lmns_ds = self.radial_derivative(self.lmns)
        self.d_gmnc_ds = self.radial_derivative(self.gmnc)
        self.d_bmnc_ds = self.radial_derivative(self.bmnc)
        self.d_bsubumnc_ds = self.radial_derivative(self.bsubumnc)
        self.d_bsubvmnc_ds = self.radial_derivative(self.bsubvmnc)
        self.d_bsubsmns_ds = self.radial_derivative(self.bsubsmns)
        self.d_currumnc_ds = self.radial_derivative(self.currumnc)
        self.d_currvmnc_ds = self.radial_derivative(self.currvmnc)
        self.d_bsupumnc_ds = self.radial_derivative(self.bsupumnc)
        self.d_bsupvmnc_ds = self.radial_derivative(self.bsupvmnc)

    def compute_surface_variables(self):
        self.rs = self.wrap_invert_fourier(self.rmnc, kind="cos")
        self.zs = self.wrap_invert_fourier(self.zmns, kind="sin")
        self.xs = self.rs * np.cos(self.ph)
        self.ys = self.rs * np.sin(self.ph)
        self.ls = self.wrap_invert_fourier(self.lmns, kind="sin")
        self.sqrt_g = self.wrap_invert_fourier(self.gmnc, kind="cos")
        self.mod_b = self.wrap_invert_fourier(self.bmnc, kind="cos")

    def compute_derivatives(self):
        self.get_coef_rad_derivatives()
        self.dr_ds = self.wrap_invert_fourier(self.d_rmnc_ds, kind="cos")
        self.dr_dth = self.wrap_invert_fourier(
            self.rmnc, kind="cos", deriv_order=1, deriv_dir="th"
        )
        self.dr_dph = self.wrap_invert_fourier(
            self.rmnc, kind="cos", deriv_order=1, deriv_dir="ph"
        )
        self.dz_ds = self.wrap_invert_fourier(self.d_zmns_ds, kind="sin")
        self.dz_dth = self.wrap_invert_fourier(
            self.zmns, kind="sin", deriv_order=1, deriv_dir="th"
        )
        self.dz_dph = self.wrap_invert_fourier(
            self.zmns, kind="sin", deriv_order=1, deriv_dir="ph"
        )

    def compute_sub_base(self):
        sph = np.sin(self.ph)
        cph = np.cos(self.ph)
        self.e_sub_s = np.array(
            [
                self.dr_ds * cph,
                self.dr_ds * sph,
                self.dz_ds,
            ]
        )
        self.e_sub_th = np.array(
            [
                self.dr_dth * cph,
                self.dr_dth * sph,
                self.dz_dth,
            ]
        )
        self.e_sub_ph = np.array(
            [
                self.dr_dph * cph - self.rs * sph,
                self.dr_dph * sph + self.rs * cph,
                self.dz_dph,
            ]
        )
        self.sqrt_g = np.einsum(
            "ijkl,ijkl->jkl",
            self.e_sub_s,
            np.cross(self.e_sub_th, self.e_sub_ph, axisa=0, axisb=0, axisc=0),
        )

    def compute_super_base(self):
        self.e_sup_s = (
            np.cross(self.e_sub_th, self.e_sub_ph, axisa=0, axisb=0, axisc=0)
            / self.sqrt_g
        )
        self.e_sup_th = (
            np.cross(self.e_sub_ph, self.e_sub_s, axisa=0, axisb=0, axisc=0)
            / self.sqrt_g
        )
        self.e_sup_ph = (
            np.cross(self.e_sub_s, self.e_sub_th, axisa=0, axisb=0, axisc=0)
            / self.sqrt_g
        )

    def get_vars(self, s_idx=np.s_[:]):
        R = self.get_var("rmnc", "c", s_idx=s_idx)
        Z = self.get_var("zmns", "s", s_idx=s_idx)
        lambd = self.get_var("lmns", "s", s_idx=s_idx)
        sqrt_g = self.get_var("gmnc", "c", s_idx=s_idx)
        mod_b = self.get_var("bmnc", "c", s_idx=s_idx)
        # j_sub_u = self.get_var("currumnc", "c", s_idx=s_idx)
        # j_sub_v = self.get_var("currvmnc", "c", s_idx=s_idx)
        b_sub_s = self.get_var("bsubsmns", "s", s_idx=s_idx)
        b_sub_u = self.get_var("bsubumnc", "c", s_idx=s_idx)
        b_sub_v = self.get_var("bsubvmnc", "c", s_idx=s_idx)
        b_sup_u = self.get_var("bsupumnc", "c", s_idx=s_idx)
        b_sup_v = self.get_var("bsupvmnc", "c", s_idx=s_idx)
        self.vars = {
            "R": R,
            "Z": Z,
            "lambd": lambd,
            "sqrt_g": sqrt_g,
            "mod_b": mod_b,
            # "j_sub_u": j_sub_u,
            # "j_sub_v": j_sub_v,
            "b_sub_s": b_sub_s,
            "b_sub_u": b_sub_u,
            "b_sub_v": b_sub_v,
            "b_sup_u": b_sup_u,
            "b_sup_v": b_sup_v,
        }

    def get_xyzs(self, s_idx=np.s_[:]):
        self.get_vars(s_idx=s_idx)
        xs = self.vars["R"] * np.cos(self.ph)
        ys = self.vars["R"] * np.sin(self.ph)
        zs = self.vars["Z"]
        self.xyzs = {
            "xs": xs,
            "ys": ys,
            "zs": zs,
        }

    def get_xyzs_old(self, s_idx):
        rs = self.get_var_old("R", s_idx=s_idx)[0]
        xs = np.einsum("ij, j->ij", rs, np.cos(self.ph))
        ys = np.einsum("ij, j->ij", rs, np.sin(self.ph))
        zs = self.get_var_old("Z", s_idx=s_idx)[0]
        return xs, ys, zs

    def get_derivatives(self, s_idx=np.s_[:]):
        ms = self.woutdata["xm"]
        ns = self.woutdata["xn"]
        dr_ds = +vh.dgen_ds(
            self.woutdata["rmnc"].T, self.s, self.th, self.ph, ms, ns, typ="c"
        )
        dz_ds = +vh.dgen_ds(
            self.woutdata["zmns"].T, self.s, self.th, self.ph, ms, ns, typ="s"
        )
        dr_dth = -vh.genvar_modi(
            self.woutdata["rmnc"].T, self.th, self.ph, ms, ns, ms, typ="s"
        )
        dz_dth = +vh.genvar_modi(
            self.woutdata["zmns"].T, self.th, self.ph, ms, ns, ms, typ="c"
        )
        dr_dph = +vh.genvar_modi(
            self.woutdata["rmnc"].T, self.th, self.ph, ms, ns, ns, typ="s"
        )
        dz_dph = -vh.genvar_modi(
            self.woutdata["zmns"].T, self.th, self.ph, ms, ns, ns, typ="c"
        )
        self.ders = {
            "dr_ds": dr_ds,
            "dz_ds": dz_ds,
            "dr_dth": dr_dth,
            "dz_dth": dz_dth,
            "dr_dph": dr_dph,
            "dz_dph": dz_dph,
        }

    def get_vectors(self):
        self.get_vars()
        self.get_derivatives()
        sph = np.sin(self.ph)
        cph = np.cos(self.ph)
        e_s = np.array(
            [
                self.ders["dr_ds"] * cph,
                self.ders["dr_ds"] * sph,
                self.ders["dz_ds"],
            ]
        )
        e_th = np.array(
            [
                self.ders["dr_dth"] * cph,
                self.ders["dr_dth"] * sph,
                self.ders["dz_dth"],
            ]
        )
        e_ph = np.array(
            [
                self.ders["dr_dph"] * cph - self.vars["R"] * sph,
                self.ders["dr_dph"] * sph + self.vars["R"] * cph,
                self.ders["dz_dph"],
            ]
        )
        jac = np.einsum(
            "ijkl,ijkl->jkl",
            e_s,
            np.cross(e_th, e_ph, axisa=0, axisb=0, axisc=0),
        )
        grad_s = np.cross(e_th, e_ph, axisa=0, axisb=0, axisc=0) / jac
        grad_th = np.cross(e_ph, e_s, axisa=0, axisb=0, axisc=0) / jac
        grad_ph = np.cross(e_s, e_th, axisa=0, axisb=0, axisc=0) / jac
        self.vecs = {
            "e_s": e_s,
            "e_th": e_th,
            "e_ph": e_ph,
            "grad_s": grad_s,
            "grad_th": grad_th,
            "grad_ph": grad_ph,
        }
        self.vars["sqrtg_vecs"] = jac

    def get_vars_old(self, s_idx=None):
        for key in var_keys:
            self.get_var_old(key, s_idx)

    def get_vecs_old(self):
        ms, ns = self.woutdata["xm"], self.woutdata["xn"]
        dr_ds = +vh.dgen_ds(
            self.woutdata["rmnc"].T, self.s, self.th, self.ph, ms, ns, typ="c"
        )
        dz_ds = +vh.dgen_ds(
            self.woutdata["zmns"].T, self.s, self.th, self.ph, ms, ns, typ="s"
        )
        dr_dth = -vh.genvar_modi(
            self.woutdata["rmnc"].T, self.th, self.ph, ms, ns, ms, typ="s"
        )
        dz_dth = +vh.genvar_modi(
            self.woutdata["zmns"].T, self.th, self.ph, ms, ns, ms, typ="c"
        )
        dr_dph = +vh.genvar_modi(
            self.woutdata["rmnc"].T, self.th, self.ph, ms, ns, ns, typ="s"
        )
        dz_dph = -vh.genvar_modi(
            self.woutdata["zmns"].T, self.th, self.ph, ms, ns, ns, typ="c"
        )
        sph = np.sin(self.ph)
        cph = np.cos(self.ph)
        self.vecs["e_s"] = np.array(
            [dr_ds * cph[None, None, :], dr_ds * sph[None, None, :], dz_ds]
        )
        self.vecs["e_th"] = np.array(
            [dr_dth * cph[None, None, :], dr_dth * sph[None, None, :], dz_dth]
        )
        self.vecs["e_ph"] = np.array(
            [
                dr_dph * cph[None, None, :]
                - self.vars["R"] * sph[None, None, :],
                dr_dph * sph[None, None, :]
                + self.vars["R"] * cph[None, None, :],
                dz_dph,
            ]
        )
        self.vecs["grad_s"] = (
            np.cross(
                self.vecs["e_th"], self.vecs["e_ph"], axisa=0, axisb=0, axisc=0
            )
            / self.vars["sqrt_g"]
        )
        self.vecs["grad_th"] = (
            np.cross(
                self.vecs["e_ph"], self.vecs["e_s"], axisa=0, axisb=0, axisc=0
            )
            / self.vars["sqrt_g"]
        )
        self.vecs["grad_ph"] = (
            np.cross(
                self.vecs["e_s"], self.vecs["e_th"], axisa=0, axisb=0, axisc=0
            )
            / self.vars["sqrt_g"]
        )

    def get_var_old(self, var: str, s_idx=None):
        # key: var, [[varname, fun]], half_mesh
        self.vars[var] = self.transform_old(
            key=var_keys[var][0], half_mesh=var_keys[var][1], s_idx=s_idx
        )
        return self.vars[var]

    def transform_old(self, key, half_mesh=False, s_idx=None):
        hm_str = ["", "_nyq"][half_mesh]
        ms = self.woutdata["xm" + hm_str]
        ns = self.woutdata["xn" + hm_str]
        if s_idx == None:
            s_idx = slice(s_idx)
        else:
            s_idx = np.array([s_idx])
        for typ in "sc":
            fnmt = key + typ
            if fnmt in self.woutdata.keys():
                out = vh.genvar(
                    self.woutdata[fnmt][s_idx].T, self.th, self.ph, ms, ns, typ
                )
        return out

    def wrap_invert_fourier(self, xmn, deriv_order=0, deriv_dir="", kind=""):
        if xmn.shape[1] == self.nm_tot:
            is_nyq = False
        elif xmn.shape[1] == self.nm_nyq_tot:
            is_nyq = True
        else:
            raise ValueError("Incorrect shape")
        if is_nyq:
            return invert_fourier(
                xmn,
                self.xm_nyq,
                self.xn_nyq,
                self.n_th,
                self.n_ph,
                deriv_order,
                deriv_dir,
                kind,
            )
        else:
            return invert_fourier(
                xmn,
                self.xm,
                self.xn,
                self.n_th,
                self.n_ph,
                deriv_order,
                deriv_dir,
                kind,
            )
