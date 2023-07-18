import numpy as np
from scipy.io import netcdf_file
from .helper import vh


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
        self.s = np.linspace(0, 1, self.woutdata["ns"])
        self.th = theta
        self.ph = phi
        return

    def read_woutdata(self, woutfile):
        with netcdf_file(woutfile, "r") as wfile:
            for var in wfile.variables:
                self.woutdata[var] = wfile.variables[var].data.copy()

    def print_vars(self):
        for key in sorted(self.woutdata.keys()):
            print(key)

    def print_sizes(self):
        for key in sorted(self.woutdata.keys()):
            print(key, self.woutdata[key].shape)

    def get_var(self, key, typ, s_idx=np.s_[:]):
        if self.woutdata[key].shape[1] == len(self.woutdata["xn"]):
            ms = self.woutdata["xm"]
            ns = self.woutdata["xn"]
        elif self.woutdata[key].shape[1] == len(self.woutdata["xn_nyq"]):
            ms = self.woutdata["xm_nyq"]
            ns = self.woutdata["xn_nyq"]
        else:
            raise (ValueError("Incorrect variable name?"))
        return vh.genvar(self.woutdata[key][s_idx].T, self.th, self.ph, ms, ns, typ=typ)

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
                dr_dph * cph[None, None, :] - self.vars["R"] * sph[None, None, :],
                dr_dph * sph[None, None, :] + self.vars["R"] * cph[None, None, :],
                dz_dph,
            ]
        )
        self.vecs["grad_s"] = (
            np.cross(self.vecs["e_th"], self.vecs["e_ph"], axisa=0, axisb=0, axisc=0)
            / self.vars["sqrt_g"]
        )
        self.vecs["grad_th"] = (
            np.cross(self.vecs["e_ph"], self.vecs["e_s"], axisa=0, axisb=0, axisc=0)
            / self.vars["sqrt_g"]
        )
        self.vecs["grad_ph"] = (
            np.cross(self.vecs["e_s"], self.vecs["e_th"], axisa=0, axisb=0, axisc=0)
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

    def get_axis(self):
        xs, ys, zs = self.get_xyzs_old(0)
        return xs[0], ys[0], zs[0]

    # def plot_surf(self, ax=None, s_idx=-1, quantity=None, **kwargs):
    #     xyzs = self.get_xyzs(s_idx)
    #     if ax is None:
    #         fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    #     if quantity is None:
    #         ax.plot_surface(*xyzs, rstride=1, cstride=1, antialiased=True, **kwargs)
    #     else:
    #         self.get_var(quantity)
    #         tmp = self.vars[quantity][s_idx]
    #         ax.plot_surface(
    #             *xyzs,
    #             rstride=1,
    #             cstride=1,
    #             cmap=plt.cm.jet,
    #             linewidth=0,
    #             antialiased=False,
    #             facecolors=plt.cm.jet(normalize(tmp)),
    #         )
    #     ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    #     equal_aspect(ax)
    #     return ax

    # def plot_cut(self, ax=None, s_idx=None, quantity=None, phi=0):
    #     phi_idx = np.argmin(np.abs(self.ph - phi * np.pi / 180))
    #     print(phi_idx)
    #     if s_idx is None:
    #         s_idx = np.array((np.linspace(0, 1, 7) ** 2) * (len(self.s) - 1), dtype=int)
    #     if ax is None:
    #         fig, ax = plt.subplots(1, 1)
    #     if quantity == None:
    #         for si in list(s_idx):
    #             rs = self.get_var("R", si)[0]
    #             zs = self.get_var("Z", si)[0]
    #             if si == 0:
    #                 ax.plot(rs[0, phi_idx], zs[0, phi_idx], "+r")
    #             ax.plot(rs[:, phi_idx], zs[:, phi_idx], "k")
    #     #   TODO:
    #     # else:
    #     #     for si in list(s_idx):
    #     #         rs = self.get_var('R', si)[0]
    #     #         zs = self.get_var('Z', si)[0]
    #     #         q  = self.get_var(quantity, si)[0]
    #     #         if si==0:
    #     #             ax.plot(rs[0,phi_idx], zs[0,phi_idx], '+r')
    #     #         ax.plot(rs[:,phi_idx], zs[:,phi_idx], 'k')
    #     ax.set(
    #         xlabel="R [m]",
    #         ylabel="Z [m]",
    #         title=rf"$\varphi$={180*self.ph[phi_idx]/np.pi:.2f}º",
    #     )
    #     ax.set_aspect("equal")
    #     return ax

    # def plot_profile(self, var, ax=None, **kwargs):
    #     if ax is None:
    #         fig, ax = plt.subplots(1, 1)
    #     ax.plot(self.s, self.woutdata[var], **kwargs)

    # def plot_axis(self, ax=None):
    #     xyzs = self.get_axis()
    #     if ax is None:
    #         fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    #     ax.plot(*xyzs, "r")
    #     equal_aspect(ax)
    #     return ax

    # def transform(self, keys, half_mesh=False, s_idx=None):
    #     if half_mesh == True:
    #         ms = self.vmec['xm_nyq']
    #         ns = self.vmec['xn_nyq']
    #     else:
    #         ms = self.vmec['xm']
    #         ns = self.vmec['xn']
    #     ph_g, th_g = np.meshgrid(self.ph, self.th)
    #     thms_phns = np.multiply.outer(th_g, ms) - np.multiply.outer(ph_g, ns)
    #     if s_idx is None:
    #         s = self.s
    #         out = np.zeros((len(s), len(self.th), len(self.ph)))
    #         for var, fun in keys:
    #             try:
    #                 var_nm = self.vmec[var]
    #                 out += np.inner(var_nm, fun(thms_phns))
    #             except:
    #                 pass
    #         return out
    #     else:
    #         out = np.zeros((len(self.th), len(self.ph)))
    #         for var, fun in keys:
    #             try:
    #                 var_nm = self.vmec[var]
    #                 out += np.inner(var_nm[s_idx], fun(thms_phns))
    #             except:
    #                 pass
    #         return out
