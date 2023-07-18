import matplotlib.pyplot as plt
import numpy as np
import vmec_library as vl

vfile = "/home/pedro/Documents/tmp/vmecs/tjii_0.10/wout_tj2_eccdw_01.nc"

thetas = np.linspace(0, 2 * np.pi, 30)
phis = np.linspace(0, 0.25 * np.pi, 20)

vmec = vl.Vmec(vfile, theta=thetas, phi=phis)

vmec.print_sizes()
# vmec.get_vectors_new()
# print(vmec.get_var_new("lmns", "s"))
# vmec.get_vectors()
vmec.get_xyzs()
vmec.get_vectors()

vmec_plotter = vl.Plotter(vmec)
fig, ax = vl.plot.make_figax_3d()
vmec_plotter.plot_slice(np.s_[::-10, :, 0], ax=ax, color="b", alpha=0.1)
vmec_plotter.plot_slice(np.s_[::-10, 4, :], ax=ax, color="g", alpha=0.1)
vmec_plotter.plot_slice(np.s_[-1, :, :], ax=ax, color="gray", alpha=0.1)
for idph in [0, 5]:
    for idth in range(0, len(vmec.th)):
        vmec_plotter.plot_vecbase(
            idth, idph, ax=ax, length=0.05, s_idx=-1, normalize=True
        )
# print(vmec.vars["R"], vmec.vars["R"].shape)

plt.show()
