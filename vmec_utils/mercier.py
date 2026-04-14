from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import numpy as np
import requests


def read_lines(filename: str):
    with open(filename, "r") as f:
        lines = f.readlines()
    return lines


def get_from_url(url: str):
    response = requests.get(url)
    lines = response.text.splitlines()
    return lines


class Mercier:
    def __init__(self, filename: str = "", url: str = ""):
        if filename:
            lines = read_lines(filename)
        elif url:
            if not url.endswith("mercier.txt"):
                url = url + "mercier.txt"
            lines = get_from_url(url)
        top, bottom = [], []
        for line in lines:
            if "----" in line:
                continue
            if "IOTA" in line:
                tmp = top
                top_leg = line.split()
                continue
            elif "DMerc" in line:
                print("bottom")
                tmp = bottom
                bottom_leg = line.split()
                continue
            if line.split() == []:
                continue
            tmp.append(np.fromstring(line, np.float64, sep=" "))
        self.top = np.array(top)
        self.bottom = np.array(bottom)
        for idx, lab in enumerate(top_leg):
            lab = lab.lower().replace("'", "p")
            self.__setattr__(lab, self.top.T[idx])
        for idx, lab in enumerate(bottom_leg):
            lab = lab.lower().replace("'", "p")
            self.__setattr__(lab, self.bottom.T[idx])

    def plot(self):
        fig, ax = plt.subplots(3, 3, figsize=[6.69, 8])
        ax[0, 0].plot(
            self.s[1:], self.iota[1:], "r", label="iota"
        )  # Rotational Transform
        ax[0, 0].set(ylim=(0.6, 1.4))

        ax[0, 1].plot(self.s[1:], self.shear[1:], "--r", label="shear")  # d\iota / ds
        ax[0, 1].plot(self.s[1:], self.dshear[1:], "g", label="dshear")  # ?

        ax[0, 2].plot(self.s[1:], self.phi[1:], "k", label="phi")  # Toroidal flux
        ax[0, 2].plot(self.s[1:], self.vp[1:], "r", label="vp")

        ax[1, 0].plot(self.s[1:], self.pres[1:], "k", label="pres")

        ax[1, 1].plot(self.s[1:], self.itor[1:], "k", label="itor")

        ax[1, 2].plot(self.s[1:], self.presp[1:], "r", label="presp", alpha=0.5)
        ax[1, 2].plot(self.s[1:], self.itorp[1:], "r", label="itorp", alpha=0.5)

        ax[2, 0].axhline(0.25, color="r", ls="--", alpha=0.5)
        ax[2, 0].plot(self.s[1:], self.dmerc[1:], "k", label="dmerc")
        ax[2, 0].set(ylim=(-1, 1))

        ax[2, 1].plot(self.s[1:], self.well[1:], "k", label="well")

        ax[2, 2].plot(self.s[1:], self.dcurr[1:], "r", label="dcurr")
        ax[2, 2].plot(self.s[1:], self.dwell[1:], "r", label="dwell")
        ax[2, 2].plot(self.s[1:], self.dgeod[1:], "b", label="dgeod")

        [a.legend() for a in ax.ravel()]
        return fig, ax


if __name__ == "__main__":
    M = Mercier(
        askopenfilename(
            initialdir="auxfiles", filetypes=[("Mercier Files", "mercier.*")]
        )
    )
    M.plot()
    # plt.savefig('figs/tmp.pdf')
    plt.show()
