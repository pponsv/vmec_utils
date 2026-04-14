### Compilation:

- Create environment and install dependencies:

  The creation of a separate conda environment (here called _vmec_env_) is recommended. My Linux machine at IPP did not have an hdf5 version modern enough to install h5py.

        conda create -n vmec_env python hdf5 h5py meson-python
        conda activate vmec_env

- Install using pip.

  - If you do not want to make changes, just run

        pip install .

  - If you will be making changes, you can install in developer mode via

        pip install --no-build-isolation --editable .
