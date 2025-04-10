[![pytest](https://github.com/jorgemarpa/tess-backml/actions/workflows/pytest.yaml/badge.svg)](https://github.com/jorgemarpa/tess-backml/actions/workflows/pytest.yaml/) [![ruff](https://github.com/jorgemarpa/tess-backml/actions/workflows/ruff.yaml/badge.svg)](https://github.com/jorgemarpa/tess-backml/actions/workflows/ruff.yaml)[![Docs](https://github.com/jorgemarpa/tess-backml/actions/workflows/deploy-mkdocs.yaml/badge.svg)](https://github.com/jorgemarpa/tess-backml/actions/workflows/deploy-mkdocs.yaml)

# TESS Back ML

This is a Python package to create training data to be used for a neural network (NN) 
model that predicts the TESS Full Frame Image (FFI) background signal, in particular,
 the time-changing scattered light.

This animation shows the scattered light of a TESS FFI. The original 2048 x 2048 pixel 
image was downsized to 128 x 128 pixels to be memory efficient.

![scatt_cube](./docs/figures/ffi_scatterlight_bin16_sector001_3-4.gif)


The next figure shows the vector maps (distance, elevation, and azimuth angles) for 
Earth and Moon with respect to the camera boresight. These maps have the same shape as
the scatter light cube shown above.

![earth_maps](./docs/figures/earth_vector_maps.png)

## Install 

Install from this GitHub repository with

```
pip install git+https://github.com/jorgemarpa/tess-backml
```

PyPI will available shortly.

## Usage

To get the data follow the steps:

```python
from tess_backml import Background_Data

# initialize the object for given sector/camera/ccd
# will do 16x16 pixel binning
tess_bkg = BackgroundCube(
    sector=1, camera=1, ccd=1, img_bin=16, downsize="binning"
)

# get the flux data from MAST/AWS, compute scatter light and downsize
tess_bkg.get_scatter_light_cube(frames=None, mask_straps=True, plot=False)
# compute the vector maps for the Earth and Moon
tess_bkg.get_vector_maps(ang_size=True)

# make an animation of the scatter light cube
tess_bkg.animate_data(data="sl", save=False, step=10);

# save data to disk
tess_bkg.save_data(save_maps=True)
```

Or you can run a Python script in the terminal (plotting flag is optional and will add
run time):
```
python build_dataset.py --sector 3 --camera 1 --ccd 2 --image-bin 16 --downsize binning --plot
```

Also check out the Jupyter notebook tutorial [here](./docs/tutorial_1.ipynb).
