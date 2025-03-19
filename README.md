# TESS Back ML

This is a Python package to create training data to be used for a neural network (NN) 
model that predicts the TESS Full Frame Image (FFI) background signal, in particular,
 the time-changing scattered light.

This animation shows the scattered light of a TESS FFI. The original 2048 x 2048 pixel 
image was downsized to 128 x 128 pixels to be memory efficient.

![scatt_cube](./data/figures/ffi_flux_cube_bin16_sector002_1-3_clippix_median.gif){width=60%}


The next figure shows the vector maps (distance, elevation, and azimuth angles) for 
Earth with respect to the camera boresight. 
Note that the distance maps have a discontinuity between the upper and lower CCDs 
that needs to be fixed.

![earth_maps](./data/figures/earth_vector_maps.png){width=60%}