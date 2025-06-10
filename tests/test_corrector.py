import os

import numpy as np
from tess_backml import PACKAGEDIR, ScatterLightCorrector

data_path = os.path.dirname(os.path.dirname(PACKAGEDIR))


def test_ScatterLightCorrector():
    sector = 2
    camera = 1
    ccd = 1
    fname = f"{data_path}/tests/data/ffi_sl_cube_sector002_1-1_test.fits"

    slcorr = ScatterLightCorrector(sector=sector, camera=camera, ccd=ccd, fname=fname)

    assert slcorr.sector == sector
    assert slcorr.camera == camera
    assert slcorr.ccd == ccd

    assert slcorr.cube_flux.shape == slcorr.cube_shape
    assert slcorr.cube_flux.shape == slcorr.cube_fluxerr.shape
    assert slcorr.time_binned
    assert slcorr.time_binsize == 2.0
    assert slcorr.image_binsize == 128
    assert slcorr.cube_time.shape == (20,)
    assert slcorr.cube_flux.shape == (20, 16, 16)

    assert slcorr.pixel_counts.shape == (16, 16)
    assert (slcorr.pixel_counts > 0).all()

    assert (slcorr.cube_row > 0).all()
    assert (slcorr.cube_row < 2050).all()
    assert (slcorr.cube_col > 0).all()
    assert (slcorr.cube_col < 2095).all()


def test_evaluate_scatterlight_model():
    sector = 2
    camera = 1
    ccd = 1
    fname = f"{data_path}/tests/data/ffi_sl_cube_sector002_1-1_test.fits"

    slcorr = ScatterLightCorrector(sector=sector, camera=camera, ccd=ccd, fname=fname)

    row_eval = np.arange(500, 505)
    col_eval = np.arange(1100, 1105)
    time_eval = np.linspace(slcorr.cube_time[0], slcorr.cube_time[-1], 12)
    # Test the evaluate_scatterlight_model method
    flux, fluxerr = slcorr.evaluate_scatterlight_model(
        row_eval=row_eval, col_eval=col_eval, times=time_eval, method="sl_cube"
    )

    assert flux.shape == (12, 5, 5)
    assert fluxerr.shape == (12, 5, 5)
    assert np.isfinite(flux).all()
