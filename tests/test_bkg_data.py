import numpy as np
from tess_backml import BackgroundCube


def test_BackgroundCube():
    sector = 2
    camera = 2
    ccd = 4

    bc = BackgroundCube(
        sector=sector, camera=camera, ccd=ccd, img_bin=16, downsize="binning"
    )

    assert bc.nt == 1235
    assert bc.sector == sector
    assert bc.camera == camera
    assert bc.ccd == ccd

    assert len(bc.time) == len(bc.cadenceno)
    assert bc.ffi_size == 2048

    bc._get_dark_frame_idx()

    assert all(
        bc.dark_frames == np.array([626, 627, 629, 628, 630, 631, 632, 633, 634, 635])
    )
    assert bc.darkest_frame == 626

    bc._get_star_mask()
    assert bc.star_mask.shape == (2048, 2048)
    assert bc.star_mask.sum() == 447320

    bc._get_straps_mask()
    assert bc.strap_mask.shape == (2048, 2048)
    assert bc.strap_mask.sum() == 466944
