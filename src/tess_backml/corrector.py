import os
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits
from tesscube import TESSCube

from . import log
from .utils import interp_1d, interp_2d


class ScatterLightCorrector:
    """
    A class to handle scatter light correction for TESS data.

    Parameters
    ----------
    sector : int
        The TESS sector number.
    camera : int
        The TESS camera number.
    ccd : int
        The TESS CCD number.
    fname : str, optional
        Path to the FITS file containing the scatter light cube. If None, a default
        path is constructed based on the sector, camera, and CCD.
    """

    def __init__(self, sector: int, camera: int, ccd: int, fname: Optional[str] = None):
        self.sector = sector
        self.camera = camera
        self.ccd = ccd

        self.rmin, self.rmax = 0, 2048
        self.cmin, self.cmax = 45, 2093
        self.btjd0 = 2457000

        if fname is None:
            # fname = f"./data/ffi_sl_cube_sector{self.sector:03}_{self.camera}-{self.ccd}.fits"
            raise ValueError(
                "Input file name is not valid, please provide a valid file."
            )

        if not os.path.isfile(fname):
            raise FileNotFoundError(f"SL cube file not found {fname}")

        hdul = fits.open(fname)

        if self.sector != hdul[0].header["SECTOR"]:
            raise ValueError("Requested sector does not match data in file")
        if self.camera != hdul[0].header["CAMERA"]:
            raise ValueError("Requested camera does not match data in file")
        if self.ccd != hdul[0].header["CCD"]:
            raise ValueError("Requested CCD does not match data in file")

        self.time_binned = hdul[3].header["BINNED"]
        self.time_binsize = hdul[0].header["TIMBINS"]
        self.cube_shape = (
            hdul[0].header["TIMSIZE"],
            hdul[0].header["IMGSIZEY"],
            hdul[0].header["IMGSIZEX"],
        )
        self.image_binsize = hdul[0].header["PIXBIN"]
        self.cube_time = hdul[3].data["time"]
        self.cube_flux = hdul[1].data.T[0]
        self.cube_fluxerr = hdul[1].data.T[1]
        self.pixel_counts = hdul[2].data
        self.tmin = hdul[0].header["TSTART"] + self.btjd0
        self.tmax = hdul[0].header["TSTOP"] + self.btjd0

        self.cube_row = (
            np.arange(self.rmin, self.rmax, self.image_binsize) + self.image_binsize / 2
        )
        self.cube_col = (
            np.arange(self.cmin, self.cmax, self.image_binsize) + self.image_binsize / 2
        )

    def __repr__(self):
        """Return a string representation of the ScatterLightCorrector object."""
        return f"TESS FFI SL Corrector (Sector, Camera, CCD): {self.sector}, {self.camera}, {self.ccd}"

    def get_original_ffi_times(self) -> np.ndarray:
        """
        Retrieve the original frame times from FFIs.

        Returns
        -------
        np.ndarray
            Array of original frame times in JD format.
        """
        tcube = TESSCube(sector=self.sector, camera=self.camera, ccd=self.ccd)
        return tcube.time + self.btjd0

    def _interpolate_pixel(
        self, row_eval: np.ndarray, col_eval: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate the SL model data into the provided pixel grid.

        Parameters
        ----------
        row_eval : np.ndarray
            Array of row indices for evaluation.
        col_eval : np.ndarray
            Array of column indices for evaluation.

        Returns
        -------
        flux_pix_interp : np.ndarray
            The interpolated flux data. Shape is (ntimes, len(row_evla), len(col_eval)).
        fluxerr_pix_interp : np.ndarray
            The interpolated flux error data. Shape is (ntimes, len(row_evla), len(col_eval)).
        """
        # iterate over frames to do 2d pixel interpolation
        flux_pix_interp = []
        fluxerr_pix_interp = []
        for tdx in range(len(self.cube_flux_rel)):
            # print(self.col_cube_rel.shape, self.row_cube_rel.shape, self.cube_flux_rel[tdx].T.shape)
            flx = interp_2d(
                x=self.col_cube_rel,
                y=self.row_cube_rel,
                z=self.cube_flux_rel[tdx].T,
                x_eval=col_eval,
                y_eval=row_eval,
            ).T
            flux_pix_interp.append(flx)
            flxe = interp_2d(
                x=self.col_cube_rel,
                y=self.row_cube_rel,
                z=self.cube_fluxerr_rel[tdx].T,
                x_eval=col_eval,
                y_eval=row_eval,
            ).T
            fluxerr_pix_interp.append(flxe)

        flux_pix_interp = np.array(flux_pix_interp)
        fluxerr_pix_interp = np.array(fluxerr_pix_interp)

        return flux_pix_interp, fluxerr_pix_interp

    def _interpolate_times(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate the SL model data into the provided times.

        Parameters
        ----------
        times : np.ndarray
            Array of times for evaluation.

        Returns
        -------
        flux_time_inter : np.ndarray
            The interpolated flux data. Shape is (len(times), *cube_spatial_shape).
        fluxerr_time_inter : np.ndarray
            The interpolated flux error data. Shape is (len(times), *cube_spatial_shape).
        """
        # compute output shape
        out_shape = (
            len(times),
            self.cube_flux_rel.shape[1],
            self.cube_flux_rel.shape[2],
        )

        # do time interp for each pixel
        flux_time_inter = []
        fluxerr_time_inter = []
        for flx, flxe in zip(
            self.cube_flux_rel.reshape((self.cube_flux_rel.shape[0], -1)).T,
            self.cube_fluxerr_rel.reshape((self.cube_fluxerr_rel.shape[0], -1)).T,
        ):
            flx_i = interp_1d(x=self.cube_time_rel, y=flx, x_eval=times)
            flux_time_inter.append(flx_i)
            flxe_i = interp_1d(x=self.cube_time_rel, y=flxe, x_eval=times)
            fluxerr_time_inter.append(flxe_i)
        # reshape arrays into 3D
        flux_time_inter = np.array(flux_time_inter).T.reshape(out_shape)
        fluxerr_time_inter = np.array(fluxerr_time_inter).T.reshape(out_shape)

        return flux_time_inter, fluxerr_time_inter

    def _get_relevant_pixels_and_times(
        self,
        row_eval: np.ndarray,
        col_eval: np.ndarray,
        times: np.ndarray,
    ):
        """
        Retrieve the relevant pixels and times for interpolation.

        Parameters
        ----------
        row_eval : np.ndarray
            Array of row indices for evaluation.
        col_eval : np.ndarray
            Array of column indices for evaluation.
        times : np.ndarray
            Array of times for evaluation.
        """
        # find the cube time range that contains the evaluation times
        dt = 3 # minimum n times in cube to do 3rd deg interp
        aux_idx = np.where(self.cube_time >= times.min())[0]
        if len(aux_idx) == 0:
            # special case when evluating at the end of the cube
            ti = len(self.cube_time) - 1 - dt
        else:
            ti = np.maximum(aux_idx[0] - dt, 0)
        aux_idx = np.where(self.cube_time <= times.max())[0]
        if len(aux_idx) == 0:
            # special case when evluating at the begining of the cube
            tf = dt + ti
        else:
            tf = np.minimum(aux_idx[-1] + dt, len(self.cube_time))
        log.info(f"time index range [{ti}:{tf}]")

        # find the cube pixel row/col range that contains the evaluation pixels
        dxy = 2
        ri = np.maximum(np.where(self.cube_row >= row_eval.min())[0][0] - dxy, 0)
        rf = np.minimum(
            np.where(self.cube_row <= row_eval.max())[0][-1] + dxy,
            self.cube_shape[1] - 1,
        )
        ci = np.maximum(np.where(self.cube_col >= col_eval.min())[0][0] - dxy, 0)
        cf = np.minimum(
            np.where(self.cube_col <= col_eval.max())[0][-1] + dxy,
            self.cube_shape[2] - 1,
        )
        # check we have > 3 points in each axis so 2d pixel interp can use deg=3
        if rf - ri < 4:
            ri = np.maximum(ri - 1, 0)
            rf = np.minimum(rf + 1, self.cube_shape[1] - 1)
        if cf - ci < 4:
            ci = np.maximum(ci - 1, 0)
            cf = np.minimum(cf + 1, self.cube_shape[2] - 1)
        log.info(f"[row,col] range  [{ri}:{rf}, {ci}:{cf}]")
        # assign segments of the SL cube for interp
        # have to make copies to ensure the original SL cube/times/row/col
        # are not changed and can be used for other evaluation grids with
        # the same corrector obeject.
        # `cube_flux_rel` and `cube_fluxerr_rel` are updated inplaced by
        # the interpolation operations
        self.cube_flux_rel = self.cube_flux[ti:tf, ri:rf, ci:cf].copy()
        self.cube_fluxerr_rel = self.cube_fluxerr[ti:tf, ri:rf, ci:cf].copy()
        self.cube_time_rel = self.cube_time[ti:tf].copy()
        self.row_cube_rel = self.cube_row[ri:rf].copy()
        self.col_cube_rel = self.cube_col[ci:cf].copy()
        # we keep a copy of the original section of the SL cube for later ref/plotting
        self.cube_flux_rel_org = self.cube_flux_rel.copy()

        return

    def evaluate_scatterlight_model(
        self,
        row_eval: np.ndarray,
        col_eval: np.ndarray,
        times: np.ndarray,
        method: str = "sl_cube",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the scatter light model and compute SL fluxes at given pixels and times.

        Parameters
        ----------
        row_eval : np.ndarray
            Array of row indices for evaluation.
        col_eval : np.ndarray
            Array of column indices for evaluation.
        times : np.ndarray
            Array of times for evaluation.
        method : str, optional
            Method to use for evaluation. Options are "sl_cube" or "nn". Default is "sl_cube".

        Returns
        -------
        sl_flux : np.ndarray
            Scatter light flux at the specified pixels and times.
        sl_fluxerr : np.ndarray
            Scatter light flux error at the specified pixels and times.

        Raises
        ------
        ValueError
            If the evaluation grid or times are out of bounds, or if an invalid method is specified.
        NotImplementedError
            If the "nn" method is selected (not implemented).

        Notes
        -----
        (5/9/2025)
        The errors returned by the `sl_cube` method, which are computed by interpolating
        a downsize version of the SL errors agregation done in `tess_backml.BacgroundData`
        is not a propper account for the real uncertainties of the evaluated SL flux,
        as the this does not account for the uncertainties of interpolation modeling.
        A better interpolation model could be done by using a linear modeling as done in
        PSFMachine. This option will be developed soon.
        """
        # check inputs are arrays
        if not isinstance(row_eval, np.ndarray) or not isinstance(col_eval, np.ndarray):
            raise ValueError("Pixel row/column for evaluation has to be a numpy array")

        # check row/cl inputs are within CCD range
        if (
            (row_eval.min() < self.rmin)
            or (row_eval.max() > self.rmax)
            or (col_eval.min() < self.cmin)
            or (col_eval.max() > self.cmax)
        ):
            raise ValueError(
                f"The evaluation pixel grid must be within CCD range [{self.rmin}:{self.rmax}, {self.cmin},{self.cmax}]"
            )
        # check times are within sector observing times
        if (times.min() < self.tmin) or (times.max() > self.tmax):
            raise ValueError(
                f"Evaluation times must be within observing times [{self.tmin:.5f}:{self.tmax:.5f}]"
            )

        # do interp from SL cube
        if method == "sl_cube":
            # get SLcube pixels near the the evaluation grid
            self._get_relevant_pixels_and_times(
                row_eval=row_eval, col_eval=col_eval, times=times
            )
            # do time interp
            if self.time_binned:
                self.cube_flux_rel, self.cube_fluxerr_rel = self._interpolate_times(
                    times=times
                )
                # safekeeping copy of time interpolated low-pixel-res of the cube
                self.cube_flux_rel_times = self.cube_flux_rel.copy()
            # do pixel interp
            sl_flux, sl_fluxerr = self._interpolate_pixel(
                row_eval=row_eval, col_eval=col_eval
            )
            # THIS IS NOT CORRRECT
            # account for pixel binning error prop
            sl_fluxerr *= self.image_binsize

        # use NN model for evaluation
        elif method == "nn":
            raise NotImplementedError
        else:
            raise ValueError("Invalid method, must be one of [sl_cube, nn]")

        return sl_flux, sl_fluxerr
