import os
from typing import Optional

import numpy as np
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline, interp1d
from tesscube import TESSCube
from tqdm import tqdm

from . import log


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
            fname = f"./data/ffi_sl_cube_sector{self.sector:03}_{self.camera}-{self.ccd}.fits"

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
        self.cube_sl = hdul[1].data.T[0]
        self.cube_sle = hdul[1].data.T[1]
        self.tmin = hdul[0].header["TSTART"] + self.btjd0
        self.tmax = hdul[0].header["TSTOP"] + self.btjd0

        self.row_cube = (
            np.arange(self.rmin, self.rmax, self.image_binsize) + self.image_binsize / 2
        )
        self.col_cube = (
            np.arange(self.cmin, self.cmax, self.image_binsize) + self.image_binsize / 2
        )

    def __repr__(self):
        """Return a string representation of the ScatterLightCorrector object."""
        return f"TESS FFI SL Corrector (Sector, Camera, CCD): {self.sector}, {self.camera}, {self.ccd}"

    def get_original_ffi_times(self):
        """
        Retrieve the original frame times from FFIs.

        Returns
        -------
        None
        """
        tcube = TESSCube(sector=self.sector, camera=self.camera, ccd=self.ccd)
        self.ffi_times = tcube.time + self.btjd0

    def _interpolate_pixel(
        self, row_eval: np.ndarray, col_eval: np.ndarray
    ) -> np.ndarray:
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
        np.ndarray
            Interpolated SL model data for the specified pixel grid.
        """
        sl_eval_pix = []
        for tdx in tqdm(range(len(self.cube_sl_rel)), desc="Pixel interp"):
            interp2d = RectBivariateSpline(
                self.col_cube_rel, self.row_cube_rel, self.cube_sl_rel[tdx].T, kx=3, ky=3
            )
            sl_eval_pix.append(interp2d(col_eval, row_eval).T)

        sl_eval_pix = np.array(sl_eval_pix)
        return sl_eval_pix

    def _interpolate_times(self, times: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Interpolate the SL model data into the provided times.

        Parameters
        ----------
        times : np.ndarray, optional
            Array of times for evaluation.

        Returns
        -------
        np.ndarray
            Interpolated SL model data for the specified times.
        """
        out_shape = (len(times), self.cube_sl_rel.shape[1], self.cube_sl_rel.shape[2])

        sl_time_inter = []
        for pix in tqdm(
            self.cube_sl_rel.reshape((self.cube_sl_rel.shape[0], -1)).T,
            desc="Time interp",
        ):
            fx = interp1d(
                self.cube_time_rel, pix, kind="slinear", bounds_error=False, fill_value="extrapolate"
            )(times)
            sl_time_inter.append(fx)
        sl_time_inter = np.array(sl_time_inter).T.reshape(out_shape)

        return sl_time_inter

    def _get_relevant_pixels_and_times(
        self,
        row_eval: np.ndarray,
        col_eval: np.ndarray,
        times: Optional[np.ndarray] = None,
    ):
        """
        Retrieve the relevant pixels and times for interpolation.

        Parameters
        ----------
        row_eval : np.ndarray
            Array of row indices for evaluation.
        col_eval : np.ndarray
            Array of column indices for evaluation.
        times : np.ndarray, optional
            Array of times for evaluation.
        """
        dt = 2
        ti = np.maximum(np.where(self.cube_time >= times.min())[0][0] - dt, 0)
        tf = np.minimum(
            np.where(self.cube_time <= times.max())[0][-1] + dt, len(self.cube_time)
        )
        log.info(f"time index range [{ti}:{tf}]")

        dxy = 2
        ri = np.maximum(
            np.where(self.row_cube >= row_eval.min())[0][0] - dxy, 0
        )
        rf = np.minimum(
            np.where(self.row_cube <= row_eval.max())[0][-1] + dxy, 
            self.cube_shape[1] - 1
        )
        ci = np.maximum(
            np.where(self.col_cube >= col_eval.min())[0][0] - dxy, 0
        )
        cf = np.minimum(
            np.where(self.col_cube <= col_eval.max())[0][-1] + dxy, 
            self.cube_shape[2] - 1
        )
        log.info(f"[row,col] range  [{ri}:{rf}, {ci}:{cf}]")
        self.cube_sl_rel = self.cube_sl[ti:tf, ri:rf, ci:cf].copy()
        self.cube_sl_rel_org = self.cube_sl_rel.copy()
        self.cube_time_rel = self.cube_time[ti:tf]
        self.row_cube_rel = self.row_cube[ri:rf]
        self.col_cube_rel = self.col_cube[ci:cf]
        return

    def evaluate_scatterlight_model(
        self,
        row_eval: np.ndarray,
        col_eval: np.ndarray,
        times: Optional[np.ndarray] = None,
        method: str = "sl_cube",
    ) -> np.ndarray:
        """
        Evaluate the scatter light model and compute SL fluxes at given pixels and times.

        Parameters
        ----------
        row_eval : np.ndarray
            Array of row indices for evaluation.
        col_eval : np.ndarray
            Array of column indices for evaluation.
        times : np.ndarray, optional
            Array of times for evaluation.
        method : str, optional
            Method to use for evaluation. Options are "sl_cube" or "nn". Default is "sl_cube".

        Returns
        -------
        np.ndarray
            Scatter light fluxes at the specified pixels and times.

        Raises
        ------
        ValueError
            If the evaluation grid or times are out of bounds, or if an invalid method is specified.
        NotImplementedError
            If the "nn" method is selected (not implemented).
        """
        if not isinstance(row_eval, np.ndarray) or not isinstance(col_eval, np.ndarray):
            raise ValueError("Pixel row/column for evaluation has to be a numpy array")

        if (
            (row_eval.min() < self.rmin)
            or (row_eval.max() > self.rmax)
            or (col_eval.min() < self.cmin)
            or (col_eval.max() > self.cmax)
        ):
            raise ValueError(
                f"The evaluation pixel grid must be within CCD range [{self.rmin}:{self.rmax}, {self.cmin},{self.cmax}]"
            )
        if (times.min() < self.tmin) or (times.max() > self.tmax):
            raise ValueError(
                f"Evaluation times must be within observing times [{self.tmin:.5f}:{self.tmax:.5f}]"
            )

        if method == "sl_cube":
            self._get_relevant_pixels_and_times(
                row_eval=row_eval, col_eval=col_eval, times=times
            )
            if self.time_binned:
                self.cube_sl_rel = self._interpolate_times(times=times)
                self.cube_sl_rel_times = self.cube_sl_rel.copy()
            sl_eval = self._interpolate_pixel(row_eval=row_eval, col_eval=col_eval)

        elif method == "nn":
            raise NotImplementedError
        else:
            raise ValueError("Invalid method, must be one of [sl_cube, nn]")

        return sl_eval
