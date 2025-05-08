import os
from typing import Optional
import numpy as np
from astropy.io import fits
from tqdm import tqdm
# import fitsio
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline, interp1d
from tesscube import TESSCube

from . import log

class ScatterLightCorrector():

    def __init__(
        self, 
        sector: int, 
        camera: int, 
        ccd: int, 
        fname: Optional[str] = None
    ):

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
        self.cube_shape = (hdul[0].header["TIMSIZE"], hdul[0].header["IMGSIZEX"], hdul[0].header["IMGSIZEY"])
        self.image_binsize = hdul[0].header["PIXBIN"]
        self.cube_time = hdul[3].data["time"]
        self.cube_sl = hdul[1].data.T[0]
        self.cube_sle = hdul[1].data.T[1]
        self.tmin = hdul[0].header["TSTART"] + self.btjd0
        self.tmax = hdul[0].header["TSTOP"] + self.btjd0

        self.row_cube = np.arange(self.rmin, self.rmax, self.image_binsize) + self.image_binsize/2
        self.col_cube = np.arange(self.cmin, self.cmax, self.image_binsize) + self.image_binsize/2


    def get_original_ffi_times(self):
        """
        Helper function to get the original frame times from FFIs
        """
        tcube = TESSCube(sector=sector, camera=camera, ccd=ccd)
        self.ffi_times = tcube.time + self.btjd0

    
    def _interpolate_pixel(self, row_eval: np.ndarray, col_eval: np.ndarray) -> np.ndarray:
        """
        Evaluates the SL model by interpolating the data in `cube_sl` into the provided pixel grid
        and cadences
        """

        sl_eval_pix = []
        for tdx in tqdm(range(len(self.cube_sl_rel)), desc="Pixel interp"):
            interp2d = RectBivariateSpline(
                self.col_cube_rel, self.row_cube_rel, self.cube_sl_rel[tdx], kx=3, ky=3
            )
            sl_eval_pix.append(interp2d(col_eval, row_eval).T)

        # need to add errors
        sl_eval_pix = np.array(sl_eval_pix)
        return sl_eval_pix

    def _interpolate_times(self, times: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluates the SL model by interpolating the data in `cube_sl` into the provided pixel grid
        and times
        """
        # define output shape
        out_shape = (len(times), self.cube_sl_rel.shape[1], self.cube_sl_rel.shape[2])

        # iterate over pixel time series and interpolate
        sl_time_inter = []
        for pix in tqdm(self.cube_sl_rel.reshape((self.cube_sl_rel.shape[0], -1)).T, desc="Time interp"):
            fx = interp1d(self.cube_time_rel, pix, kind="slinear")(times)
            sl_time_inter.append(fx)
        sl_time_inter = np.array(sl_time_inter).reshape(out_shape)

        return sl_time_inter

    def _get_relevant_pixels_and_times(self, row_eval: np.ndarray, col_eval: np.ndarray, times: Optional[np.ndarray] = None):
        """
        Auxiliar function to retrieve the relevant pixels when interpolating from the SL cube.
        For example, if evaluating pixels [50:55, 1020:1025], we only need the pixels in the SL cube 
        that correspond to that portion of the CCD, so we can save memory and computing time
        
        """
        # find the start and and end of times in the downsize cube
        dt = 2
        ti = np.maximum(np.where(self.cube_time >= times.min())[0][0] - dt, 0)
        tf = np.minimum(np.where(self.cube_time <= times.max())[0][-1] + dt, len(self.cube_time))
        log.info(f"time index range [{ti}:{tf}]")

        # find the start and and end of pixel row/col in the downsize cube
        dxy = 2
        ri = np.maximum(np.where(self.row_cube >= row_eval.min())[0][0] - dxy, self.rmin)
        rf = np.minimum(np.where(self.row_cube <= row_eval.max())[0][-1] + dxy, self.rmax)
        ci = np.maximum(np.where(self.col_cube >= col_eval.min())[0][0] - dxy, self.cmin)
        cf = np.minimum(np.where(self.col_cube <= col_eval.max())[0][-1] + dxy, self.cmax)
        log.info(f"[row,col] range  [{ri}:{rf}, {ci}:{cf}]")
        self.cube_sl_rel = self.cube_sl[ti:tf, ri:rf, ci:cf].copy()
        self.cube_sl_rel_ = self.cube_sl_rel.copy()
        self.cube_time_rel = self.cube_time[ti:tf]
        self.row_cube_rel = self.row_cube[ri:rf]
        self.col_cube_rel = self.col_cube[ci:cf]
        return

    def get_scatterlight_model(self, row_eval: np.ndarray, col_eval: np.ndarray, times: Optional[np.ndarray] = None, method: str="sl_cube") -> np.ndarray:
        """
        Main function to evaluate the SL models and compute SL fluxes at given pixels/times
        """
        if not isinstance(row_eval, np.ndarray) or not isinstance(col_eval, np.ndarray):
            raise ValueError("Pixel row/column for evaluation has to be a numpy array")
        
        # we check pixel grid and times of eval are within CCD and sector observing times
        if (row_eval.min() < self.rmin) or (row_eval.max() > self.rmax) or (col_eval.min() < self.cmin) or (col_eval.max() > self.cmax):
            raise ValueError(
                f"The evaluation pixel grid must be within CCD range [{self.rmin}:{self.rmax}, {self.cmin},{self.cmax}]"
            )
        if (times.min() < self.tmin) or (times.max() > self.tmax):
            raise ValueError(f"Evaluation times must be within observing times [{self.tmin:.5f}:{self.tmax:.5f}]")
        
        if method == "sl_cube":
            self._get_relevant_pixels_and_times(row_eval=row_eval, col_eval=col_eval, times=times)
            if self.time_binned:
                self.cube_sl_rel = self._interpolate_times(times=times)
            sl_eval = self._interpolate_pixel(row_eval=row_eval, col_eval=col_eval)
            
            
        elif method == "nn":
            raise NotImplementedError
        else:
            raise ValueError("Invalid method, must be one of [sl_cube, nn]")

        return sl_eval
        
            
            