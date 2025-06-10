import datetime
import os
import pickle
import warnings
from typing import Optional, Tuple

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import constants as const
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy import ndimage
from tesscube import TESSCube
from tqdm import tqdm

from . import PACKAGEDIR, __version__, log
from .utils import animate_cube, fill_nans_interp, get_tess_vectors, has_bit, pooling_2d

warnings.filterwarnings("ignore", category=RuntimeWarning)

camccd_orient = {
    "cam1": {
        1: "top left",
        2: "top left",
        3: "bottom right",
        4: "bottom right",
    },
    "cam2": {
        1: "top left",
        2: "top left",
        3: "bottom right",
        4: "bottom right",
    },
    "cam3": {
        1: "bottom right",
        2: "bottom right",
        3: "top left",
        4: "top left",
    },
    "cam4": {
        1: "bottom right",
        2: "bottom right",
        3: "top left",
        4: "top left",
    },
}

camccd_order = {
    "cam1": [1, 2, 4, 3],
    "cam2": [1, 2, 4, 3],
    "cam3": [3, 4, 2, 1],
    "cam4": [3, 4, 2, 1],
}


class BackgroundCube(object):
    """
    Class for creating TESS Full Frame Image background cubes to train a deep learning
    model that predicts the scatter light.

    It uses `tesscube` to retrieve FFI cubes from MAST/AWS, does spatia binning to
    downsize the 2k x 2k image, e.g. to 128 x 128 pixels.
    It uses `tessvectors` to obtain Earth/Moon angles/distances with respect to each
    TESS camera and creates a pixel map for each object angle and distance with the
    same resolution as the downsize FFI cube.

    Package the data into local files or returns batches fot ML training.

    Parameters
    ----------
    sector : int, optional
        TESS sector number. Must be between 1 and 100. Default is 1.
    camera : int, optional
        TESS camera number. Must be between 1 and 4. Default is 1.
    ccd : int, optional
        TESS CCD number. Must be between 1 and 4. Default is 1.
    img_bin : int, optional
        Binning factor for spatial downsampling of the FFI image.
        Must be a divisor of 2048. Default is 16. #
    downsize : str, optional
        Method for spatial downsampling. Options are 'binning' (median binning)
        or 'sparse' (select sparse pixels). Default is "binning".
    """

    def __init__(
        self,
        sector: int = 1,
        camera: int = 1,
        ccd: int = 1,
        img_bin: int = 16,
        downsize: str = "binning",
    ):
        """ """
        self.rmin, self.rmax = 0, 2048
        self.cmin, self.cmax = 45, 2093
        self.btjd0 = 2457000

        if sector not in range(1, 100):
            raise ValueError("Sector must be a valid number between 1 and 100")
        if camera not in range(1, 5):
            raise ValueError("Camera must be a valid number between 1 and 4")
        if ccd not in range(1, 5):
            raise ValueError("Sector must be a valid number between 1 and 4")

        if 2048 % img_bin != 0:
            raise ValueError("The bin factor `img_bin` for the image must divide 2048")

        if downsize not in ["sparse", "binning"]:
            raise ValueError("The `downsize` mode must be one of ['sparse', 'binning']")

        self.sector = sector
        self.camera = camera
        self.ccd = ccd
        self.img_bin = img_bin
        self.time_binned = False
        self.downsize = downsize
        self.tess_pscale = 0.21 * u.arcsec / u.pixel
        self.tess_psize = 15 * u.micrometer

        self.tcube = TESSCube(sector=sector, camera=camera, ccd=ccd)
        self.time = self.tcube.time + self.btjd0
        self.cadenceno = self.tcube.cadence_number
        self.ffi_size = 2048

        self._make_quality_mask()
        # apply quality mask to remove bad frames
        self.time = self.time[~self.quality]
        self.cadenceno = self.cadenceno[~self.quality]
        self.nt = len(self.time)
        # NOTE: self.tcube will have the original number of total frames for future ref

    def __repr__(self):
        """Return a string representation of the BackgroundCube object."""
        return f"TESS FFI Background object (Sector, Camera, CCD, N times): {self.sector}, {self.camera}, {self.ccd}, {self.nt}"

    def _make_quality_mask(self, bits: list = [3]):
        """
        Idetify specific bits in the quality mask and remove the cadences from the cubes

        Parameters
        ----------
        bits: list
            List of bits to be removed
        """
        mask = np.zeros(self.tcube.nframes).astype(bool)
        for bit in bits:
            mask |= has_bit(self.tcube.quality, bit=bit)
        self.quality = mask

    def _get_dark_frame_idx(self, low_per: float = 3.0):
        """
        Identifies indices of the darkest frames in the FFI time series.

        Attempts to load pre-computed indices from a local pickle file. If not
        found, it downloads sparse pixel time series, calculates a median
        background light curve, identifies the frames below a specified percentile,
        and saves these indices to the pickle file for future use.

        Parameters
        ----------
        low_per : float, optional
            The lower percentile threshold used to identify dark frames based on
            the median background light curve. Default is 3.0.
        """
        # check if dark frames are in file
        darkframe_file = f"{PACKAGEDIR}/data/dark_frame_indices_tess_ffi.pkl"
        if os.path.isfile(darkframe_file):
            with open(darkframe_file, "rb") as f:
                frames = pickle.load(f)
                if self.sector in frames.keys():
                    if f"{self.camera}-{self.ccd}" in frames[self.sector].keys():
                        self.dark_frames = frames[self.sector][
                            f"{self.camera}-{self.ccd}"
                        ]
                        self.darkest_frame = self.dark_frames[0]
                        return
                else:
                    frames[self.sector] = {}
        else:
            frames = {}
            frames[self.sector] = {}

        # if not pre computed, we have to find darkest frames by downloading some
        # FFI pixels and making a median LC
        step = 16
        srow = np.arange(
            self.rmin + step / 2,
            self.rmax - step / 2,
            step,
            dtype=int,
        )
        scol = np.arange(
            self.cmin + step / 2,
            self.cmax - step / 2,
            step,
            dtype=int,
        )
        srow_2d, scol_2d = np.meshgrid(srow, scol, indexing="ij")
        sparse_rc = [(r, c) for r, c in zip(srow_2d.ravel(), scol_2d.ravel())][::2]
        pix_ts = self.tcube.get_pixel_timeseries(sparse_rc, return_errors=False)

        # reshape as [ntimes, npix]
        pix_ts = pix_ts.reshape((self.tcube.nframes, -1))
        # take median
        self.bkg_lc = np.nanmedian(pix_ts, axis=-1)
        # find darkes and < 3% frame indices
        dark_frames = np.where(self.bkg_lc <= np.percentile(self.bkg_lc, low_per))[0]
        # we remove bad frames from the list
        dark_frames = dark_frames[
            np.isin(dark_frames, np.where(self.quality)[0], invert=True)
        ]
        # take the top 10 frames
        self.dark_frames = dark_frames[np.argsort(self.bkg_lc[dark_frames])][:10]
        # take the darkest
        self.darkest_frame = self.dark_frames[0]

        # we update the local file to cache the results
        frames[self.sector][f"{self.camera}-{self.ccd}"] = self.dark_frames
        with open(darkframe_file, "wb") as f:
            pickle.dump(frames, f)

        return

    def _get_star_mask(self, sigma: float = 5.0, dilat_iter: int = 2):
        """
        Computes a mask to identify star pixels in the darkest frame.

        The mask is created by sigma-clipping the darkest frame image and its
        gradient. The resulting mask is then dilated to encompass surrounding pixels.

        Parameters
        ----------
        sigma : float, optional
            The sigma threshold used for sigma clipping both the image and its
            gradient. Default is 5.0.
        dilat_iter : int, optional
            Number of iterations for binary dilation to expand the mask.
            Default is 2.
        """
        self.ffi_dark = self.tcube.get_ffi(self.darkest_frame)[1].data[
            self.rmin : self.rmax, self.cmin : self.cmax
        ]
        grad = np.hypot(*np.gradient(self.ffi_dark))
        star_mask = (
            sigma_clip(self.ffi_dark, sigma=sigma).mask
            & sigma_clip(grad, sigma=sigma).mask
        )
        self.star_mask = ndimage.binary_dilation(star_mask, iterations=dilat_iter)

        return

    def _get_straps_mask(self, dilat_iter: int = 1):
        """
        Creates a mask for the CCD strap locations based on predefined column indices.

        Loads strap column indices from a CSV file (`data/straps.csv`), creates
        a boolean mask for these columns, and optionally dilates the mask.

        Parameters
        ----------
        dilat_iter : int, optional
            Number of iterations for binary dilation to expand the strap mask.
            Default is 1.
        """
        # load straps column indices from file
        straps = pd.read_csv(f"{PACKAGEDIR}/data/straps.csv", comment="#")
        self.strap_mask = np.zeros((2048, 2048)).astype(bool)
        # the indices in the file are 1-based in the science portion of the ccd
        self.strap_mask[:, straps["Column"].values - 1] = True
        if dilat_iter > 0:
            self.strap_mask = ndimage.binary_dilation(
                self.strap_mask, iterations=dilat_iter
            )

        return

    def plot_dark_frame(self, mask_straps: bool = False):
        """
        Displays diagnostic plots of the darkest frame and associated masks.

        Shows the darkest FFI frame, the generated star mask, and optionally
        the strap mask.

        Parameters
        ----------
        mask_straps : bool, optional
            If True, also fetches and plots the strap mask. Requires the
            `_get_straps_mask` method to have been called or will call it.
            Default is False.
        """
        vmin, vmax = np.percentile(self.ffi_dark.ravel(), [3, 97])

        if mask_straps:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        else:
            fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        ax[0].set_title("Darkest frame FFI")
        bar = ax[0].imshow(self.ffi_dark, origin="lower", vmin=vmin, vmax=vmax)
        plt.colorbar(bar, ax=ax[0], shrink=0.8, label="Flux [-e/s]")

        ax[1].set_title(r"Star mask 5$\sigma$")
        bar = ax[1].imshow(self.star_mask, origin="lower", vmin=0, vmax=1)
        plt.colorbar(bar, ax=ax[1:], shrink=0.8)
        ax[1].set_yticks([])

        if mask_straps:
            ax[2].set_title("Strap Mask")
            bar = ax[2].imshow(self.strap_mask, origin="lower", vmin=0, vmax=1)
            ax[2].set_yticks([])
            # plt.colorbar(bar, ax=ax[2], shrink=0.8)
        plt.show()
        return

    def get_scatter_light_cube(
        self,
        plot: bool = False,
        mask_straps: bool = True,
        frames: Optional[Tuple] = None,
        rolling: bool = True,
        errors: bool = True,
    ):
        """
        Computes the scattered light cube by processing FFIs.

        This method orchestrates the process of:
        1. Identifying dark frames (`_get_dark_frame_idx`).
        2. Creating a star mask (`_get_star_mask`).
        3. Optionally creating a strap mask (`_get_straps_mask`).
        4. Defining background pixels (excluding stars and optionally straps).
        5. Calculating the static scene by median-combining dark frames (`_get_static_scene`).
        6. Iterating through FFIs (all or a specified range):
           - Fetching FFI data.
           - Applying the background pixel mask (masking stars/straps).
           - Subtracting the static scene.
           - Downsampling the result using the specified `downsize` method ('binning' or 'sparse').
        7. Storing the final scattered light cube.

        Parameters
        ----------
        plot : bool, optional
            If True, displays the diagnostic dark frame plots after mask creation.
            Default is False.
        mask_straps : bool, optional
            If True, masks out pixels corresponding to CCD straps in addition
            to stars. Default is False.
        frames : Optional[Tuple], optional
            Specifies a range of frame indices to process. Can be:
            - (N,): Process frames from 0 up to (but not including) N.
            - (start, stop): Process frames from index `start` up to `stop`.
            - (start, stop, step): Process frames from `start` to `stop` with `step`.
            If None, processes all frames. Default is None.
        rolling: bool, optional
            If True, pooling downsizing will be done with an iterative rolling windown and stride
            that will match the output desired shape, this will make the downsizing step slower.
            If False, pooling downsizing will use fixed window and stride.
        errors : bool, optional
            Retrieve and propagate errors
        """
        # get dark frame
        log.info("Computing sector darkest frames...")
        self._get_dark_frame_idx()
        # get star mask
        log.info("Computing star mask...")
        self._get_star_mask(sigma=5.0, dilat_iter=2)
        self.bkg_pixels = ~self.star_mask
        # mask out straps
        if mask_straps:
            dilat_iter = 2 if self.img_bin >= 12 else 1
            self._get_straps_mask(dilat_iter=dilat_iter)
            self.bkg_pixels &= ~self.strap_mask
        if plot:
            self.plot_dark_frame(mask_straps=mask_straps)

        srow = np.arange(
            self.rmin + self.img_bin / 2,
            self.rmax - self.img_bin / 2,
            self.img_bin,
            dtype=int,
        )
        scol = np.arange(
            self.cmin + self.img_bin / 2,
            self.cmax - self.img_bin / 2,
            self.img_bin,
            dtype=int,
        )

        self.row_2d, self.col_2d = np.meshgrid(srow, scol, indexing="ij")
        log.info("Getting FFI flux cube...")
        # get flux cube with downsampling
        if self.downsize == "sparse":
            # sparse pixels across the CCD
            sparse_rc = [
                (r, c) for r, c in zip(self.row_2d.ravel(), self.col_2d.ravel())
            ]
            flux_cube = self.tcube.get_pixel_timeseries(sparse_rc, return_errors=False)
            flux_cube = flux_cube.reshape((self.tcube.nframes, *self.row_2d.shape))
            flux_cube = flux_cube[~self.quality]
            pixel_counts = np.ones(flux_cube.shape[1:], dtype=int)

        elif self.downsize == "binning":
            times = int(np.log2(self.img_bin))
            # all pixels the binning with seld.
            flux_cube = []
            if errors:
                flux_e_cube = []
            # make a static scene from median of darkes frames
            if not hasattr(self, "static"):
                self.static = self._get_static_scene()
            mask_pixels = ~self.bkg_pixels
            # create time index array to iterate when asked for specific frames
            if isinstance(frames, tuple):
                if len(frames) == 1:
                    fi, ff, step = 0, frames, 1
                elif len(frames) == 2:
                    fi, ff, step = frames[0], frames[1], 1
                elif len(frames) == 3:
                    fi, ff, step = frames[0], frames[1], frames[2]
                frange = range(fi, ff, step)
            else:
                frange = range(0, self.tcube.nframes)
            for f in tqdm(frange, desc="Iterating frames"):
                # skip bad cadences
                if self.quality[f]:
                    continue
                current = self.tcube.get_ffi(f)[1].data[
                    self.rmin : self.rmax, self.cmin : self.cmax
                ]
                current[mask_pixels] = np.nan
                current -= self.static
                if rolling:
                    for k in range(times):
                        current = pooling_2d(
                            current,
                            kernel_size=4,
                            stride=2,
                            stat=np.nanmedian,
                            padding=1,
                        )
                else:
                    current = pooling_2d(
                        current,
                        kernel_size=self.img_bin,
                        stride=self.img_bin,
                        stat=np.nanmedian,
                    )
                if errors:
                    # collect errors
                    err = self.tcube.get_ffi(f)[2].data[
                        self.rmin : self.rmax, self.cmin : self.cmax
                    ]
                    err = pooling_2d(
                        err,
                        kernel_size=self.img_bin,
                        stride=self.img_bin,
                        stat=np.nanmean,
                    )
                    flux_e_cube.append(err)

                flux_cube.append(current)
            flux_cube = np.array(flux_cube)
            if len(flux_cube) != self.nt:
                aux = np.zeros((self.nt, flux_cube.shape[1], flux_cube.shape[2]))
                aux[fi:ff:step] = flux_cube
                flux_cube = aux
            # count the number of valid pixels in each bin
            # when rolling is True, the number of pixels that contribuite to a bin is not easy to
            # define due to the stride + window side + rolling. In this case we simplify and use
            # the full size + stride to approximate the pixel counts.
            pixel_counts = np.ones((self.rmax - self.rmin, self.cmax - self.cmin))
            pixel_counts[mask_pixels] = np.nan
            pixel_counts = pooling_2d(
                pixel_counts,
                kernel_size=self.img_bin,
                stride=self.img_bin,
                stat=np.nansum,
            ).astype(int)
            # replace bins with zero count with sample values, but keep 0s where the resulting flux is nan
            nzeros = (pixel_counts == 0).sum()
            print(nzeros)
            pixel_counts[pixel_counts == 0] = np.random.choice(
                pixel_counts.ravel(), nzeros
            )
            pixel_counts[np.isnan(flux_cube[0])] = 0

        else:
            log.info("Wrong pixel grid option...")

        if np.isnan(flux_cube).any():
            log.info("Filling nans with interpolation...")
            flux_cube = fill_nans_interp(flux_cube)

        self.scatter_cube = flux_cube.astype(np.float32)
        self.pixel_counts = pixel_counts.astype(np.int64)
        if errors:
            # propagate errors
            self.scatter_err_cube = np.array(flux_e_cube)
            # use aprox when dist is close to normal
            self.scatter_err_cube = 1.253 * (
                self.scatter_err_cube / np.sqrt(pixel_counts)
            )
        else:
            # if no error collection we use Poison approx
            self.scatter_err_cube = np.sqrt(flux_cube) / pixel_counts
        self.scatter_err_cube = self.scatter_err_cube.astype(np.float32)

        return

    def _get_static_scene(self):
        """
        Computes the static astrophysical scene by median-combining dark frames.

        Fetches the FFI data for the pre-identified dark frames, crops them to
        the active CCD area, and calculates the median frame along the time axis.

        Requires Attributes
        -------------------
        dark_frames : np.ndarray
            Indices of the dark frames to use.
        tcube : tesscube.TESSCube
            TESSCube object to fetch FFI data.
        rmin, rmax, cmin, cmax : int
            Indices defining the active CCD area.
        """
        log.info("Computing average static scene from darkes frames...")
        static = np.median(
            [
                self.tcube.get_ffi(f)[1].data[
                    self.rmin : self.rmax, self.cmin : self.cmax
                ]
                for f in self.dark_frames
            ],
            axis=0,
        )

        return static

    def _get_object_vectors(self, object: str = "Earth", ang_size: bool = True):
        """
        Calculates distance, altitude, and azimuth maps for Earth or Moon.

        For each pixel in the downsampled grid and for each time step, computes
        the distance, altitude (elevation angle), and azimuth angle of the specified
        celestial object ('Earth' or 'Moon') relative to that pixel. It uses
        trigonometric approximations based on the object's vector relative to the
        camera boresight (obtained from `tessvectors`).

        Parameters
        ----------
        object : str, optional
            The celestial object to calculate vectors for. Must be 'Earth' or 'Moon'.
            Default is "Earth".
        ang_size : bool, optional
            If True, the 'dist' map represents the angular size of the object
            as seen from each pixel (in degrees). If False, it represents the
            physical distance (in meters, needs unit handling). Default is True.
        """
        if object not in ["Earth", "Moon"]:
            raise ValueError("Object must be one of ['Earth', 'Moon']")

        # make a low res pixel grid with physical units
        grid_row, grid_col = np.mgrid[
            self.rmin : self.rmax : self.img_bin, self.cmin : self.cmax : self.img_bin
        ]
        grid_row_d = (grid_row * self.tess_psize).to("m")
        grid_col_d = (grid_col * self.tess_psize).to("m")

        # we need to flip some axis to add or subtract with respect to the
        # camera's boresight
        # this implementation asumes CCD origins are in the camera's boresight for
        # simplicity. We flip the value maps later to account for CCd orientations.
        if self.camera in [1, 2]:
            if self.ccd == 1:
                grid_col_d *= +1
                grid_row_d *= -1
            if self.ccd == 2:
                grid_col_d *= -1
                grid_row_d *= -1
            if self.ccd == 3:
                grid_col_d *= -1
                grid_row_d *= +1
            if self.ccd == 4:
                grid_row_d *= +1
                grid_col_d *= +1
        if self.camera in [3, 4]:
            if self.ccd == 3:
                grid_col_d *= +1
                grid_row_d *= -1
            if self.ccd == 4:
                grid_col_d *= -1
                grid_row_d *= -1
            if self.ccd == 1:
                grid_col_d *= -1
                grid_row_d *= +1
            if self.ccd == 2:
                grid_row_d *= +1
                grid_col_d *= +1

        object_alt_map = []
        object_az_map = []
        object_dist_map = []

        # iterate over frames to compute the maps
        # the new Alt/Az/Dist values are calculated following trigonoetric rules using
        # # the original values w.r.t the camera's boresight.
        for t in tqdm(range(len(self.vectors)), total=len(self.vectors)):
            if self.quality[t]:
                continue
            dist = self.vectors[f"{object}_Distance"][t] * const.R_earth
            aux_elev = np.sin(
                np.deg2rad(self.vectors[f"{object}_Camera_Angle"][t] - 90)
            ) + grid_row_d.to("m") / dist.to("m")
            aux_elev = np.rad2deg(np.arcsin(aux_elev)).value + 90
            object_alt_map.append(aux_elev)

            cos_ip = np.cos(np.deg2rad(aux_elev - 90))
            cos_i = np.cos(np.deg2rad(self.vectors[f"{object}_Camera_Angle"][t] - 90))
            sin_az = np.cos(
                np.deg2rad(self.vectors[f"{object}_Camera_Azimuth"][t] - 180)
            )
            aux_az = (cos_i * sin_az) / (cos_ip) + (grid_col_d.to("m")) / (
                dist.to("m") * cos_ip
            )
            aux_az = np.rad2deg(np.arcsin(aux_az)).value + 180
            object_az_map.append(aux_az)

            ang_dist = np.sqrt(grid_row**2 + grid_col**2) * u.pixel * self.tess_pscale
            pix_dist = (np.sqrt(grid_row**2 + grid_col**2) * 15 * u.micrometer).to("m")
            aux_dist = dist.to("m") * np.cos(ang_dist.to("rad"))

            aux_dist += np.sqrt(
                pix_dist**2 + (dist.to("m") * np.sin(ang_dist.to("rad"))) ** 2
            ) * np.sign(grid_row_d)
            aux_dist[aux_dist.value < 1000] = 0
            if ang_size:
                aux_dist = 2 * np.arctan(const.R_earth.to("m") / (2 * aux_dist))
                aux_dist = aux_dist.to("deg").value
                aux_dist[aux_dist == 180] = np.nan
            object_dist_map.append(aux_dist)

        object_alt_map = np.array(object_alt_map)
        object_az_map = np.array(object_az_map)
        object_dist_map = np.array(object_dist_map)

        if hasattr(self, "scatter_cube"):
            if object_alt_map.shape != self.scatter_cube.shape:
                raise ValueError(
                    "The resulting maps have different shape as the scatter cube "
                    f"{object_alt_map.shape}!= {self.scatter_cube.shape}"
                )

        # we need to flip the value maps to account for cam/CCD orientations
        if self.camera in [1, 2]:
            if self.ccd == 1:
                object_alt_map = np.flip(object_alt_map, axis=1)
                object_alt_map = np.flip(object_alt_map, axis=2)
                object_az_map = np.flip(object_az_map, axis=1)
                object_az_map = np.flip(object_az_map, axis=2)
                object_dist_map = np.flip(object_dist_map, axis=1)
                object_dist_map = np.flip(object_dist_map, axis=2)
            if self.ccd == 2:
                object_alt_map = np.flip(object_alt_map, axis=1)
                object_az_map = np.flip(object_az_map, axis=1)
                object_dist_map = np.flip(object_dist_map, axis=1)
            if self.ccd == 3:
                object_alt_map = np.flip(object_alt_map, axis=1)
                object_alt_map = np.flip(object_alt_map, axis=2)
                object_az_map = np.flip(object_az_map, axis=1)
                object_az_map = np.flip(object_az_map, axis=2)
                object_dist_map = np.flip(object_dist_map, axis=1)
                object_dist_map = np.flip(object_dist_map, axis=2)
            if self.ccd == 4:
                object_alt_map = np.flip(object_alt_map, axis=1)
                object_az_map = np.flip(object_az_map, axis=1)
                object_dist_map = np.flip(object_dist_map, axis=1)
        if self.camera in [3, 4]:
            if self.ccd == 3:
                object_alt_map = np.flip(object_alt_map, axis=1)
                object_alt_map = np.flip(object_alt_map, axis=2)
                object_az_map = np.flip(object_az_map, axis=1)
                object_az_map = np.flip(object_az_map, axis=2)
                object_dist_map = np.flip(object_dist_map, axis=1)
                object_dist_map = np.flip(object_dist_map, axis=2)
            if self.ccd == 4:
                object_alt_map = np.flip(object_alt_map, axis=1)
                object_az_map = np.flip(object_az_map, axis=1)
                object_dist_map = np.flip(object_dist_map, axis=1)
            if self.ccd == 1:
                object_alt_map = np.flip(object_alt_map, axis=1)
                object_alt_map = np.flip(object_alt_map, axis=2)
                object_az_map = np.flip(object_az_map, axis=1)
                object_az_map = np.flip(object_az_map, axis=2)
                object_dist_map = np.flip(object_dist_map, axis=1)
                object_dist_map = np.flip(object_dist_map, axis=2)
            if self.ccd == 2:
                object_alt_map = np.flip(object_alt_map, axis=1)
                object_az_map = np.flip(object_az_map, axis=1)
                object_dist_map = np.flip(object_dist_map, axis=1)

        return {
            "dist": np.array(object_dist_map),
            "alt": np.array(object_alt_map),
            "az": np.array(object_az_map),
        }

    def get_vector_maps(self, ang_size: bool = True):
        """
        Generates pixel-wise maps of Earth and Moon positions and sizes.

        Fetches spacecraft orientation and Earth/Moon position vectors using
        `tessvectors`. Then, for both Earth and Moon, it calls `_get_object_vectors`
        to compute 3D maps (time, row, col) representing the distance/angular size,
        altitude, and azimuth for each pixel in the downsampled grid over time.
        Also stores the boresight vectors for reference.

        Parameters
        ----------
        ang_size : bool, optional
            If True, the 'dist' maps and vectors represent angular size (degrees).
            If False, they represent physical distance ([m] for maps).
            Default is True.
        """
        self.vectors = get_tess_vectors(
            cadence="FFI", sector=self.sector, camera=self.camera
        )

        self.earth_maps = self._get_object_vectors(object="Earth", ang_size=ang_size)
        self.earth_vectors = {
            "dist": (self.vectors["Earth_Distance"].values * const.R_earth)
            .to("m")
            .value[~self.quality],
            "alt": self.vectors["Earth_Camera_Angle"].values[~self.quality],
            "az": self.vectors["Earth_Camera_Azimuth"].values[~self.quality],
        }
        self.moon_maps = self._get_object_vectors(object="Moon", ang_size=ang_size)
        self.moon_vectors = {
            "dist": (self.vectors["Earth_Distance"].values * const.R_earth)
            .to("m")
            .value[~self.quality],
            "alt": self.vectors["Earth_Camera_Angle"].values[~self.quality],
            "az": self.vectors["Earth_Camera_Azimuth"].values[~self.quality],
        }
        if ang_size:
            self.earth_vectors["dist"] = 2 * np.arctan(
                const.R_earth.to("m").value / (2 * self.earth_vectors["dist"])
            )
            self.earth_vectors["dist"] *= 180.0 / np.pi
            self.moon_vectors["dist"] = 2 * np.arctan(
                const.R_earth.to("m").value / (2 * self.moon_vectors["dist"])
            )
            self.moon_vectors["dist"] *= 180.0 / np.pi

        return

    def bin_time_axis(self, bin_size: float = 2.5):
        """
        Performs temporal binning on the calculated data cubes and vectors.

        If `self.time_bin` is greater than 1, this method bins the time axis
        of `scatter_cube`, `time`, `cadenceno`, and all Earth/Moon vector maps
        and boresight vectors by taking the mean or median within each time bin.

        Parameters
        ----------
        bin_size : float, optional
            Bin size for time axis in units of hours (e.g. 2.5 hours).
        """
        diff = np.diff(self.time)
        if bin_size <= np.median(diff):
            raise ValueError(
                f"Bin size must be larger than the median elapsed time between observations {np.median(diff)*24:.2f} H."
            )

        # first find data gaps due to downlink
        gaps = np.where(diff > 3 * np.median(diff))[0] + 1
        # avoid bad frames
        bad_frames = np.where(self.quality)[0]

        # compute indices in each bin for time aggregation
        indices = []
        # we do binning per orbits to account for data discontinuity
        for no, orb in enumerate(np.array_split(np.arange(len(self.time)), gaps)):
            # bin eges in time units
            time_binedges = np.arange(
                self.time[orb].min(), self.time[orb].max(), bin_size / 24.0
            )
            # find indices within time bin
            for s, f in zip(time_binedges[:-1], time_binedges[1:]):
                idx = np.where((self.time >= s) & (self.time < f))[0]
                idx = idx[np.isin(idx, bad_frames, invert=True)]
                indices.append(idx)
            # add remaining indices at the end of orbit
            indices.append(np.arange(indices[-1][-1] + 1, orb[-1] + 1))

        # agregate cubes/arrays in the time axis
        self.scatter_cube_bin = np.array(
            [np.mean(self.scatter_cube[x], axis=0) for x in indices]
        )
        self.scatter_err_cube_bin = np.array(
            [np.mean(self.scatter_err_cube[x], axis=0) for x in indices]
        )
        self.time_bin = np.array([np.mean(self.time[x], axis=0) for x in indices])
        self.cadenceno_bin = np.array(
            [np.mean(self.cadenceno[x], axis=0) for x in indices]
        )

        if hasattr(self, "earth_maps") and hasattr(self, "earth_vectors"):
            self.earth_vectors_bin = {}
            self.earth_maps_bin = {}
            self.moon_vectors_bin = {}
            self.moon_maps_bin = {}
            for key in ["dist", "alt", "az"]:
                self.earth_vectors_bin[key] = np.array(
                    [np.mean(self.earth_vectors[key][x], axis=0) for x in indices]
                )
                self.earth_maps_bin[key] = np.array(
                    [np.mean(self.earth_maps[key][x], axis=0) for x in indices]
                )
                self.moon_vectors_bin[key] = np.array(
                    [np.mean(self.moon_vectors[key][x], axis=0) for x in indices]
                )
                self.moon_maps_bin[key] = np.array(
                    [np.mean(self.moon_maps[key][x], axis=0) for x in indices]
                )
        self.time_binned = True
        self.time_binsie = bin_size

        return

    def save_to_fits(self, out_file: Optional[str] = None, binned: bool = True):
        """
        Saves the downsize version of the scatter light cube as a FITS file similar to
        the MAST FFI cubes.

        Parameters
        ----------
        out_file : Optional[str], optional
            Path to the output .npz file. If None, a default filename is
            generated based on sector, camera, ccd, and binning factor, saved
            in the current directory. Default is None.

        Returns
        -------
        hdul : HDUList object
            Header unit list with data and metadata
        """
        if self.time_binned and binned:
            cube_sets = np.array([self.scatter_cube_bin, self.scatter_err_cube_bin])
            time_col = self.time_bin
        else:
            cube_sets = np.array([self.scatter_cube, self.scatter_err_cube])
            time_col = self.time

        priheader = self.tcube.primary_hdu.header.copy()
        del (
            priheader["NAXIS1"],
            priheader["NAXIS2"],
            priheader["NAXIS3"],
            priheader["NAXIS4"],
        )

        prihdu = fits.PrimaryHDU(header=priheader)

        prihdu.header["ORIGIN"] = (
            "NASA/GSFC",
            "Institution responsible for creating this file",
        )
        prihdu.header["DATE"] = (
            datetime.date.today().isoformat(),
            "file creation date",
        )
        prihdu.header["OBJECT"] = ("Scatter Light cube", "Type of data")
        prihdu.header["CREATOR"] = ("tess-backml", "Software of origin")
        prihdu.header["PROCVER"] = (__version__, "Software versin")

        prihdu.header["pixbin"] = (self.img_bin, "Bin size in pixel space")
        prihdu.header["pixbinm"] = ("median", "Method of binning in pixel space")
        prihdu.header["imgsizex"] = (cube_sets.shape[2], "Image size in axis X")
        prihdu.header["imgsizey"] = (cube_sets.shape[3], "Image size in axis Y")
        prihdu.header["timbins"] = (self.time_binsie, "[h] bin size in time axis")
        prihdu.header["timbinm"] = ("mean", "Method of binning in time")
        prihdu.header["timsize"] = (cube_sets.shape[1], "Cube size in time axis")

        imghdu = fits.ImageHDU(
            cube_sets.T.astype(np.float32), name="Scatter Light Cube"
        )
        pcthdu = fits.ImageHDU(self.pixel_counts, name="Pixel Counts")

        timhdu = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="time", array=time_col, format="D", unit="jd"),
            ]
        )
        timhdu.header["binned"] = (self.time_binned, "Is cube binned in time")
        timhdu.header["EXTNAME"] = "TIME"

        hdul = fits.HDUList([prihdu, imghdu, pcthdu, timhdu])
        if out_file is None:
            return hdul
        else:
            hdul.writeto(out_file, overwrite=True)
            return

    def save_to_npz(self, out_file: Optional[str] = None, save_maps: bool = False):
        """
        Saves the processed background data to a NumPy .npz file.

        Parameters
        ----------
        out_file : Optional[str], optional
            Path to the output .npz file. If None, a default filename is
            generated based on sector, camera, ccd, and binning factor, saved
            in the current directory. Default is None.
        save_maps : bool, optional
            If True, saves the detailed pixel-wise maps for Earth and Moon
            angles/distances alongside the scatter cube and boresight vectors.
            If False, saves only the scatter cube, time, cadence, and boresight
            vectors (smaller file size). Default is False.
        """
        if out_file is None:
            out_file = f"./ffi_cubes_bin{self.img_bin}_sector{self.sector:03}_{self.camera}-{self.ccd}.npz"
            log.info(f"Saving to {out_file}")
            if save_maps:
                out_file = out_file.replace("cubes", "sl")

        if save_maps:
            np.savez(
                out_file,
                scatter_cube=self.scatter_cube,
                time=self.time,
                cadenceno=self.cadenceno,
                earth_alt=self.earth_vectors["alt"],
                earth_az=self.earth_vectors["az"],
                earth_dist=self.earth_vectors["dist"],
                moon_alt=self.moon_vectors["alt"],
                moon_az=self.moon_vectors["az"],
                moon_dist=self.moon_vectors["dist"],
                earth_alt_map=self.earth_maps["alt"],
                earth_az_map=self.earth_maps["az"],
                earth_dist_map=self.earth_maps["dist"],
                moon_alt_map=self.moon_maps["alt"],
                moon_az_map=self.moon_maps["az"],
                moon_dist_map=self.moon_maps["dist"],
            )
        else:
            np.savez(
                out_file,
                scatter_cube=self.scatter_cube,
                time=self.time,
                cadenceno=self.cadenceno,
            )
        return

    def animate_data(
        self,
        data: str = "sl",
        step: int = 10,
        file_name: Optional[str] = None,
        save: bool = False,
    ):
        """
        Creates and optionally saves an animation of the processed data cubes.

        Generates an animation showing the time evolution of either the scattered
        light cube or one of the Earth/Moon angle/distance maps.

        Parameters
        ----------
        data : str, optional
            Which data cube to animate. Options are:
            - 'sl': Scattered light cube (`scatter_cube`).
            - 'earth_alt': Earth altitude map (`earth_maps['alt']`).
            - 'earth_az': Earth azimuth map (`earth_maps['az']`).
            - 'earth_dist': Earth distance/angular size map (`earth_maps['dist']`).
            - 'moon_alt', 'moon_az', 'moon_dist': Corresponding Moon maps.
            Default is "sl".
        step : int, optional
            Frame step for the animation (e.g., step=10 shows every 10th frame).
            Default is 10.
        file_name : Optional[str], optional
            Filename for saving the animation as a GIF. If None and `save` is True,
            a default filename is generated. Default is None.
        save : bool, optional
            If True, saves the animation to the specified `file_name` (or default).
            If False, attempts to display the animation directly (requires IPython).
            Default is False.

        Returns
        -------
        IPython.display.HTML or None
            If `save` is False and IPython is available, returns an HTML object
            for displaying the animation. Otherwise returns None.
        """
        if data == "sl":
            plot_cube = self.scatter_cube
            title = "Scatter Light"
            cbar_label = ("Flux [e-/s]",)
        elif data in ["sl_tbin", "sl_bin"]:
            plot_cube = self.scatter_cube_bin
            title = "Scatter Light"
            cbar_label = ("Flux [e-/s]",)
        elif data in ["earth_alt", "earth_elev"]:
            plot_cube = (
                self.earth_maps["alt"] / self.earth_vectors["alt"][:, None, None]
            )
            title = "Earth Elevation Angle"
            cbar_label = "Angle [normalized]"
        elif data == "earth_az":
            plot_cube = self.earth_maps["az"] / self.earth_vectors["az"][:, None, None]
            title = "Earth Azimuth Angle"
            cbar_label = "Angle [normalized]"
        elif data == "earth_dist":
            plot_cube = (
                self.earth_maps["dist"] / self.earth_vectors["dist"][:, None, None]
            )
            title = "Earth Angular Size"
            cbar_label = "Angular Size [normalized]"
        else:
            raise ValueError("`cube` must be une of [sl, bkg].")

        # Create animation
        ani = animate_cube(
            plot_cube,
            cadenceno=self.cadenceno,
            time=self.time,
            interval=50,
            plot_type="img",
            extent=(self.cmin - 0.5, self.cmax - 0.5, self.rmin - 0.5, self.rmax - 0.5),
            step=step,
            suptitle=f"{title} Sector {self.sector} Camera {self.camera} CCD {self.ccd}",
            bar_label=cbar_label,
        )

        # Save animation
        if save:
            # Create default file name
            if file_name is None:
                file_name = f"./ffi_{data}_bin{self.img_bin}_sector{self.sector:03}_{self.camera}-{self.ccd}.gif"
            # Check format of file_name and outdir
            if not file_name.endswith(".gif"):
                raise ValueError(f"`file_name` must be a .gif file. Not `{file_name}`")

            ani.save(file_name, writer="pillow")
            return

        try:
            from IPython.display import HTML

            return HTML(ani.to_jshtml())
        except ModuleNotFoundError as err:
            # To make installing `tess-asteroids` easier, ipython is not a dependency
            # because we can assume it is installed when notebook-specific features are called.
            raise err(
                "ipython needs to be installed for animate() to work (e.g., `pip install ipython`)"
            )

        return
