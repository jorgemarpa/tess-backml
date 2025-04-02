import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tessvectors

from tesscube import TESSCube
from tqdm import tqdm
from astropy.stats import sigma_clip
from scipy import ndimage
import astropy.units as u
from astropy import constants as const
from .utils import pooling_2d, animate_cube

from typing import Optional, Tuple
from .. import PACKAGEDIR

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


class Background_Data(object):
    """
    Class for creating TESS Full Frame Image background cubes to train a deep learning
    model that predicts the scatter light.

    It uses `tesscube` to retrieve FFI cubes from MAST/AWS, does spatia binning to
    downsize the 2k x 2k image, e.g. to 128 x 128 pixels.
    It uses `tessvectors` to obtain Earth/Moon angles/distances with respect to each
    TESS camera and creates a pixel map for each object angle and distance with the
    same resolution as the downsize FFI cube.

    Package the data into local files or returns batches fot ML training.
    """

    def __init__(
        self,
        sector: int = 1,
        camera: int = 1,
        ccd: int = 1,
        img_bin: int = 16,
        time_bin: int = 1,
        downsize: str = "binning",
    ):
        """
        Paramerters
        -----------
        """
        self.rmin, self.rmax = 0, 2048
        self.cmin, self.cmax = 45, 2093

        if not sector in range(1, 100):
            raise ValueError("Sector must be a valid number between 1 and 100")
        if not camera in range(1, 5):
            raise ValueError("Camera must be a valid number between 1 and 4")
        if not ccd in range(1, 5):
            raise ValueError("Sector must be a valid number between 1 and 4")

        if 2048 % img_bin != 0:
            raise ValueError("The bin factor `img_bin` for the image must divide 2048")

        if downsize not in ["sparse", "binning"]:
            raise ValueError("The `downsize` mode must be one of ['sparse', 'binning']")

        self.sector = sector
        self.camera = camera
        self.ccd = ccd
        self.img_bin = img_bin
        self.time_bin = time_bin
        self.downsize = downsize
        self.tess_pscale = 0.21 * u.arcsec / u.pixel
        self.tess_psize = 15 * u.micrometer

        self.tcube = TESSCube(sector=sector, camera=camera, ccd=ccd)
        self.time = self.tcube.time
        self.cadenceno = self.tcube.cadence_number
        self.nt = len(self.time)
        self.ffi_size = 2048

    def __repr__(self):
        return f"TESS FFI Background object (Sector, Camera, CCD, N times): {self.sector}, {self.camera}, {self.ccd}, {self.nt}"

    def _get_dark_frame_idx(self, low_per: float = 3.0):
        """
        Finds the indices of darkest frames.
        """
        # check if dark frames are in file
        darkframe_file = f"{PACKAGEDIR}/database/data/dark_frame_indices_tess_ffi.pkl"
        with open(darkframe_file, "rb") as f:
            frames = pickle.load(f)
            if self.sector in frames.keys():
                if f"{self.camera}-{self.ccd}" in frames[self.sector].keys():
                    self.dark_frames = frames[self.sector][f"{self.camera}-{self.ccd}"]
                    self.darkest_frame = self.dark_frames[0]
                    return
            else:
                frames[self.sector] = {}

        # if not pre computed, we have to find darkest frames by downloading some
        # FFI pixels and making a median LC
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
        srow_2d, scol_2d = np.meshgrid(srow, scol, indexing="ij")
        sparse_rc = [(r, c) for r, c in zip(srow_2d.ravel(), scol_2d.ravel())][::2]
        pix_ts = self.tcube.get_pixel_timeseries(sparse_rc, return_errors=False)

        # reshape as [ntimes, npix]
        pix_ts = pix_ts.reshape((self.tcube.nframes, -1))
        # take median
        self.bkg_lc = np.nanmedian(pix_ts, axis=-1)
        # find darkes and < 3% frame indices
        dark_frames = np.where(self.bkg_lc <= np.percentile(self.bkg_lc, low_per))[0]
        self.dark_frames = dark_frames[np.argsort(self.bkg_lc[dark_frames])][:10]
        self.darkest_frame = self.dark_frames[0]

        # we update the local file to cache the results
        frames[self.sector][f"{self.camera}-{self.ccd}"] = self.dark_frames
        with open(darkframe_file, "wb") as f:
            pickle.dump(frames, f)

        return

    def _get_star_mask(self, sigma: float = 5.0, dilat_iter: int = 2):
        """
        Computes a star mask using sigma clipping on the image and on the gradients
        of the image.
        Then uses dilation to enlarge the mask `dilat_iter` times.
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
        Gets straps locations from a file and creates a pixel mask
        """
        # load straps column indices from file
        straps = pd.read_csv(f"{PACKAGEDIR}/database/data/straps.csv", comment="#")
        self.strap_mask = np.zeros((2048, 2048)).astype(bool)
        # the indices in the file are 1-based in the science portion of the ccd
        self.strap_mask[:, straps["Column"].values - 1] = True
        self.strap_mask = ndimage.binary_dilation(self.strap_mask, iterations=dilat_iter)

        return

    def plot_dark_frame(self, mask_straps: bool = False):
        """
        Creates a diagnostic plot of the darkes frame and the star mask
        """
        vmin, vmax = np.percentile(self.ffi_dark.ravel(), [3, 97])

        if mask_straps:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        else:
            fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].set_title("Darkest frame FFI")
        bar = ax[0].imshow(self.ffi_dark, origin="lower", vmin=vmin, vmax=vmax)
        plt.colorbar(bar, ax=ax[0], shrink=0.8, label="Flux [-e/s]")

        ax[1].set_title(r"Star mask 5$\sigma$")
        bar = ax[1].imshow(self.star_mask, origin="lower", vmin=0, vmax=1)
        plt.colorbar(bar, ax=ax[1], shrink=0.8)

        if mask_straps:
            ax[2].set_title("Strap Mask")
            bar = ax[2].imshow(self.strap_mask, origin="lower", vmin=0, vmax=1)
            plt.colorbar(bar, ax=ax[2], shrink=0.8)
        plt.show()
        return

    def get_scatter_light_cube(
        self, 
        plot: bool = False, 
        mask_straps: bool = False, 
        frames: Optional[Tuple] = None,
    ):
        """
        Gets flux cube from MAST/AWS and does downsampling.
        Subtracts the static scene (from dark frames) to compute the 
        time-changing scatter light component only.
        """
        # get dark frame
        print("Computing sector darkest frames...")
        self._get_dark_frame_idx()
        # get star mask
        print("Computing star mask...")
        self._get_star_mask(sigma=5.0, dilat_iter=2)
        self.bkg_pixels = ~self.star_mask
        # mask out straps
        if mask_straps:
            self._get_straps_mask()
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
        print("Getting FFI flux cube...")
        # get flux cube with downsampling
        if self.downsize == "sparse":
            # sparse pixels across the CCD
            sparse_rc = [
                (r, c) for r, c in zip(self.row_2d.ravel(), self.col_2d.ravel())
            ]
            flux_cube = self.tcube.get_pixel_timeseries(sparse_rc, return_errors=False)
            flux_cube = flux_cube.reshape((self.tcube.nframes, *self.row_2d.shape))

        elif self.downsize == "binning":
            # all pixels the binning with seld.
            flux_cube = []
            self.static = self._get_static_scene()
            mask_pixels = ~self.bkg_pixels
            if isinstance(frames, tuple):
                if len(frames) == 1:
                    fi, ff, step = 0, frames, 1
                elif len(frames) == 2:
                    fi, ff, step = frames[0], frames[1], 1
                elif len(frames) == 3:
                    fi, ff, step = frames[0], frames[1], frames[2]
                frange = range(fi, ff, step)
            else:
                frange = range(0, self.nt)
            for f in tqdm(frange, desc="Iterating frames"):
                current = self.tcube.get_ffi(f)[1].data[
                    self.rmin : self.rmax, self.cmin : self.cmax
                ]
                current[mask_pixels] = np.nan
                current -= self.static

                flux_cube.append(
                    pooling_2d(
                        current,
                        kernel_size=self.img_bin,
                        stride=self.img_bin,
                        stat=np.nanmedian,
                    )
                )
            flux_cube = np.array(flux_cube)
            if len(flux_cube) != self.nt:
                aux = np.zeros((self.nt, flux_cube.shape[1], flux_cube.shape[2]))
                aux[fi:ff:step] = flux_cube
                flux_cube = aux

        else:
            print("Wrong pixel grid option...")

        self.scatter_cube = flux_cube
        # self.static = np.median(flux_cube[self.dark_frames], axis=0)
        # self.scatter_cube -= self.static
        
        return


    def _get_static_scene(self):
        """
        Computes a static scene of the cube by averaging the top 10 darkest frames
        """
        print("Computing average static scene from darkes frames...")
        static = np.median([self.tcube.get_ffi(f)[1].data[
                self.rmin : self.rmax, self.cmin : self.cmax
            ] for f in self.dark_frames], axis=0)
        
        return static
    

    def _get_object_vectors(self, object: str = "Earth", ang_size: bool = True):
        """
        Auxiliar function to get vector maps for an object
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

        # we need to flip the value maps to account for cam/CCD orientations
        if self.camera in [1, 2]:
            if self.ccd == 1:
                object_alt_map = np.flip(np.array(object_alt_map), axis=1)
                object_alt_map = np.flip(np.array(object_alt_map), axis=2)
                object_az_map = np.flip(np.array(object_az_map), axis=1)
                object_az_map = np.flip(np.array(object_az_map), axis=2)
                object_dist_map = np.flip(np.array(object_dist_map), axis=1)
                object_dist_map = np.flip(np.array(object_dist_map), axis=2)
            if self.ccd == 2:
                object_alt_map = np.flip(np.array(object_alt_map), axis=1)
                object_az_map = np.flip(np.array(object_az_map), axis=1)
                object_dist_map = np.flip(np.array(object_dist_map), axis=1)
            if self.ccd == 3:
                object_alt_map = np.flip(np.array(object_alt_map), axis=1)
                object_alt_map = np.flip(np.array(object_alt_map), axis=2)
                object_az_map = np.flip(np.array(object_az_map), axis=1)
                object_az_map = np.flip(np.array(object_az_map), axis=2)
                object_dist_map = np.flip(np.array(object_dist_map), axis=1)
                object_dist_map = np.flip(np.array(object_dist_map), axis=2)
            if self.ccd == 4:
                object_alt_map = np.flip(np.array(object_alt_map), axis=1)
                object_az_map = np.flip(np.array(object_az_map), axis=1)
                object_dist_map = np.flip(np.array(object_dist_map), axis=1)
        if self.camera in [3, 4]:
            if self.ccd == 3:
                object_alt_map = np.flip(np.array(object_alt_map), axis=1)
                object_alt_map = np.flip(np.array(object_alt_map), axis=2)
                object_az_map = np.flip(np.array(object_az_map), axis=1)
                object_az_map = np.flip(np.array(object_az_map), axis=2)
                object_dist_map = np.flip(np.array(object_dist_map), axis=1)
                object_dist_map = np.flip(np.array(object_dist_map), axis=2)
            if self.ccd == 4:
                object_alt_map = np.flip(np.array(object_alt_map), axis=1)
                object_az_map = np.flip(np.array(object_az_map), axis=1)
                object_dist_map = np.flip(np.array(object_dist_map), axis=1)
            if self.ccd == 1:
                object_alt_map = np.flip(np.array(object_alt_map), axis=1)
                object_alt_map = np.flip(np.array(object_alt_map), axis=2)
                object_az_map = np.flip(np.array(object_az_map), axis=1)
                object_az_map = np.flip(np.array(object_az_map), axis=2)
                object_dist_map = np.flip(np.array(object_dist_map), axis=1)
                object_dist_map = np.flip(np.array(object_dist_map), axis=2)
            if self.ccd == 2:
                object_alt_map = np.flip(np.array(object_alt_map), axis=1)
                object_az_map = np.flip(np.array(object_az_map), axis=1)
                object_dist_map = np.flip(np.array(object_dist_map), axis=1)

        return {
            "dist": np.array(object_dist_map),
            "alt": np.array(object_alt_map),
            "az": np.array(object_az_map),
        }

    def get_vector_maps(self, ang_size: bool = True):
        """
        Gets the tess space raft vectors using `tessvetors` which have the Earth/Moon
        angles and distances  with respect to each camera. Then computes maps with
        values for the angles and distance for each pixel in the downsize cube.
        """
        self.vectors = tessvectors.getvector(("FFI", self.sector, self.camera))

        self.earth_maps = self._get_object_vectors(object="Earth", ang_size=ang_size)
        self.earth_vectors = {
            "dist": (self.vectors["Earth_Distance"].values * const.R_earth).to("km").value,
            "alt": self.vectors["Earth_Camera_Angle"].values,
            "az": self.vectors["Earth_Camera_Azimuth"].values,
        }
        self.moon_maps = self._get_object_vectors(object="Moon", ang_size=ang_size)
        self.moon_vectors = {
            "dist": (self.vectors["Earth_Distance"].values * const.R_earth).to("km").value,
            "alt": self.vectors["Earth_Camera_Angle"].values,
            "az": self.vectors["Earth_Camera_Azimuth"].values,
        }
        if ang_size:
            self.earth_vectors["dist"] = 2 * np.arctan(const.R_earth.to("km").value / (2 * self.earth_vectors["dist"]))
            self.earth_vectors["dist"] *= 180. / np.pi
            self.moon_vectors["dist"] = 2 * np.arctan(const.R_earth.to("km").value / (2 * self.moon_vectors["dist"]))
            self.moon_vectors["dist"] *= 180. / np.pi

        return

    def time_bin(self):
        """
        Inplace binning of the data cubes in the time axis.
        """
        # bin in time if asked
        if self.time_bin > 1:
            indices = np.array_split(np.arange(self.nt), int(self.nt / self.time_bin))

            self.scatter_cube = np.array(
                [np.nanmedian(self.scatter_cube[x], axis=0) for x in indices]
            )
            self.time = np.array([np.mean(self.times[x], axis=0) for x in indices])
            self.cadenceno = np.array(
                [np.mean(self.cadenceno[x], axis=0) for x in indices]
            )

            for key in ["dist", "alt", "az"]:
                self.earth_vectors[key] = np.array(
                    [np.mean(self.earth_vectors[key][x], axis=0) for x in indices]
                )
                self.earth_maps[key] = np.array(
                    [np.mean(self.earth_maps[key][x], axis=0) for x in indices]
                )
                self.moon_vectors[key] = np.array(
                    [np.mean(self.moon_vectors[key][x], axis=0) for x in indices]
                )
                self.moon_maps[key] = np.array(
                    [np.mean(self.moon_maps[key][x], axis=0) for x in indices]
                )

        return

    def save_data(self, out_file: Optional[str] = None):
        """
        Save data to disk
        """
        if out_file is None:
            out_file = f"../../data/ffi_cube_bin{self.img_bin}_sector{self.sector:03}_{self.camera}-{self.ccd}.npz"
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
        return

    def animate_data(
        self,
        data: str = "sl",
        step: int = 10,
        file_name: Optional[str] = None,
        save: bool = False,
    ):
        """
        Makes an animation of the scatter light cube
        """
        if data == "sl":
            plot_cube = self.scatter_cube
            title = "Scatter Light"
            cbar_label = "Flux [e-/s]",
        elif data in ["earth_alt", "earth_elev"]:
            plot_cube = self.earth_maps["alt"] / self.earth_vectors["alt"][:, None, None]
            title = "Earth Elevation Angle"
            cbar_label = "Angle [nor,alized]"
        elif data == "earth_az":
            plot_cube = self.earth_maps["az"] / self.earth_vectors["az"][:, None, None]
            title = "Earth Azimuth Angle"
            cbar_label = "Angle [nor,alized]"
        elif data == "earth_dist":
            plot_cube = self.earth_maps["dist"] / self.earth_vectors["dist"][:, None, None]
            title = "Earth Angular Size"
            cbar_label = "Angular Size [nor,alized]"
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

    def animate_maps(self):
        return
