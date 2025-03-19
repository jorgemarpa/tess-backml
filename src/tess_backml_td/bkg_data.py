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
from .utils import pooling_2d

from typing import Optional, Tuple, Union

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
        sector: int = 1,
        camera: int = 1,
        ccd: int = 1,
        img_bin: int = 16,
        time_bin: int = 1,
        downsize: str = "binning"
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
        self.cadenceno_bin = self.tcube.cadence_number 
        self.nt = len(self.time)


    def _get_dark_frame_idx(self, low_per: float=3.0):
        """
        Finds the indices of darkest frames.
        """
        # get sparse pixels time series to find dark frames
        srow = np.arange(self.rmin + self.img_bin/2, self.rmax - self.img_bin/2, self.img_bin, dtype=int)
        scol = np.arange(self.cmin + self.img_bin/2, self.cmax - self.img_bin/2, self.img_bin, dtype=int)
        srow_2d, scol_2d = np.meshgrid(srow, scol, indexing="ij")
        sparse_rc = [(r, c) for r, c in zip(srow_2d.ravel(), scol_2d.ravel())][::2]
        pix_ts = self.tcube.get_pixel_timeseries(sparse_rc, return_errors=False)
        
        # reshape as [ntimes, npix]
        pix_ts = pix_ts.reshape((tcube.nframes, -1))
        # take median
        bkg_lc = np.nanmedian(pix_ts, axis=-1)
        # find darkes and < 3% frame indices
        self.darkest_frame = np.argmin(bkg_lc)
        self.dark_frames = np.where(bkg_lc <= np.percentile(bkg_lc, low_per))[0]
        return
    
    def _get_star_mask(self, sigma_clip: float=5.0, dilat_iter: int=2):
        """
        Computes a star mask using sigma clipping on the image and on the gradients 
        of the image.
        Then uses dilation to enlarge the mask `dilat_iter` times.
        """
        self.ffi_dark = self.tcube.get_ffi(self.darkest_frame)[1].data[self.rmin:self.rmax, self.cmin:self.cmax]
        grad = np.hypot(*np.gradient(self.ffi_dark))
        star_mask = sigma_clip(ffi_dark, sigma=3).mask & sigma_clip(grad, sigma=3).mask
        self.star_mask = ndimage.binary_dilation(star_mask, iterations=dilat_iter)

        return
    
    def plot_dark_frame(self):
        """
        Creates a diagnostic plot of the darkes frame and the star mask
        """
        vmin, vmax = np.percentile(self.ffi_dark.ravel(), [3,97])

        fig, ax = plt.subplots(1,2,figsize=(9,4))
        ax[0].set_title("Darkest frame FFI")
        bar = ax[0].imshow(self.ffi_dark, origin="lower", vmin=vmin, vmax=vmax)
        plt.colorbar(bar, ax=ax[0], shrink=0.8, label="Flux [-e/s]")
        
        ax[1].set_title(r"Star mask 5$\sigma$")
        bar = ax[1].imshow(self.star_mask, origin="lower", vmin=0, vmax=1)
        plt.colorbar(bar, ax=ax[1], shrink=0.8)

        return

    
    def get_flux_data(slef, plot: bool=False):
        """
        Gets flux cube from MAST/AWS and does downsampling
        """
        # get dark frame
        self._get_dark_frame_idx()
        # get star mask
        self._get_star_mask(sigma_clip=5.0, dilat_iter=2)
        if plot:
            self.plot_dark_frame()

        srow = np.arange(self.rmin + self.img_bin/2, self.rmax - self.img_bin/2, self.img_bin, dtype=int)
        scol = np.arange(self.cmin + self.img_bin/2, self.cmax - self.img_bin/2, self.img_bin, dtype=int)
        
        self.row_2d, self.col_2d = np.meshgrid(srow, scol, indexing="ij")
        # get flux cube with downsampling
        if self.downsize == "sparse":
            # sparse pixels across the CCD
            sparse_rc = [(r, c) for r, c in zip(self.row_2d.ravel(), self.col_2d.ravel())]
            flux_cube = self.tcube.get_pixel_timeseries(sparse_rc, return_errors=False)
            flux_cube = flux_cube.reshape((self.tcube.nframes, *self.row_2d.shape))

        elif self.downsize == "binning":
            # all pixels the binning with seld.
            flux_cube = []
            for f in tqdm(range(self.tcube.nframes)):
                current = tcube.get_ffi(f)[1].data[rmin:rmax, cmin:cmax]
                current[star_mask5] = np.nan

                flux_cube.append(pooling_2d(current, kernel_size=self.img_bin, stride=self.img_bin, stat=np.nanmedian))
            flux_cube = np.array(flux_cube)

        else:
            print("Wrong pixel grid option...")

        self.flux_cube = flux_cube
        return
    
    def get_scatter_light_cube(self):
        """
        Removes the mean static scene to the flux cube to obtain the tine changing 
        scatter light signal only.
        """
        static = np.median(self.flux_cube[self.dark_frames], axis=0)
        self.scatter_cube = self.flux_cube - static
        return
    
    def _get_object_vectors(self, object: str="Earth"):
        """
        Auxiliar function to get vector maps for an object
        """
        if object not in ["Earth", "Moon"]:
            raise ValueError("Object must be one of ['Earth', 'Moon']")

        # make a low res pixel grid with physical units
        grid_row, grid_col = np.mgrid[self.rmin:self.rmax:self.img_bin, self.cmin:self.cmax:self.img_bin]
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
            aux_elev = np.sin(np.deg2rad(self.vectors[f"{object}_Camera_Angle"][t] - 90)) + grid_row_d.to("m") / dist.to("m")
            aux_elev = np.rad2deg(np.arcsin(aux_elev)).value + 90
            object_alt_map.append(aux_elev)
            
            cos_ip = np.cos(np.deg2rad(aux_elev - 90))
            cos_i =  np.cos(np.deg2rad(self.vectors[f"{object}_Camera_Angle"][t] - 90))
            sin_az =  np.cos(np.deg2rad(self.vectors[f"{object}_Camera_Azimuth"][t] - 180))
            aux_az = (cos_i * sin_az) / (cos_ip) + (grid_col_d.to("m")) / (dist.to("m") * cos_ip)
            aux_az = np.rad2deg(np.arcsin(aux_az)).value + 180
            object_az_map.append(aux_az)

            ang_dist = (np.sqrt(grid_row ** 2 + grid_col ** 2) * u.pixel * tess_pscale)
            pix_dist = (np.sqrt(grid_row ** 2 + grid_col ** 2) * 15 * u.micrometer).to("m")
            aux_dist = dist.to("m") * np.cos(ang_dist.to("rad"))

            aux_dist += np.sqrt(pix_dist ** 2 + (dist.to("m") * np.sin(ang_dist.to("rad")))**2) * np.sign(grid_row_d)
            object_dist_map.append(aux_dist)

        # we need to flip the value maps to account for cam/CCD orientations
        if camera in [1, 2]:
            if ccd == 1:
                object_alt_map = np.flip(np.array(object_alt_map), axis=1)
                object_alt_map = np.flip(np.array(object_alt_map), axis=2)
                object_az_map = np.flip(np.array(object_az_map), axis=1)
                object_az_map = np.flip(np.array(object_az_map), axis=2)
                object_dist_map = np.flip(np.array(object_dist_map), axis=1)
                object_dist_map = np.flip(np.array(object_dist_map), axis=2)
            if ccd == 2:
                object_alt_map = np.flip(np.array(object_alt_map), axis=1)
                object_az_map = np.flip(np.array(object_az_map), axis=1)
                object_dist_map = np.flip(np.array(object_dist_map), axis=1)
            if ccd == 3:
                object_alt_map = np.flip(np.array(object_alt_map), axis=1)
                object_alt_map = np.flip(np.array(object_alt_map), axis=2)
                object_az_map = np.flip(np.array(object_az_map), axis=1)
                object_az_map = np.flip(np.array(object_az_map), axis=2)
                object_dist_map = np.flip(np.array(object_dist_map), axis=1)
                object_dist_map = np.flip(np.array(object_dist_map), axis=2)
            if ccd == 4:
                object_alt_map = np.flip(np.array(object_alt_map), axis=1)
                object_az_map = np.flip(np.array(object_az_map), axis=1)
                object_dist_map = np.flip(np.array(object_dist_map), axis=1)
        if camera in [3, 4]:
            if ccd == 3:
                object_alt_map = np.flip(np.array(object_alt_map), axis=1)
                object_alt_map = np.flip(np.array(object_alt_map), axis=2)
                object_az_map = np.flip(np.array(object_az_map), axis=1)
                object_az_map = np.flip(np.array(object_az_map), axis=2)
                object_dist_map = np.flip(np.array(object_dist_map), axis=1)
                object_dist_map = np.flip(np.array(object_dist_map), axis=2)
            if ccd == 4:
                object_alt_map = np.flip(np.array(object_alt_map), axis=1)
                object_az_map = np.flip(np.array(object_az_map), axis=1)
                object_dist_map = np.flip(np.array(object_dist_map), axis=1)
            if ccd == 1:
                object_alt_map = np.flip(np.array(object_alt_map), axis=1)
                object_alt_map = np.flip(np.array(object_alt_map), axis=2)
                object_az_map = np.flip(np.array(object_az_map), axis=1)
                object_az_map = np.flip(np.array(object_az_map), axis=2)
                object_dist_map = np.flip(np.array(object_dist_map), axis=1)
                object_dist_map = np.flip(np.array(object_dist_map), axis=2)
            if ccd == 2:
                object_alt_map = np.flip(np.array(object_alt_map), axis=1)
                object_az_map = np.flip(np.array(object_az_map), axis=1)
                object_dist_map = np.flip(np.array(object_dist_map), axis=1)

        
        
        return {"dist": np.array(object_dist_map), "alt":np.array(object_alt_map), "az":np.array(object_az_map)}
            
    
    def get_vector_maps(self):
        """
        Gets the tess space raft vectors using `tessvetors` which have the Earth/Moon 
        angles and distances  with respect to each camera. Then computes maps with 
        values for the angles and distance for each pixel in the downsize cube.
        """
        self.vectors = tessvectors.getvector(("FFI", sector, camera))

        self.earth_maps = self._get_object_vectors(object="Earth")
        self.earth_vectors = {
            "dist": (self.vectors["Earth_Distance"] * const.R_earth).to("km").value, 
            "alt": self.vectors["Earth_Camera_Angle"], 
            "az":self.vectors["Earth_Camera_Azimuth"]
            }
        self.moon_maps = self._get_object_vectors(object="Moon")
        self.moon_vectors = {
            "dist": (self.vectors["Earth_Distance"] * const.R_earth).to("km").value, 
            "alt": self.vectors["Earth_Camera_Angle"], 
            "az":self.vectors["Earth_Camera_Azimuth"]
            }

        return
    
    def time_bin(self):
        """
        Inplace binning of the cube in the time axis.
        """
        # bin in time if asked
        if self.time_bin > 1:
            indices = np.array_split(np.arange(self.nt), int(self.nt/self.time_bin))

            self.flux_cube = np.array([np.nanmedian(self.flux_cube[x], axis=0) for x in indices])
            self.time_bin = np.array([np.mean(self.times[x], axis=0) for x in indices])
            self.cadenceno_bin = np.array([np.mean(self.cadenceno[x], axis=0) for x in indices])

            for key in ["dist", "alt", "az"]:
                self.earth_vectors[key] = np.array([np.mean(self.earth_vectors[key][x], axis=0) for x in indices])
                self.earth_maps[key] = np.array([np.mean(self.earth_maps[key][x], axis=0) for x in indices])
                self.moon_vectors[key] = np.array([np.mean(self.moon_vectors[key][x], axis=0) for x in indices])
                self.moon_maps[key] = np.array([np.mean(self.moon_maps[key][x], axis=0) for x in indices])

        return
    
    def save_data(self, out_file: Optional[str]=None):
        """
        Save data to disk
        """
        if out_file is None:
            out_file = (f"../../data/ffi_cube_bin{self.img_bin}_sector{self.sector:03}_{self.camera}-{self.ccd}.npz")
        np.savez(
            out_file,
            scatter_cube=self.scatter_cube,
            time=self.sime,
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
    
    def animate_flux(self, step: int=10):
        """
        Makes an animation of the scatter light cube
        """

        # Create animation
        ani = animate_cube(
            self.scatter_cube,
            cadenceno=self.cadence_number,
            time=self.time,
            plot_type="img",
            extent=(self.cmin-0.5, self.cmax-0.5, self.rmin-0.5, self.rmax-0.5),
            step=step,
            suptitle=f"Scatter Light Sector {self.sector} Camera {self.camera} CCD {self.ccd}",
        )

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
    





