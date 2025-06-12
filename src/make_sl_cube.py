import argparse
import os

import matplotlib
from tess_backml import BackgroundCube, log

matplotlib.rcParams['animation.embed_limit'] = 2**128


def build_dataset(
    sector: int = 1,
    camera: int = 1,
    ccd: int = 1,
    img_bin: int = 8,
    time_bin: float = 2.0,
    plot: bool = False,
    out_dir: str= "./",
):
    cube = BackgroundCube(
        sector=sector, camera=camera, ccd=ccd, img_bin=img_bin, downsize="binning"
    )
    log.info(cube)
    cube.get_scatter_light_cube(
        frames=None, mask_straps=True, plot=False, rolling=True, errors=True
    )
    
    if plot:
        fig_dir = f"{out_dir}/figures/sector{sector:03}"
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        fig_file = (
            f"{fig_dir}/ffi_slcube_sector{cube.sector:03}_{cube.camera}-{cube.ccd}"
            f"_bin{cube.img_bin}_{int(time_bin)}h.gif"
            )
        log.info("Time binning...")
        cube.bin_time_axis(bin_size=time_bin)
        log.info(f"Saving animation to {fig_file}")
        cube.animate_data(data="sl_bin", file_name=fig_file, save=True, step=4)

    data_dir = f"{out_dir}/cubes/sector{sector:03}"
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    out_file = f"{data_dir}/ffi_slcube_sector{cube.sector:03}_{cube.camera}-{cube.ccd}.fits"

    log.info(f"Saving cubes to {out_file}")
    hdul = cube.save_to_fits(out_file=None, binned=True)
    hdul[0].header["FILEVER"] = ("1.0", "File version")
    hdul.writeto(out_file, overwrite=True)
    log.info("Done!")
    return


if __name__ == "__main__":
    # program flags
    parser = argparse.ArgumentParser(
        description="Build TESS FFI background dataset for a Sector/Camera/CCD."
    )
    parser.add_argument(
        "--sector",
        dest="sector",
        type=int,
        default="1",
        help="TESS sector.",
    )
    parser.add_argument(
        "--camera",
        dest="camera",
        type=int,
        default=1,
        help="TESS camera.",
    )
    parser.add_argument(
        "--ccd",
        dest="ccd",
        type=int,
        default=1,
        help="TESS camera.",
    )
    parser.add_argument(
        "--image-bin",
        dest="img_bin",
        type=int,
        default=16,
        help="Image binning size, must divide 2048.",
    )
    parser.add_argument(
        "--time-bin",
        dest="time_bin",
        type=float,
        default=2,
        help="Time binning size, in hours.",
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        default=False,
        help="Plot target light curve.",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        type=str,
        default="./",
        help="Outputh directory path where files and figures will be saved.",
    )

    args = parser.parse_args()

    log.info(args)

    build_dataset(
        sector=args.sector,
        camera=args.camera,
        ccd=args.ccd,
        img_bin=args.img_bin,
        time_bin=args.time_bin,
        plot=args.plot,
        out_dir=args.out_dir,
    )
