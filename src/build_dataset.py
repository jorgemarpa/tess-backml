import argparse
import os

import matplotlib
from tess_backml import BackgroundCube, log

matplotlib.rcParams['animation.embed_limit'] = 2**128


def build_dataset(
    sector: int = 1,
    camera: int = 1,
    ccd: int = 1,
    img_bin: int = 16,
    # time_bin: int = 1,
    downsize: str = "binning",
    plot: bool = False,
    out_dir: str= "./",
):
    bkg_data = BackgroundCube(
        sector=sector, camera=camera, ccd=ccd, img_bin=img_bin, downsize=downsize
    )
    log.info(bkg_data)
    bkg_data.get_scatter_light_cube(frames=None, mask_straps=True, plot=False)
    
    if plot:
        fig_dir = f"{out_dir}/figures/sector{sector:03}"
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        fig_file = (
            f"{fig_dir}/ffi_scatterlight_bin{bkg_data.img_bin}"
            f"_sector{bkg_data.sector:03}_{bkg_data.camera}-{bkg_data.ccd}.gif"
            )
        log.info(f"Saving animation to {fig_file}")
        bkg_data.animate_data(data="sl", file_name=fig_file, save=True, step=10)

    bkg_data.get_vector_maps(ang_size=True)

    data_dir = f"{out_dir}/cubes/sector{sector:03}"
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    out_file = f"{data_dir}/ffi_cubes_bin{bkg_data.img_bin}_sector{bkg_data.sector:03}_{bkg_data.camera}-{bkg_data.ccd}.npz"

    log.info(f"Saving cubes to {out_file}")
    bkg_data.save_data(out_file=out_file, save_maps=True)
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
        "--downsize",
        dest="downsize",
        type=str,
        default="binning",
        help="Method for downsizing the image, one of [sparse, binning].",
    )
    parser.add_argument(
        "--image-bin",
        dest="img_bin",
        type=int,
        default=16,
        help="Image binning size, must divide 2048.",
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
        downsize=args.downsize,
        plot=args.plot,
        out_dir=args.out_dir,
    )
