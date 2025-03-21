import argparse
import os

from tess_backml import PACKAGEDIR, Background_Data

# import sys
# from typing import Optional, Union

# import numpy as np
# import pandas as pd


def build_dataset(
    sector: int = 1,
    camera: int = 1,
    ccd: int = 1,
    img_bin: int = 16,
    # time_bin: int = 1,
    downsize: str = "binning",
    plot: bool = False,
):
    bkg_data = Background_Data(
        sector=sector, camera=camera, ccd=ccd, img_bin=img_bin, downsize=downsize
    )
    print(bkg_data)
    bkg_data.get_flux_data()
    tess_bkg.get_scatter_light_cube()
    tess_bkg.get_vector_maps()

    fig_dir = f"{PACKAGEDIR}/data/figures/sector{sector:03}"
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    fig_file = f"{fig_dir}//ffi_flux_bin{bkg_data.img_bin}_sector{bkg_data.sector:03}_{bkg_data.camera}-{bkg_data.ccd}.gif"
    tess_bkg.animate_flux(file_name=fig_file)

    data_dir = f"{PACKAGEDIR}/data/bkg_data/sector{sector:03}"
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    out_file = f"{data_dir}/ffi_cube_bin{bkg_data.img_bin}_sector{bkg_data.sector:03}_{bkg_data.camera}-{bkg_data.ccd}.npz"
    tess_bkg.save_data(out_file=out_file)

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

    args = parser.parse_args()

    print(args)

    build_dataset(
        sector=args.sector,
        camera=args.camera,
        ccd=args.ccd,
        img_bin=args.img_bin,
        downsize=args.downsize,
        plot=args.plot,
    )
