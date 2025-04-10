from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import simple_norm
from matplotlib import animation, axes, colors
from scipy.interpolate import griddata
from tqdm import tqdm


def pooling_2d(
    input_array: np.ndarray,
    kernel_size: int = 4,
    stride: int = 4,
    stat: Callable = np.nanmedian,
) -> np.ndarray:
    """
    Performs 2D pooling on the input array.

    Parameters
    ----------
    input_array : np.ndarray
        A 2D numpy array representing the input data.
    kernel_size : int, optional
        The size of the pooling kernel (square), by default 4.
    stride : int, optional
        The stride of the pooling operation, by default 4.
    stat : Callable, optional
        The aggregation function to use for pooling (e.g., np.mean, np.max), 
        by default np.mean.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the pooled output.
    """
    if input_array.ndim != 2:
        raise ValueError("Input array must be 2D.")

    input_height, input_width = input_array.shape
    
    output_height = (input_height - kernel_size) // stride + 1
    output_width = (input_width - kernel_size) // stride + 1
    
    shape_view = (output_height, output_width, kernel_size, kernel_size)
    strides_view = (input_array.strides[0] * stride, input_array.strides[1] * stride, input_array.strides[0], 
                    input_array.strides[1])
    
    window_view = np.lib.stride_tricks.as_strided(input_array, shape=shape_view, strides=strides_view)
    
    output_array = stat(window_view, axis=(2, 3))
    return output_array

def fill_nans_interp(cube, deg=3):
    print("Linear modeling method")
    x, y = np.arange(cube.shape[2], dtype=float), np.arange(cube.shape[1], dtype=float)
    x = (x - np.median(x)) / np.std(x)
    y = (y - np.median(y)) / np.std(y)
    xx, yy = np.meshgrid(x, y)
    DM = np.vstack(
        [
            yy.ravel() ** idx * xx.ravel() ** jdx
            for idx in range(deg + 1)
            for jdx in range(deg + 1)
        ]
    ).T
    print(DM.shape)
    print(cube.shape)
    mask = ~np.isnan(cube[0]).ravel()
    print((~mask).sum())

    filled = []
    for idx in tqdm(range(len(cube))):
        array = cube[idx].ravel().copy()
        ws = np.linalg.solve(DM[mask].T.dot(DM[mask]), DM[mask].T.dot(array[mask]))
        array[~mask] = DM[~mask].dot(ws)
        filled.append(array.reshape(xx.shape))

    return np.array(filled)

def plot_img(
    img: np.ndarray,
    scol_2d: Optional[np.ndarray] = None,
    srow_2d: Optional[np.ndarray] = None,
    plot_type: str = "img",
    extent: Optional[Tuple] = None,
    cbar: bool = True,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cnorm: Optional[colors.Normalize] = None,
    bar_label: str = "Flux [e-/s]",
) -> axes.Axes:
    """
    Plots an image with optional scatter points and colorbar.

    Parameters
    ----------
    img : np.ndarray
        The 2D image data to be plotted.
    scol_2d : Optional[np.ndarray], optional
        The column coordinates of scatter points, by default None.
    srow_2d : Optional[np.ndarray], optional
        The row coordinates of scatter points, by default None.
    plot_type : str, optional
        The type of plot to create ('img' or 'scatter'), by default "img".
    extent : Optional[Tuple], optional
        The extent of the image (left, right, bottom, top), by default None.
    cbar : bool, optional
        Whether to display a colorbar, by default True.
    ax : Optional[plt.Axes], optional
        The matplotlib Axes object to plot on, by default None. If None, a new figure and axes are created.
    title : str, optional
        The title of the plot, by default "".
    vmin : Optional[float], optional
        The minimum value for the colormap, by default None.
    vmax : Optional[float], optional
        The maximum value for the colormap, by default None.
    cnorm : Optional[colors.Normalize], optional
        Custom color normalization, by default None.
    bar_label : str, optional
        The label for the colorbar, by default "Flux [e-/s]"."

    Returns
    -------
    matplotlib.axes
    """
    # Initialise ax
    if ax is None:
        _, ax = plt.subplots()

    # Define vmin and vmax
    if cnorm is None:
        vmin, vmax = np.nanpercentile(img.ravel(), [3, 97])

    if scol_2d is not None and srow_2d is not None:
        scol_2d = scol_2d.ravel()
        srow_2d = srow_2d.ravel()

    # Plot image, colorbar and marker
    if plot_type == "scatter":
        im = ax.scatter(
            scol_2d,
            srow_2d,
            c=img,
            marker="s",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            norm=cnorm,
            rasterized=True,
            s=10,
        )
    if plot_type == "img":
        im = ax.imshow(
            img,
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            norm=cnorm,
            extent=extent,
        )
    if cbar:
        plt.colorbar(im, location="right", shrink=0.8, label=bar_label)

    ax.set_aspect("equal", "box")
    ax.set_title(title)

    ax.set_xlabel("Pixel Column")
    ax.set_ylabel("Pixel Row")

    return ax


def animate_cube(
    cube: np.ndarray,
    scol_2d: Optional[np.ndarray] = None,
    srow_2d: Optional[np.ndarray] = None,
    cadenceno: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    plot_type: str = "img",
    extent: Optional[Tuple] = None,
    interval: int = 200,
    repeat_delay: int = 1000,
    step: int = 1,
    suptitle: str = "",
    bar_label: str = "Flux [e-/s]",
) -> animation.FuncAnimation:
    """
    Animates a 3D data cube (time series of 2D images).

    Parameters
    ----------
    cube : np.ndarray
        The 3D data cube to be animated (time, row, column).
    scol_2d : Optional[np.ndarray], optional
        The column coordinates of scatter points, by default None.
    srow_2d : Optional[np.ndarray], optional
        The row coordinates of scatter points, by default None.
    cadenceno : Optional[np.ndarray], optional
        Cadence numbers corresponding to the time axis, by default None.
    time : Optional[np.ndarray], optional
        Time values corresponding to the time axis, by default None.
    plot_type : str, optional
        The type of plot to create ('img' or 'scatter'), by default "img".
    extent : Optional[Tuple], optional
        The extent of the images (left, right, bottom, top), by default None.
    interval : int, optional
        Delay between frames in milliseconds, by default 200.
    repeat_delay : int, optional
        Delay in milliseconds before repeating the animation, by default 1000.
    step : int, optional
        Step size for frame selection, by default 1.
    suptitle : str, optional
        Overall title for the animation figure, by default "".
    bar_label : str, optional
        The label for the colorbar, by default "Flux [e-/s]".

    Returns
    -------
    animation.FuncAnimation
        The matplotlib FuncAnimation object.
    """
    # Initialise figure and set title
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.tight_layout()
    fig.suptitle(suptitle)

    norm = simple_norm(cube.ravel(), "linear", percent=98)

    # Plot first image in cube.
    nt = 0
    if cadenceno is not None and time is not None:
        title = f"CAD {cadenceno[nt]} | BTJD {time[nt]:.4f}"
    else:
        title = f"Time index {nt}"
    ax = plot_img(
        cube[nt],
        scol_2d=scol_2d,
        srow_2d=srow_2d,
        plot_type=plot_type,
        extent=extent,
        cbar=True,
        ax=ax,
        title=title,
        cnorm=norm,
        bar_label=bar_label,
    )

    # Define function for animation
    def animate(nt):
        if cadenceno is not None and time is not None:
            title = f"CAD {cadenceno[nt]} | BTJD {time[nt]:.4f}"
        else:
            title = f"Time index {nt}"
        ax.clear()
        _ = plot_img(
            cube[nt],
            scol_2d=scol_2d,
            srow_2d=srow_2d,
            plot_type=plot_type,
            extent=extent,
            cbar=False,
            ax=ax,
            title=title,
            cnorm=norm,
            bar_label=bar_label,
        )

        return ()

    # Prevent second figure from showing up in interactive mode
    plt.close(ax.figure)  # type: ignore

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=range(0, len(cube), step),
        interval=interval,
        blit=True,
        repeat_delay=repeat_delay,
        repeat=True,
    )

    return ani

def int_to_binary_array(integer: int, num_bits: int) -> np.ndarray:
    """
    Converts a non-negative integer to its binary representation as a NumPy array.

    Parameters
    ----------
    integer : int
        The non-negative integer to convert.
    num_bits : int
        The desired number of bits in the output binary representation.
        The binary string will be left-padded with zeros if necessary.
        Must be greater than zero.

    Returns
    -------
    np.ndarray
        A NumPy array of dtype uint8 representing the binary digits (0 or 1),
        with the most significant bit first. The length of the array is `num_bits`.
    """
    if not isinstance(integer, int):
        raise TypeError("Input must be an integer.")
    if integer < 0:
        raise ValueError("Input must be a non-negative integer.")
    if num_bits <= 0:
        raise ValueError("Number of bits must be greater than zero.")

    binary_string = bin(integer)[2:].zfill(num_bits)

    raw_binary_string = bin(integer)[2:]
    if len(raw_binary_string) > num_bits:
         print(f"Warning: Integer {integer} requires {len(raw_binary_string)} bits, "
               f"but only {num_bits} requested. Result might be misleading "
               f"if relying on implicit truncation by downstream slicing.")
    binary_string = raw_binary_string.zfill(num_bits)


    binary_array = np.array([int(bit) for bit in binary_string], dtype=np.uint8)
    return binary_array

def has_bit(quality_array: np.ndarray, bit: int) -> np.ndarray:
    """
    Checks if a specific bit is set in each element of a quality flag array.

    Parameters
    ----------
    quality_array : np.ndarray
        A NumPy array of integers (quality flags).
    bit : int
        The bit position to check (1-based index, where 1 is the LSB).
        Must be between 1 and 16 (inclusive).

    Returns
    -------
    np.ndarray
        A boolean NumPy array of the same shape as `quality_array`, where
        True indicates that the specified `bit` is set (1) in the
        corresponding quality flag integer.
    """
    if not isinstance(bit, int):
         raise TypeError("`bit` must be an integer.")
    if not 1 <= bit <= 16:
        raise ValueError("`bit` must be between 1 and 16 (inclusive).")

    mask = np.array([int_to_binary_array(int(x), 16)[-bit] for x in quality_array], dtype=bool)
    return mask