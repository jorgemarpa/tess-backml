from typing import Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.visualization import simple_norm
from matplotlib import animation, axes, colors
from scipy.interpolate import RectBivariateSpline, interp1d
from tqdm import tqdm


def pooling_2d(
    input_array: np.ndarray,
    kernel_size: int = 4,
    stride: int = 4,
    padding: Union[int, Tuple[int, int]] = 0,
    stat: Callable = np.nanmedian,
) -> np.ndarray:
    """
    Performs 2D pooling on the input array with optional zero padding.

    Parameters
    ----------
    input_array : np.ndarray
        A 2D numpy array representing the input data.
    kernel_size : int, optional
        The size of the pooling kernel (square), by default 4. Must be positive.
    stride : int, optional
        The stride of the pooling operation, by default 4. Must be positive.
    padding : Union[int, Tuple[int, int]], optional
        Padding to be added to the input array before pooling.
        - If an int `p`: applies symmetric padding of `p` zeros.
          `p` rows of zeros are added to the top and `p` to the bottom.
          `p` columns of zeros are added to the left and `p` to the right.
        - If a tuple `(p_h, p_w)`: applies `p_h` rows of zeros to the top and `p_h` to the bottom,
          and `p_w` columns of zeros to the left and `p_w` to the right.
        Padding values must be non-negative. Default is 0 (no padding).
    stat : Callable, optional
        The aggregation function to use for pooling (e.g., np.mean, np.max, np.nanmedian),
        by default np.nanmedian.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the pooled output.
    """
    if input_array.ndim != 2:
        raise ValueError("Input array must be 2D.")

    if kernel_size <= 0:
        raise ValueError("Kernel size must be positive.")
    if stride <= 0:
        raise ValueError("Stride must be positive.")

    # Determine padding amounts for top/bottom and left/right
    pad_h_amount: int
    pad_w_amount: int

    if isinstance(padding, int):
        if padding < 0:
            raise ValueError("Padding value cannot be negative.")
        pad_h_amount = padding
        pad_w_amount = padding
    elif isinstance(padding, tuple):
        if len(padding) != 2:
            raise ValueError("Padding tuple must have two elements (pad_h, pad_w).")
        if not (isinstance(padding[0], int) and isinstance(padding[1], int)):
            raise ValueError("Padding tuple elements must be integers.")
        if padding[0] < 0 or padding[1] < 0:
            raise ValueError("Padding values in tuple cannot be negative.")
        pad_h_amount = padding[0]
        pad_w_amount = padding[1]
    else:
        raise ValueError("Padding must be an int or a tuple of two non-negative ints.")

    # Apply padding if necessary
    if pad_h_amount > 0 or pad_w_amount > 0:
        # np.pad expects ((pad_top, pad_bottom), (pad_left, pad_right))
        # Here, pad_h_amount is for top AND bottom, pad_w_amount for left AND right.
        padded_array = np.pad(
            input_array,
            ((pad_h_amount, pad_h_amount), (pad_w_amount, pad_w_amount)),
            mode="constant",
            constant_values=np.nan,
        )
    else:
        padded_array = input_array  # No padding needed or padding values were zero

    current_input_height, current_input_width = padded_array.shape

    # Calculate output dimensions
    # The formula for output dimension is O = floor((I - K) / S) + 1
    # where I is the input dimension *after* padding.
    # We use max(0, ...) to ensure dimensions are not negative, as negative
    # dimensions are invalid for array shapes.
    output_height_raw = (current_input_height - kernel_size) // stride + 1
    output_width_raw = (current_input_width - kernel_size) // stride + 1

    output_height = max(0, output_height_raw)
    output_width = max(0, output_width_raw)

    # If output_height or output_width is 0, shape_view will have a 0 dimension.
    # np.lib.stride_tricks.as_strided will create a view with this 0 dimension.
    # Subsequent application of `stat` (e.g., np.nanmedian) on such a view
    # typically results in an empty array with the correct remaining dimensions,
    # (e.g., shape (0, N) or (N, 0) or (0,0)), which is the desired behavior.

    shape_view = (output_height, output_width, kernel_size, kernel_size)
    strides_view = (
        padded_array.strides[0] * stride,
        padded_array.strides[1] * stride,
        padded_array.strides[0],
        padded_array.strides[1],
    )

    window_view = np.lib.stride_tricks.as_strided(
        padded_array, shape=shape_view, strides=strides_view
    )

    output_array = stat(window_view, axis=(2, 3))

    return output_array


def fill_nans_interp(cube: np.ndarray, deg: int = 3) -> np.ndarray:
    """
    Replace nan values in a data cube using plynomial interpolation

    Parameters
    ----------
    cube: np.ndarray
        Data cube with nan values to be interpolated
    deg: int, optional
        Degree of polynomial, defualt is 3.

    Returns
    -------
    np.ndarray
        Interpolated data cube without nan values
    """
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
    mask = ~np.isnan(cube[0]).ravel()

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
        print(
            f"Warning: Integer {integer} requires {len(raw_binary_string)} bits, "
            f"but only {num_bits} requested. Result might be misleading "
            f"if relying on implicit truncation by downstream slicing."
        )
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

    mask = np.array(
        [int_to_binary_array(int(x), 16)[-bit] for x in quality_array], dtype=bool
    )
    return mask


def interp_1d(x: np.ndarray, y: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """
    Performs 1D linear interpolation or extrapolation.

    This function uses `scipy.interpolate.interp1d` with 'slinear' kind
    for interpolation. It allows extrapolation by setting `fill_value`
    to "extrapolate".

    Parameters
    ----------
    x : np.ndarray
        A 1-D array of real values (the x-coordinates of the data points).
        Must be sorted in ascending order.
    y : np.ndarray
        A 1-D array of real values (the y-coordinates of the data points).
        Must have the same length as `x`.
    x_eval : np.ndarray
        A 1-D array of real values at which to evaluate the interpolated
        function.

    Returns
    -------
    np.ndarray
        An array of the same shape as `x_eval` containing the interpolated
        values.
    """
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(y, np.ndarray)
        or not isinstance(x_eval, np.ndarray)
    ):
        raise TypeError("Inputs x, y, and x_eval must be NumPy arrays.")
    if x.shape != y.shape:
        raise ValueError(
            f"Input arrays x and y must have the same shape, but got {x.shape} and {y.shape}."
        )

    fn = interp1d(x, y, kind="quadratic", bounds_error=False, fill_value="extrapolate")
    return fn(x_eval)


def interp_2d(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray
) -> np.ndarray:
    """
    Performs 1D linear interpolation or extrapolation.

    This function uses `scipy.interpolate.interp1d` with 'slinear' kind
    for interpolation. It allows extrapolation by setting `fill_value`
    to "extrapolate".

    Parameters
    ----------
    x : np.ndarray
        A 1-D array of real values (the x-coordinates of the data points).
        Must be sorted in ascending order.
    y : np.ndarray
        A 1-D array of real values (the y-coordinates of the data points).
        Must have the same length as `x`.
    x_eval : np.ndarray
        A 1-D array of real values at which to evaluate the interpolated
        function.

    Returns
    -------
    np.ndarray
        An array of the same shape as `x_eval` containing the interpolated
        values.
    """
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(y, np.ndarray)
        or not isinstance(z, np.ndarray)
    ):
        raise TypeError("Inputs x, y, and z must be NumPy arrays.")
    if not isinstance(x_eval, np.ndarray) or not isinstance(y_eval, np.ndarray):
        raise TypeError("Inputs x_eval and y_eval must be NumPy arrays.")

    if x.shape[0] != z.shape[0]:
        raise ValueError("Input arrays x and z must have the same shape in axis 1.")
    if y.shape[0] != z.shape[1]:
        raise ValueError("Input arrays y and z must have the same shape in axis 2.")

    if x.shape[0] <= 3:
        kx = 1
    else:
        kx = 3
    if y.shape[0] <= 3:
        ky = 1
    else:
        ky = 3
    fn = RectBivariateSpline(x, y, z, kx=kx, ky=ky)
    return fn(x_eval, y_eval)


def _resolve_remote_file(cadence: str = "FFI", sector: int = 1, camera: int = 1) -> str:
    """
    Resolves the remote file URL for TESS vectors based on cadence, sector, and camera.

    Parameters
    ----------
    cadence : str, optional
        The cadence type, one of ["FFI", "020", "120"], by default "FFI".
    sector : int, optional
        The TESS sector number, by default 1.
    camera : int, optional
        The TESS camera number, by default 1.

    Returns
    -------
    str
        The full URL to the TESS vectors CSV file.
    """
    remote_base = "https://heasarc.gsfc.nasa.gov/docs/tess/data/TESSVectors/Vectors/"
    cadence_folder = f"{cadence}_Cadence/"
    file_base = f"TessVectors_S{sector:03d}_C{camera}_{cadence}"
    file_ext = ".csv"
    fname = remote_base + cadence_folder + file_base + file_ext
    return fname


def get_tess_vectors(
    cadence: str = "FFI", sector: int = 1, camera: int = 1
) -> pd.DataFrame:
    """
    Fetches TESS vectors from the HEASARC database.

    Parameters
    ----------
    cadence : str, optional
        The cadence type, one of ["FFI", "020", "120"], by default "FFI".
    sector : int, optional
        The TESS sector number, by default 1.
    camera : int, optional
        The TESS camera number, by default 1.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the TESS vectors.
    """
    fname = _resolve_remote_file(cadence=cadence, sector=sector, camera=camera)
    vector = pd.read_csv(fname, comment="#", index_col=False)
    return vector
