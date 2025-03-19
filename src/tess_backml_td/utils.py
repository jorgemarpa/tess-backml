import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import Callable, Optional, tuple
from astropy.visualization import simple_norm


def pooling_2d(
    input_array: np.ndarray,
    kernel_size: int = 4,
    stride: int = 4,
    stat: Callable = np.mean,
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
    output_array = np.zeros((output_height, output_width))
    
    for i in range(output_height):
        for j in range(output_width):
            start_i = i * stride
            start_j = j * stride
            end_i = start_i + kernel_size
            end_j = start_j + kernel_size
            output_array[i, j] = stat(input_array[start_i:end_i, start_j:end_j])
            
    return output_array

def plot_img(
    img: np.ndarray,
    scol_2d: Optional[np.ndarray] = None,
    srow_2d: Optional[np.ndarray] = None,
    plot_type: str = "img",
    extent: Optional[tuple] = None,
    cbar: bool = True,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cnorm: Optional = None,
):
    # Initialise ax
    if ax is None:
        _, ax = plt.subplots()

    # Define vmin and vmax
    if cnorm is None:
        vmin, vmax = np.nanpercentile(img.ravel(), [3, 97])

    # Plot image, colorbar and marker
    if plot_type == "scatter":
        im = ax.scatter(
            scol_2d.ravel(),
            srow_2d.ravel(),
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
        plt.colorbar(im, location="right", shrink=0.8, label="Flux [e-/s]")

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
    extent: Optional[tuple] = None,
    interval: int = 200,
    repeat_delay: int = 1000,
    step: int = 1,
    suptitle: str = "",
):
    # Initialise figure and set title
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.set_tight_layout(True)
    fig.suptitle(suptitle)

    norm = simple_norm(cube.ravel(), "linear", percent=98)

    # Plot first image in cube.
    nt = 0
    ax = plot_img(
        cube[nt],
        scol_2d=scol_2d,
        srow_2d=srow_2d,
        plot_type=plot_type,
        extent=extent,
        cbar=True,
        ax=ax,
        title=f"CAD {cadenceno[nt]} | BTJD {time[nt]:.4f}",
        cnorm=norm,
    )

    # Define function for animation
    def animate(nt):
        ax.clear()
        _ = plot_img(
            cube[nt],
            scol_2d=scol_2d,
            srow_2d=srow_2d,
            plot_type=plot_type,
            extent=extent,
            cbar=False,
            ax=ax,
            title=f"CAD {cadenceno[nt]} | BTJD {time[nt]:.4f}",
            cnorm=norm,
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