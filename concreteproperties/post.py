from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Optional, Tuple, Union

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import matplotlib


@contextlib.contextmanager
def plotting_context(
    ax: Optional[matplotlib.axes.Axes] = None,  # type: ignore
    pause: bool = True,
    title: str = "",
    aspect: bool = False,
    filename: str = "",
    render: bool = True,
    axis_index: Optional[Union[int, Tuple[int]]] = None,
    **kwargs,
):
    """Executes code required to set up a matplotlib figure.

    :param ax: Axes object on which to plot
    :param pause: If set to true, the figure pauses the script until the window is
        closed. If set to false, the script continues immediately after the window is
        rendered.
    :param title: Plot title
    :param aspect: If set to True, the axes of the figure are set to an equal aspect
        ratio
    :param filename: Pass a non-empty string or path to save the image as. If
        this option is used, the figure is closed after the file is saved.
    :param render: If set to False, the image is not displayed. This may be useful
        if the figure or axes will be embedded or further edited before being
        displayed.
    :param axis_index: If more than 1 axes is created by subplot, then this is the axis
        to plot on. This may be a tuple if a 2D array of plots is returned.  The
        default value of None will select the top left plot.
    :param kwargs: Passed to :func:`matplotlib.pyplot.subplots`
    """

    if filename:
        render = False

    if ax is None:
        if not render:
            plt.ioff()
        elif pause:
            plt.ioff()
        else:
            plt.ion()

        ax_supplied = False
        fig, ax = plt.subplots(**kwargs)

        try:
            if axis_index is None:
                axis_index = (0,) * ax.ndim  # type: ignore
            ax = ax[axis_index]  # type: ignore
        except (AttributeError, TypeError):
            pass  # only 1 axis, not an array
        except IndexError as exc:
            raise ValueError(
                f"axis_index={axis_index} is not compatible with arguments to subplots: {kwargs}"
            ) from exc
    else:
        fig = ax.get_figure()
        ax_supplied = True
        if not render:
            plt.ioff()

    yield fig, ax

    ax.set_title(title)  # type: ignore

    if ax_supplied:
        # if an axis was supplied, don't continue with displaying or configuring the plot
        return

    # if no axes was supplied, finish the plot and return the figure and axes
    plt.tight_layout()

    if aspect:
        ax.set_aspect("equal", anchor="C")  # type: ignore

    if filename:
        fig.savefig(filename, dpi=fig.dpi)
        plt.close(fig)  # close the figure to free the memory
        return  # if the figure was to be saved, then don't show it also

    if render:
        if pause:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.001)  # type: ignore
