"""Post-processor methods."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt


if TYPE_CHECKING:
    import matplotlib.axes


@contextlib.contextmanager
def plotting_context(
    ax: matplotlib.axes.Axes | None = None,
    pause: bool = True,
    title: str = "",
    aspect: bool = False,
    filename: str = "",
    render: bool = True,
    axis_index: int | tuple[int, int] | None = None,
    **kwargs,
):
    """Executes code required to set up a matplotlib figure.

    Args:
        ax: Axes object on which to plot
        pause: If set to true, the figure pauses the script until the window is closed.
            If set to false, the script continues immediately after the window is
            rendered.
        title: Plot title
        aspect: If set to True, the axes of the figure are set to an equal aspect ratio
        filename: Pass a non-empty string or path to save the image as. If this option
            is used, the figure is closed after the file is saved.
        render: If set to False, the image is not displayed. This may be useful if the
            figure or axes will be embedded or further edited before being displayed.
        axis_index: If more than 1 axes is created by subplot, then this is the axis to
            plot on. This may be a tuple if a 2D array of plots is returned. The default
            value of None will select the top left plot.
        kwargs: Passed to :func:`matplotlib.pyplot.subplots`

    Raises:
        ValueError: ``axis_index`` is invalid

    Yields:
        Matplotlib figure and axes
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
                axis_index = (0,) * ax.ndim
            ax = ax[axis_index]
        except (AttributeError, TypeError):
            pass  # only 1 axis, not an array
        except IndexError as exc:
            msg = f"axis_index={axis_index} is not compatible "
            msg += f"with arguments to subplots: {kwargs}"
            raise ValueError(msg) from exc
    else:
        fig = ax.get_figure()
        ax_supplied = True
        if not render:
            plt.ioff()

    yield fig, ax

    if ax:
        ax.set_title(title)

    if ax_supplied:
        # if an axis was supplied, don't continue displaying or configuring the plot
        return

    # if no axes was supplied, finish the plot and return the figure and axes
    plt.tight_layout()

    if aspect and ax:
        ax.set_aspect("equal", anchor="C")

    if filename and fig:
        fig.savefig(filename, dpi=fig.dpi)
        plt.close(fig)  # close the figure to free the memory
        return  # if the figure was to be saved, then don't show it also

    if render:
        if pause:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.001)
