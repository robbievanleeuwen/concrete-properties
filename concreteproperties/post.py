from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import contextlib
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import matplotlib
    from concreteproperties.concrete_section import ConcreteSection
    from concreteproperties.analysis_section import AnalysisSection


@contextlib.contextmanager
def plotting_context(
    ax: Optional[matplotlib.axes.Axes] = None,
    pause: Optional[bool] = True,
    title: Optional[str] = "",
    filename: Optional[str] = "",
    render: Optional[bool] = True,
    axis_index: Optional[Union[None, int, Tuple[int]]] = None,
    **kwargs,
):
    """Executes code required to set up a matplotlib figure.

    :param ax: Axes object on which to plot
    :type ax: Optional[matplotlib.axes.Axes]
    :param pause: If set to true, the figure pauses the script until the window is
        closed. If set to false, the script continues immediately after the window is
        rendered.
    :type pause: Optional[bool]
    :param title: Plot title
    :type title: Optional[str]
    :param filename: Pass a non-empty string or path to save the image as. If
        this option is used, the figure is closed after the file is saved.
    :type filename: Optional[str]
    :param render: If set to False, the image is not displayed. This may be useful
        if the figure or axes will be embedded or further edited before being
        displayed.
    :type render: Optional[bool]
    :param axis_index: If more than 1 axes is created by subplot, then this is the axis
        to plot on. This may be a tuple if a 2D array of plots is returned.  The
        default value of None will select the top left plot.
    :type axis_index: Optional[Union[None, int, Tuple[int]]
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
                axis_index = (0,) * ax.ndim
            ax = ax[axis_index]
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

    ax.set_title(title)

    if ax_supplied:
        # if an axis was supplied, don't continue with displaying or configuring the plot
        return

    # if no axes was supplied, finish the plot and return the figure and axes
    plt.tight_layout()

    if filename:
        fig.savefig(filename, dpi=fig.dpi)
        plt.close(fig)  # close the figure to free the memory
        return  # if the figure was to be saved, then don't show it also

    if render:
        if pause:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.001)
