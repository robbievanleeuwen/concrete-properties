from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import contextlib
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.tri as tri
import matplotlib.cm as cm
from matplotlib.colors import CenteredNorm
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from rich.pretty import pprint

if TYPE_CHECKING:
    import matplotlib
    from concreteproperties.concrete_section import ConcreteSection
    from concreteproperties.analysis_section import AnalysisSection


@contextlib.contextmanager
def plotting_context(
    ax: Optional[matplotlib.axes._subplots.AxesSubplot] = None,
    pause: Optional[bool] = True,
    title: Optional[str] = "",
    filename: Optional[str] = "",
    render: Optional[bool] = True,
    axis_index: Optional[Union[None, int, Tuple[int]]] = None,
    **kwargs,
):
    """Executes code required to set up a matplotlib figure.

    :param ax: Axes object on which to plot
    :type ax: Optional[matplotlib.axes._subplots.AxesSubplot]
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


def plot_stress(
    concrete_section: ConcreteSection,
    analysis_sections: List[AnalysisSection],
    conc_sigs: List[np.ndarray],
    steel_sigs: List[float],
    title: str,
    conc_cmap: str,
    steel_cmap: str,
    **kwargs,
) -> matplotlib.axes._subplots.AxesSubplot:
    """Plots concrete and steel stresses on a concrete section.

    :param concrete_section: Concrete section object
    :type concrete_section:
        :class:`~concreteproperties.concrete_section.ConcreteSection`
    :param analysis_sections: List of analysis section objects
    :type analysis_section:
        List[:class:`~concreteproperties.analysis_section.AnalysisSection`]
    :param conc_sigs: List of concrete stresses corresponding to the list of analysis
        sections
    :type conc_sigs: List[:class:`numpy.ndarray`]
    :param steel_sigs: List of steel stresses
    :type steel_sigs: List[float]
    :param str title: Plot title
    :param str conc_cmap: Colour map for the concrete stress
    :param str steel_cmap: Colour map for the steel stress
    :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

    :return: Matplotlib axes object
    :rtype: :class:`matplotlib.axes._subplots.AxesSubplot`
    """

    with plotting_context(
        title=title,
        **dict(kwargs, nrows=1, ncols=3, gridspec_kw={"width_ratios": [1, 0.08, 0.08]}),
    ) as (fig, ax):
        # plot background
        concrete_section.plot_section(background=True, **dict(kwargs, ax=fig.axes[0]))

        # set up the colormaps
        cmap_conc = cm.get_cmap(name=conc_cmap)
        cmap_steel = cm.get_cmap(name=steel_cmap)

        # determine minimum and maximum stress values for the contour list
        conc_sig_min = min([min(x) for x in conc_sigs])
        conc_sig_max = max([max(x) for x in conc_sigs])
        steel_sig_min = min(steel_sigs)
        steel_sig_max = max(steel_sigs)

        # set up ticks
        v_conc = np.linspace(conc_sig_min, conc_sig_max, 15, endpoint=True)
        v_steel = np.linspace(steel_sig_min, steel_sig_max, 15, endpoint=True)

        if np.isclose(v_conc[0], v_conc[-1], atol=1e-12):
            v_conc = 15
            ticks_conc = None
        else:
            ticks_conc = v_conc

        if np.isclose(v_steel[0], v_steel[-1], atol=1e-12):
            v_steel = 15
            ticks_steel = None
        else:
            ticks_steel = v_steel

        # plot the concrete stresses
        for idx, sig in enumerate(conc_sigs):
            # create triangulation
            triang = tri.Triangulation(
                analysis_sections[idx].mesh_nodes[:, 0],
                analysis_sections[idx].mesh_nodes[:, 1],
                analysis_sections[idx].mesh_elements[:, 0:3],
            )

            # plot the filled contour
            trictr = fig.axes[0].tricontourf(
                triang, sig, v_conc, cmap=cmap_conc, norm=CenteredNorm()
            )

            # plot a zero stress contour, supressing warning
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="No contour levels were found within the data range.",
                )

                # set zero stress for neutral axis contour
                zero_level = 0

                if min(sig) > 0:
                    if min(sig) < 1e-3:
                        zero_level = min(sig) + 1e-12

                if max(sig) < 0:
                    if max(sig) > -1e-3:
                        zero_level = max(sig) - 1e-12

                if min(sig) == 0:
                    zero_level = 1e-12

                if max(sig) == 0:
                    zero_level = -1e-12

                CS = fig.axes[0].tricontour(
                    triang, sig, [zero_level], linewidths=1, linestyles="dashed"
                )

        # plot the steel stresses
        steel_patches = []
        colours = []

        for idx, sig in enumerate(steel_sigs):
            steel_patches.append(
                mpatches.Polygon(
                    xy=list(concrete_section.steel_geometries[idx].geom.exterior.coords)
                )
            )
            colours.append(sig)

        patch = PatchCollection(steel_patches, cmap=cmap_steel)
        patch.set_array(colours)
        fig.axes[0].add_collection(patch)

        # add the colour bars
        fig.colorbar(
            trictr,
            label="Concrete Stress",
            format="%.2e",
            ticks=ticks_conc,
            cax=fig.axes[1],
        )
        fig.colorbar(
            patch,
            label="Steel Stress",
            format="%.2e",
            ticks=ticks_steel,
            cax=fig.axes[2],
        )

        ax.set_aspect("equal", anchor="C")

    return ax
