"""Post-processor methods."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from quantiphy import Quantity

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
        if not render or pause:
            plt.ioff()
        else:
            plt.ion()

        ax_supplied = False
        fig, ax = plt.subplots(**kwargs)

        try:
            if axis_index is None:
                axis_index = (0,) * ax.ndim  # pyright: ignore
            ax = ax[axis_index]  # pyright: ignore
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


def string_formatter(
    value: float,
    eng: bool,
    prec: int,
    scale: float = 1.0,
) -> str:
    """Formats a float using engineering or fixed notation.

    Args:
        value: Number to format
        eng: If set to ``True``, formats with engineering notation. If set to ``False``,
            formats with fixed notation.
        prec: The desired precision (i.e. one plus this value is the desired number of
            digits)
        scale: Factor by which to scale the value. Defaults to ``1.0``.

    Returns:
        Formatted string
    """
    q = Quantity(value)
    form = "eng" if eng else "fixed"
    val_fmt = q.render(
        form=form,
        show_units=False,
        prec=prec,
        scale=scale,
        strip_zeros=False,
    )
    spl = val_fmt.split("e")

    # if there is an exponent, render as 'x 10^n'
    if eng and len(spl) > 1:
        num = spl[0]
        exp = spl[1]
        return f"{num} x 10^{exp}"
    else:
        return val_fmt


def string_formatter_plots(
    value: float,
    prec: int,
) -> str:
    """Formats a float using engineering notation for plotting.

    Args:
        value: Number to format
        prec: The desired precision (i.e. one plus this value is the desired number of
            digits)

    Returns:
        Formatted string
    """
    q = Quantity(value)
    return q.render(
        form="eng", show_units=False, prec=prec, strip_zeros=True, strip_radix=True
    )


def string_formatter_stress(
    value: float,
    eng: bool,
    prec: int,
) -> str:
    """Formats a float using engineering notation for stress plotting.

    Args:
        value: Number to format
        eng: If set to ``True``, formats with engineering notation. If set to ``False``,
            formats with fixed notation.
        prec: The desired precision (i.e. one plus this value is the desired number of
            digits)

    Returns:
        Formatted string
    """
    q = Quantity(value)
    form = "eng" if eng else "fixed"
    val_fmt = q.render(
        form=form, show_units=False, prec=prec, strip_zeros=False, strip_radix=False
    )

    # ensure there is an e0
    if "e" not in val_fmt and eng:
        val_fmt += "e0"

    return val_fmt


@dataclass
class UnitDisplay:
    """Class for displaying units in concreteproperties.

    Attributes:
        length: Length unit string
        force: Force unit string
        mass: Mass unit string
        radians: If set to ``True``, displays angles in radians, otherwise displays
            angles in degrees. Defaults to ``True``.
        length_factor: Factor by which the ``length`` unit differs from the base units
        force_factor: Factor by which the ``force`` unit differs from the base units
        mass_factor: Factor by which the ``mass`` unit differs from the base units
    """

    length: str
    force: str
    mass: str
    radians: bool = True
    length_factor: float = 1.0
    force_factor: float = 1.0
    mass_factor: float = 1.0

    @property
    def length_unit(self) -> str:
        """Returns the length unit string."""
        return self.length if self.length == "" else f" {self.length}"

    @property
    def length_scale(self) -> float:
        """Returns the length scale."""
        return 1 / self.length_factor

    @property
    def force_unit(self) -> str:
        """Returns the force unit string."""
        return self.force if self.force == "" else f" {self.force}"

    @property
    def force_scale(self) -> float:
        """Returns the force scale."""
        return 1 / self.force_factor

    @property
    def mass_unit(self) -> str:
        """Returns the mass unit string."""
        return self.mass if self.mass == "" else f" {self.mass}"

    @property
    def mass_scale(self) -> float:
        """Returns the mass scale."""
        return 1 / self.mass_factor

    @property
    def angle_unit(self) -> str:
        """Returns the angle unit string."""
        return " rads" if self.radians else " degs"

    @property
    def angle_scale(self) -> float:
        """Returns the angle scale."""
        return 1 if self.radians else 180.0 / np.pi

    @property
    def area_unit(self) -> str:
        """Returns the area unit string."""
        return self.length if self.length == "" else f" {self.length}^2"

    @property
    def area_scale(self) -> float:
        """Returns the area scale."""
        return 1 / self.length_factor / self.length_factor

    @property
    def mass_per_length_unit(self) -> str:
        """Returns the mass/length unit string."""
        return self.mass if self.mass == "" else f" {self.mass}/{self.length}"

    @property
    def mass_per_length_scale(self) -> float:
        """Returns the mass/length scale."""
        return 1 / self.mass_factor * self.length_factor

    @property
    def moment_unit(self) -> str:
        """Returns the moment unit string."""
        return self.length if self.length == "" else f" {self.force}.{self.length}"

    @property
    def moment_scale(self) -> float:
        """Returns the moment scale."""
        return 1 / self.force_factor / self.length_factor

    @property
    def flex_rig_unit(self) -> str:
        """Returns the flexural rigidity unit string."""
        return self.length if self.length == "" else f" {self.force}.{self.length}^2"

    @property
    def flex_rig_scale(self) -> float:
        """Returns the flexural rigidity scale."""
        return 1 / self.force_factor / self.length_factor / self.length_factor

    @property
    def stress_unit(self) -> str:
        """Returns the stress unit string."""
        if self.length == "mm" and self.force == "N":
            return " MPa"
        elif self.length == "m" and self.force == "kN":
            return " kPa"
        elif self.length == "":
            return ""
        else:
            return f" {self.force}/{self.length}^2"

    @property
    def stress_scale(self) -> float:
        """Returns the stress scale."""
        return 1 / self.force_factor * self.length_factor * self.length_factor

    @property
    def length_3_unit(self) -> str:
        """Returns the length^3 unit string."""
        return self.length if self.length == "" else f" {self.length}^3"

    @property
    def length_3_scale(self) -> float:
        """Returns the length^3 scale."""
        return 1 / self.length_factor**3

    @property
    def length_4_unit(self) -> str:
        """Returns the length^4 unit string."""
        return self.length if self.length == "" else f" {self.length}^4"

    @property
    def length_4_scale(self) -> float:
        """Returns the length^4 scale."""
        return 1 / self.length_factor**4


si_n_mm = UnitDisplay(length="mm", force="N", mass="kg")
si_kn_m = UnitDisplay(
    length="m", force="kN", mass="kg", length_factor=1e3, force_factor=1e3
)
