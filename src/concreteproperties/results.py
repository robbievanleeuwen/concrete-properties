"""Contains results objects specific to various analysis types."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import CenteredNorm
from rich.console import Console
from rich.table import Table
from scipy.interpolate import interp1d
from sectionproperties.pre.geometry import CompoundGeometry
from shapely import Point, Polygon

from concreteproperties.post import plotting_context


if TYPE_CHECKING:
    import matplotlib.axes

    from concreteproperties.analysis_section import AnalysisSection
    from concreteproperties.concrete_section import ConcreteSection
    from concreteproperties.pre import CPGeom


@dataclass
class GrossProperties:
    """Class for storing gross concrete section properties.

    All properties with an ``e_`` preceding the property are multiplied by the elastic
    modulus. In order to obtain transformed properties, call the
    :meth:`~concreteproperties.concrete_section.ConcreteSection.get_transformed_gross_properties`
    method.
    """

    # section areas
    total_area: float = 0
    concrete_area: float = 0
    reinf_meshed_area: float = 0
    reinf_lumped_area: float = 0
    strand_area: float = 0
    e_a: float = 0

    # section mass
    mass: float = 0

    # section perimeter
    perimeter: float = 0

    # first moments of area
    e_qx: float = 0
    e_qy: float = 0
    qx_gross: float = 0
    qy_gross: float = 0

    # centroids
    cx: float = 0
    cy: float = 0
    cx_gross: float = 0
    cy_gross: float = 0

    # second moments of area
    e_ixx_g: float = 0
    e_iyy_g: float = 0
    e_ixy_g: float = 0
    e_ixx_c: float = 0
    e_iyy_c: float = 0
    e_ixy_c: float = 0
    e_i11: float = 0
    e_i22: float = 0

    # principal axis angle
    phi: float = 0

    # section moduli
    e_zxx_plus: float = 0
    e_zxx_minus: float = 0
    e_zyy_plus: float = 0
    e_zyy_minus: float = 0
    e_z11_plus: float = 0
    e_z11_minus: float = 0
    e_z22_plus: float = 0
    e_z22_minus: float = 0

    # other properties
    conc_ultimate_strain: float = 0
    n_prestress: float = 0
    m_prestress: float = 0

    def print_results(
        self,
        fmt: str = "8.6e",
    ) -> None:
        """Prints the gross concrete section properties to the terminal.

        Args:
            fmt: Number format
        """
        table = Table(title="Gross Concrete Section Properties")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        table.add_row("Total Area", "{:>{fmt}}".format(self.total_area, fmt=fmt))
        table.add_row("Concrete Area", "{:>{fmt}}".format(self.concrete_area, fmt=fmt))

        if self.reinf_meshed_area:
            table.add_row(
                "Meshed Reinforcement Area",
                "{:>{fmt}}".format(self.reinf_meshed_area, fmt=fmt),
            )

        table.add_row(
            "Lumped Reinforcement Area",
            "{:>{fmt}}".format(self.reinf_lumped_area, fmt=fmt),
        )

        if self.strand_area:
            table.add_row("Strand Area", "{:>{fmt}}".format(self.strand_area, fmt=fmt))

        table.add_row("Axial Rigidity (EA)", "{:>{fmt}}".format(self.e_a, fmt=fmt))
        table.add_row("Mass (per unit length)", "{:>{fmt}}".format(self.mass, fmt=fmt))
        table.add_row("Perimeter", "{:>{fmt}}".format(self.perimeter, fmt=fmt))
        table.add_row("E.Qx", "{:>{fmt}}".format(self.e_qx, fmt=fmt))
        table.add_row("E.Qy", "{:>{fmt}}".format(self.e_qy, fmt=fmt))
        table.add_row("x-Centroid", "{:>{fmt}}".format(self.cx, fmt=fmt))
        table.add_row("y-Centroid", "{:>{fmt}}".format(self.cy, fmt=fmt))
        table.add_row("x-Centroid (Gross)", "{:>{fmt}}".format(self.cx_gross, fmt=fmt))
        table.add_row("y-Centroid (Gross)", "{:>{fmt}}".format(self.cy_gross, fmt=fmt))
        table.add_row("E.Ixx_g", "{:>{fmt}}".format(self.e_ixx_g, fmt=fmt))
        table.add_row("E.Iyy_g", "{:>{fmt}}".format(self.e_iyy_g, fmt=fmt))
        table.add_row("E.Ixy_g", "{:>{fmt}}".format(self.e_ixy_g, fmt=fmt))
        table.add_row("E.Ixx_c", "{:>{fmt}}".format(self.e_ixx_c, fmt=fmt))
        table.add_row("E.Iyy_c", "{:>{fmt}}".format(self.e_iyy_c, fmt=fmt))
        table.add_row("E.Ixy_c", "{:>{fmt}}".format(self.e_ixy_c, fmt=fmt))
        table.add_row("E.I11", "{:>{fmt}}".format(self.e_i11, fmt=fmt))
        table.add_row("E.I22", "{:>{fmt}}".format(self.e_i22, fmt=fmt))
        table.add_row("Principal Axis Angle", "{:>{fmt}}".format(self.phi, fmt=fmt))
        table.add_row("E.Zxx+", "{:>{fmt}}".format(self.e_zxx_plus, fmt=fmt))
        table.add_row("E.Zxx-", "{:>{fmt}}".format(self.e_zxx_minus, fmt=fmt))
        table.add_row("E.Zyy+", "{:>{fmt}}".format(self.e_zyy_plus, fmt=fmt))
        table.add_row("E.Zyy-", "{:>{fmt}}".format(self.e_zyy_minus, fmt=fmt))
        table.add_row("E.Z11+", "{:>{fmt}}".format(self.e_z11_plus, fmt=fmt))
        table.add_row("E.Z11-", "{:>{fmt}}".format(self.e_z11_minus, fmt=fmt))
        table.add_row("E.Z22+", "{:>{fmt}}".format(self.e_z22_plus, fmt=fmt))
        table.add_row("E.Z22-", "{:>{fmt}}".format(self.e_z22_minus, fmt=fmt))
        table.add_row(
            "Ultimate Concrete Strain",
            "{:>{fmt}}".format(self.conc_ultimate_strain, fmt=fmt),
        )

        # add prestressed results if they exist
        if self.n_prestress:
            table.add_row("n_prestress", "{:>{fmt}}".format(self.n_prestress, fmt=fmt))
            table.add_row("m_prestress", "{:>{fmt}}".format(self.m_prestress, fmt=fmt))

        console = Console()
        console.print(table)


@dataclass
class TransformedGrossProperties:
    """Class for storing transformed gross concrete section properties.

    Args:
        concrete_properties: Concrete properties object
        elastic_modulus: Reference elastic modulus
    """

    concrete_properties: GrossProperties = field(repr=False)
    elastic_modulus: float

    # area
    area: float = 0

    # first moments of area
    qx: float = 0
    qy: float = 0

    # second moments of area
    ixx_g: float = 0
    iyy_g: float = 0
    ixy_g: float = 0
    ixx_c: float = 0
    iyy_c: float = 0
    ixy_c: float = 0
    i11: float = 0
    i22: float = 0

    # section moduli
    zxx_plus: float = 0
    zxx_minus: float = 0
    zyy_plus: float = 0
    zyy_minus: float = 0
    z11_plus: float = 0
    z11_minus: float = 0
    z22_plus: float = 0
    z22_minus: float = 0

    def __post_init__(self) -> None:
        """Post init method."""
        self.area = self.concrete_properties.e_a / self.elastic_modulus
        self.qx = self.concrete_properties.e_qx / self.elastic_modulus
        self.qy = self.concrete_properties.e_qy / self.elastic_modulus
        self.ixx_g = self.concrete_properties.e_ixx_g / self.elastic_modulus
        self.iyy_g = self.concrete_properties.e_iyy_g / self.elastic_modulus
        self.ixy_g = self.concrete_properties.e_ixy_g / self.elastic_modulus
        self.ixx_c = self.concrete_properties.e_ixx_c / self.elastic_modulus
        self.iyy_c = self.concrete_properties.e_iyy_c / self.elastic_modulus
        self.ixy_c = self.concrete_properties.e_ixy_c / self.elastic_modulus
        self.i11 = self.concrete_properties.e_i11 / self.elastic_modulus
        self.i22 = self.concrete_properties.e_i22 / self.elastic_modulus
        self.zxx_plus = self.concrete_properties.e_zxx_plus / self.elastic_modulus
        self.zxx_minus = self.concrete_properties.e_zxx_minus / self.elastic_modulus
        self.zyy_plus = self.concrete_properties.e_zyy_plus / self.elastic_modulus
        self.zyy_minus = self.concrete_properties.e_zyy_minus / self.elastic_modulus
        self.z11_plus = self.concrete_properties.e_z11_plus / self.elastic_modulus
        self.z11_minus = self.concrete_properties.e_z11_minus / self.elastic_modulus
        self.z22_plus = self.concrete_properties.e_z22_plus / self.elastic_modulus
        self.z22_minus = self.concrete_properties.e_z22_minus / self.elastic_modulus

    def print_results(
        self,
        fmt: str = "8.6e",
    ) -> None:
        """Prints the transformed gross concrete section properties to the terminal.

        Args:
            fmt: Number format
        """
        table = Table(title="Transformed Gross Concrete Section Properties")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        table.add_row("E_ref", "{:>{fmt}}".format(self.elastic_modulus, fmt=fmt))
        table.add_row("Area", "{:>{fmt}}".format(self.area, fmt=fmt))
        table.add_row("Qx", "{:>{fmt}}".format(self.qx, fmt=fmt))
        table.add_row("Qy", "{:>{fmt}}".format(self.qy, fmt=fmt))
        table.add_row("Ixx_g", "{:>{fmt}}".format(self.ixx_g, fmt=fmt))
        table.add_row("Iyy_g", "{:>{fmt}}".format(self.iyy_g, fmt=fmt))
        table.add_row("Ixy_g", "{:>{fmt}}".format(self.ixy_g, fmt=fmt))
        table.add_row("Ixx_c", "{:>{fmt}}".format(self.ixx_c, fmt=fmt))
        table.add_row("Iyy_c", "{:>{fmt}}".format(self.iyy_c, fmt=fmt))
        table.add_row("Ixy_c", "{:>{fmt}}".format(self.ixy_c, fmt=fmt))
        table.add_row("I11", "{:>{fmt}}".format(self.i11, fmt=fmt))
        table.add_row("I22", "{:>{fmt}}".format(self.i22, fmt=fmt))
        table.add_row("Zxx+", "{:>{fmt}}".format(self.zxx_plus, fmt=fmt))
        table.add_row("Zxx-", "{:>{fmt}}".format(self.zxx_minus, fmt=fmt))
        table.add_row("Zyy+", "{:>{fmt}}".format(self.zyy_plus, fmt=fmt))
        table.add_row("Zyy-", "{:>{fmt}}".format(self.zyy_minus, fmt=fmt))
        table.add_row("Z11+", "{:>{fmt}}".format(self.z11_plus, fmt=fmt))
        table.add_row("Z11-", "{:>{fmt}}".format(self.z11_minus, fmt=fmt))
        table.add_row("Z22+", "{:>{fmt}}".format(self.z22_plus, fmt=fmt))
        table.add_row("Z22-", "{:>{fmt}}".format(self.z22_minus, fmt=fmt))

        console = Console()
        console.print(table)


@dataclass
class CrackedResults:
    r"""Class for storing cracked concrete section properties.

    All properties with an ``e_`` preceding the property are multiplied by the elastic
    modulus. In order to obtain transformed properties, call the
    :meth:`~concreteproperties.results.CrackedResults.calculate_transformed_properties`
    method.

    Args:
        theta: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \theta \leq \pi`)
    """

    theta: float
    n: float = 0
    m: float = 0
    m_cr: float | tuple[float, float] = 0
    d_nc: float = 0
    cracked_geometries: list[CPGeom] = field(default_factory=list, repr=False)
    e_a_cr: float = 0
    e_qx_cr: float = 0
    e_qy_cr: float = 0
    cx: float = 0
    cy: float = 0
    e_ixx_g_cr: float = 0
    e_iyy_g_cr: float = 0
    e_ixy_g_cr: float = 0
    e_ixx_c_cr: float = 0
    e_iyy_c_cr: float = 0
    e_ixy_c_cr: float = 0
    e_iuu_cr: float = 0
    e_i11_cr: float = 0
    e_i22_cr: float = 0
    phi_cr: float = 0

    # transformed properties
    elastic_modulus_ref: float | None = None
    a_cr: float | None = None
    qx_cr: float | None = None
    qy_cr: float | None = None
    ixx_g_cr: float | None = None
    iyy_g_cr: float | None = None
    ixy_g_cr: float | None = None
    ixx_c_cr: float | None = None
    iyy_c_cr: float | None = None
    ixy_c_cr: float | None = None
    iuu_cr: float | None = None
    i11_cr: float | None = None
    i22_cr: float | None = None

    def reset_results(self) -> None:
        """Resets the analysis results."""
        self.e_a_cr = 0
        self.e_qx_cr = 0
        self.e_qy_cr = 0
        self.cx = 0
        self.cy = 0
        self.e_ixx_g_cr = 0
        self.e_iyy_g_cr = 0
        self.e_ixy_g_cr = 0
        self.e_ixx_c_cr = 0
        self.e_iyy_c_cr = 0
        self.e_ixy_c_cr = 0
        self.e_iuu_cr = 0
        self.e_i11_cr = 0
        self.e_i22_cr = 0
        self.phi_cr = 0

    def calculate_transformed_properties(
        self,
        elastic_modulus: float,
    ) -> None:
        """Calculates and stores transformed cracked properties.

        Args:
            elastic_modulus: Reference elastic modulus
        """
        self.elastic_modulus_ref = elastic_modulus
        self.a_cr = self.e_a_cr / elastic_modulus
        self.qx_cr = self.e_qx_cr / elastic_modulus
        self.qy_cr = self.e_qy_cr / elastic_modulus
        self.ixx_g_cr = self.e_ixx_g_cr / elastic_modulus
        self.iyy_g_cr = self.e_iyy_g_cr / elastic_modulus
        self.ixy_g_cr = self.e_ixy_g_cr / elastic_modulus
        self.ixx_c_cr = self.e_ixx_c_cr / elastic_modulus
        self.iyy_c_cr = self.e_iyy_c_cr / elastic_modulus
        self.ixy_c_cr = self.e_ixy_c_cr / elastic_modulus
        self.iuu_cr = self.e_iuu_cr / elastic_modulus
        self.i11_cr = self.e_i11_cr / elastic_modulus
        self.i22_cr = self.e_i22_cr / elastic_modulus

    def plot_cracked_geometries(
        self,
        title: str = "Cracked Geometries",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots geometries that remain (compression/reinf.) after a cracked analysis.

        Args:
            title: Plot title
            kwargs: Passed to
                :meth:`~sectionproperties.pre.geometry.CompoundGeometry.plot_geometry`

        Returns:
            Matplotlib axes object
        """
        return CompoundGeometry(
            [geom.to_sp_geom() for geom in self.cracked_geometries]
        ).plot_geometry(title=title, **kwargs)

    def print_results(
        self,
        fmt: str = "8.6e",
    ) -> None:
        """Prints the cracked concrete section properties to the terminal.

        Args:
            fmt: Number format
        """
        table = Table(title="Cracked Concrete Section Properties")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        table.add_row("theta", "{:>{fmt}}".format(self.theta, fmt=fmt))
        table.add_row("n", "{:>{fmt}}".format(self.n, fmt=fmt))
        table.add_row("m", "{:>{fmt}}".format(self.m, fmt=fmt))

        if self.elastic_modulus_ref:
            table.add_row(
                "E_ref", "{:>{fmt}}".format(self.elastic_modulus_ref, fmt=fmt)
            )

        if isinstance(self.m_cr, tuple):
            table.add_row("m_cr_pos", "{:>{fmt}}".format(self.m_cr[0], fmt=fmt))
            table.add_row("m_cr_neg", "{:>{fmt}}".format(self.m_cr[1], fmt=fmt))
        else:
            table.add_row("m_cr", "{:>{fmt}}".format(self.m_cr, fmt=fmt))

        table.add_row("d_nc", "{:>{fmt}}".format(self.d_nc, fmt=fmt))

        if self.a_cr:
            table.add_row("A_cr", "{:>{fmt}}".format(self.a_cr, fmt=fmt))

        table.add_row("E.A_cr", "{:>{fmt}}".format(self.e_a_cr, fmt=fmt))

        if self.qx_cr:
            table.add_row("Qx_cr", "{:>{fmt}}".format(self.qx_cr, fmt=fmt))
            table.add_row("Qy_cr", "{:>{fmt}}".format(self.qy_cr, fmt=fmt))

        table.add_row("E.Qx_cr", "{:>{fmt}}".format(self.e_qx_cr, fmt=fmt))
        table.add_row("E.Qy_cr", "{:>{fmt}}".format(self.e_qy_cr, fmt=fmt))
        table.add_row("x-Centroid", "{:>{fmt}}".format(self.cx, fmt=fmt))
        table.add_row("y-Centroid", "{:>{fmt}}".format(self.cy, fmt=fmt))

        if self.ixx_g_cr:
            table.add_row("Ixx_g_cr", "{:>{fmt}}".format(self.ixx_g_cr, fmt=fmt))
            table.add_row("Iyy_g_cr", "{:>{fmt}}".format(self.iyy_g_cr, fmt=fmt))
            table.add_row("Ixy_g_cr", "{:>{fmt}}".format(self.ixy_g_cr, fmt=fmt))
            table.add_row("Ixx_c_cr", "{:>{fmt}}".format(self.ixx_c_cr, fmt=fmt))
            table.add_row("Iyy_c_cr", "{:>{fmt}}".format(self.iyy_c_cr, fmt=fmt))
            table.add_row("Ixy_c_cr", "{:>{fmt}}".format(self.ixy_c_cr, fmt=fmt))
            table.add_row("Iuu_cr", "{:>{fmt}}".format(self.iuu_cr, fmt=fmt))
            table.add_row("I11_cr", "{:>{fmt}}".format(self.i11_cr, fmt=fmt))
            table.add_row("I22_cr", "{:>{fmt}}".format(self.i22_cr, fmt=fmt))

        table.add_row("E.Ixx_g_cr", "{:>{fmt}}".format(self.e_ixx_g_cr, fmt=fmt))
        table.add_row("E.Iyy_g_cr", "{:>{fmt}}".format(self.e_iyy_g_cr, fmt=fmt))
        table.add_row("E.Ixy_g_cr", "{:>{fmt}}".format(self.e_ixy_g_cr, fmt=fmt))
        table.add_row("E.Ixx_c_cr", "{:>{fmt}}".format(self.e_ixx_c_cr, fmt=fmt))
        table.add_row("E.Iyy_c_cr", "{:>{fmt}}".format(self.e_iyy_c_cr, fmt=fmt))
        table.add_row("E.Ixy_c_cr", "{:>{fmt}}".format(self.e_ixy_c_cr, fmt=fmt))
        table.add_row("E.Iuu_cr", "{:>{fmt}}".format(self.e_iuu_cr, fmt=fmt))
        table.add_row("E.I11_cr", "{:>{fmt}}".format(self.e_i11_cr, fmt=fmt))
        table.add_row("E.I22_cr", "{:>{fmt}}".format(self.e_i22_cr, fmt=fmt))
        table.add_row("phi_cr", "{:>{fmt}}".format(self.phi_cr, fmt=fmt))

        console = Console()
        console.print(table)


@dataclass
class MomentCurvatureResults:
    r"""Class for storing moment curvature results.

    Args:
        theta: Angle (in radians) the neutral axis makes with the horizontal
        n_target: Target axial force axis (:math:`-\pi \leq \theta \leq \pi`)
        kappa: List of curvatures
        n: List of axial forces
        m_x: List of bending moments about the x-axis
        m_y: List of bending moments about the y-axis
        m_xy: List of resultant bending moments
        failure_geometry: Geometry object of the region of the cross-section that
            failed, ending the moment curvature analysis
        convergence: The critical ratio between the strain and the failure strain within
            the cross-section for each curvature step in the analysis. A value of one
            indicates failure.
    """

    # results
    theta: float
    n_target: float
    kappa: list[float] = field(default_factory=list)
    n: list[float] = field(default_factory=list)
    m_x: list[float] = field(default_factory=list)
    m_y: list[float] = field(default_factory=list)
    m_xy: list[float] = field(default_factory=list)
    failure_geometry: CPGeom = field(init=False, repr=False)
    convergence: list[float] = field(default_factory=list)

    # for analysis
    _kappa: float = field(default=0, repr=False)
    _n_i: float = field(default=0, repr=False)
    _m_x_i: float = field(default=0, repr=False)
    _m_y_i: float = field(default=0, repr=False)
    _failure: bool = field(default=False, repr=False)
    _failure_convergence: float = field(default=0, repr=False)

    def plot_results(
        self,
        m_scale: float = 1e-6,
        fmt: str = "o-",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots the moment curvature results.

        Args:
            m_scale: Scaling factor to apply to bending moment
            fmt: Plot format string
            kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        Returns:
            Matplotlib axes object
        """
        # scale moments
        moments = np.array(self.m_xy) * m_scale

        # create plot and setup the plot
        with plotting_context(title="Moment-Curvature", **kwargs) as (fig, ax):
            assert ax
            ax.plot(self.kappa, moments, fmt)
            plt.xlabel("Curvature")
            plt.ylabel("Moment")
            plt.grid(True)

        return ax

    @staticmethod
    def plot_multiple_results(
        moment_curvature_results: list[MomentCurvatureResults],
        labels: list[str],
        m_scale: float = 1e-6,
        fmt: str = "o-",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots multiple moment curvature results.

        Args:
            moment_curvature_results: List of moment curvature results objects
            labels: List of labels for each moment curvature diagram
            m_scale: Scaling factor to apply to bending moment
            fmt: Plot format string
            kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        Returns:
            Matplotlib axes object
        """
        # create plot and setup the plot
        with plotting_context(title="Moment-Curvature", **kwargs) as (fig, ax):
            assert ax
            idx = 0

            # for each M-k curve
            for idx, mk_result in enumerate(moment_curvature_results):
                # scale results
                kappas = np.array(mk_result.kappa)
                moments = np.array(mk_result.m_xy) * m_scale

                ax.plot(kappas, moments, fmt, label=labels[idx])

            plt.xlabel("Curvature")
            plt.ylabel("Moment")
            plt.grid(True)

            # if there is more than one curve show legend
            if idx > 0:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        return ax

    def plot_failure_geometry(
        self,
        title: str = "Failure Geometry",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots the geometry that fails in the moment curvature analysis.

        Args:
            title: Plot title
            kwargs: Passed to
                :meth:`~sectionproperties.pre.geometry.CompoundGeometry.plot_geometry`

        Returns:
            Matplotlib axes object
        """
        return self.failure_geometry.plot_geometry(title=title, **kwargs)

    def get_curvature(
        self,
        moment: float,
    ) -> float:
        """Given a moment, uses the moment-curvature results to interpolate a curvature.

        Args:
            moment: Bending moment at which to obtain curvature

        Raises:
            ValueError: If supplied moment is outside bounds of moment-curvature
                results.

        Returns:
            Curvature
        """
        # check moment is within bounds of results
        m_min = min(self.m_xy)
        m_max = max(self.m_xy)

        if moment > m_max or moment < m_min:
            raise ValueError(
                "moment must be within the bounds of the moment-curvature results."
            )

        f_kappa = interp1d(
            x=self.m_xy,
            y=self.kappa,
            kind="linear",
        )

        return float(f_kappa(moment))


@dataclass(order=True)
class UltimateBendingResults:
    r"""Class for storing ultimate bending results.

    Args:
        theta: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \theta \leq \pi`)
        d_n: Ultimate neutral axis depth
        k_u: Neutral axis parameter *(d_n / d)*
        n: Resultant axial force
        m_x: Resultant bending moment about the x-axis
        m_y: Resultant bending moment about the y-axis
        m_xy: Resultant bending moment
        label: Result label
    """

    # bending angle
    theta: float

    # ultimate neutral axis depth
    d_n: float = 0
    k_u: float = 0

    # resultant actions
    n: float = 0
    m_x: float = 0
    m_y: float = 0
    m_xy: float = 0

    # label
    label: str | None = field(default=None, compare=False)

    def print_results(
        self,
        fmt: str = "8.6e",
    ) -> None:
        """Prints the ultimate bending results to the terminal.

        Args:
            fmt: Number format
        """
        table = Table(title="Ultimate Bending Results")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        if self.label:
            table.add_row("Label", self.label)

        table.add_row("Bending Angle - theta", "{:>{fmt}}".format(self.theta, fmt=fmt))
        table.add_row("Neutral Axis Depth - d_n", "{:>{fmt}}".format(self.d_n, fmt=fmt))
        table.add_row(
            "Neutral Axis Parameter - k_u", "{:>{fmt}}".format(self.k_u, fmt=fmt)
        )
        table.add_row("Axial Force", "{:>{fmt}}".format(self.n, fmt=fmt))
        table.add_row("Bending Capacity - m_x", "{:>{fmt}}".format(self.m_x, fmt=fmt))
        table.add_row("Bending Capacity - m_y", "{:>{fmt}}".format(self.m_y, fmt=fmt))
        table.add_row("Bending Capacity - m_xy", "{:>{fmt}}".format(self.m_xy, fmt=fmt))

        console = Console()
        console.print(table)


@dataclass
class MomentInteractionResults:
    """Class for storing moment interaction results.

    Args:
        results: List of ultimate bending result objects
    """

    results: list[UltimateBendingResults] = field(default_factory=list)

    def sort_results(self) -> None:
        """Sorts the results by decreasing axial force."""
        self.results.sort(reverse=True)

        # remove duplicates from sorted list
        new_results = []

        for res in self.results:
            if res not in new_results:
                new_results.append(res)

        self.results = new_results

    def get_results_lists(
        self,
        moment: str,
    ) -> tuple[list[float], list[float]]:
        """Returns a list of axial forces and moments.

        Args:
            moment: Which moment to plot, acceptable values are ``"m_x"``, ``"m_y"`` or
                ``"m_xy"``

        Raises:
            ValueError: If the moment string is not valid

        Returns:
            List of axial forces and moments (``n``, ``m``)
        """
        # build list of results
        n_list = []
        m_list = []

        for result in self.results:
            n_list.append(result.n)

            if moment == "m_x":
                m_list.append(result.m_x)
            elif moment == "m_y":
                m_list.append(result.m_y)
            elif moment == "m_xy":
                m_list.append(result.m_xy)
            else:
                raise ValueError(f"{moment} not an acceptable value for moment.")

        return n_list, m_list

    def plot_diagram(
        self,
        n_scale: float = 1e-3,
        m_scale: float = 1e-6,
        moment: str = "m_x",
        fmt: str = "o-",
        labels: bool = False,
        label_offset: bool = False,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots a moment interaction diagram.

        Args:
            n_scale: Scaling factor to apply to axial force
            m_scale: Scaling factor to apply to the bending moment
            moment: Which moment to plot, acceptable values are ``"m_x"``, ``"m_y"`` or
                ``"m_xy"``
            fmt: Plot format string
            labels: If set to True, also plots labels on the diagram
            label_offset: If set to True, attempts to offset the label from the diagram
            kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        Returns:
            Matplotlib axes object
        """
        # create plot and setup the plot
        with plotting_context(title="Moment Interaction Diagram", **kwargs) as (
            fig,
            ax,
        ):
            assert ax

            # get results
            n_list, m_list = self.get_results_lists(moment=moment)

            # scale results
            forces = np.array(n_list) * n_scale
            moments = np.array(m_list) * m_scale

            # plot diagram
            ax.plot(moments, forces, fmt)

            # plot labels
            if labels:
                if label_offset:
                    # compute gradients of curve and aspect ratio of plot
                    grad = np.gradient([moments, forces], axis=1)
                    x_diff = ax.get_xlim()
                    y_diff = ax.get_ylim()
                    ar = (y_diff[1] - y_diff[0]) / (x_diff[1] - x_diff[0])

                for idx, m in enumerate(m_list):
                    if self.results[idx].label:
                        # get x,y position on plot
                        x = m * m_scale
                        y = n_list[idx] * n_scale

                        if label_offset:
                            # calculate text offset
                            grad_pt = grad[1, idx] / grad[0, idx] / ar
                            if grad_pt == 0:
                                norm_angle = np.pi / 2
                            else:
                                norm_angle = np.arctan2(-1 / grad_pt, 1)
                            x_t = np.cos(norm_angle) * 20
                            y_t = np.sin(norm_angle) * 20
                            annotate_dict = {
                                "xytext": (x_t, y_t),
                                "textcoords": "offset points",
                                "arrowprops": dict(
                                    arrowstyle="->",
                                    connectionstyle="angle,angleA=0,angleB=90,rad=10",
                                ),
                                "bbox": dict(boxstyle="round", fc="0.8"),
                            }
                        else:
                            annotate_dict = {}

                        # plot text
                        ax.annotate(
                            text=self.results[idx].label, xy=(x, y), **annotate_dict
                        )

            plt.xlabel("Bending Moment")
            plt.ylabel("Axial Force")
            plt.grid(True)

        return ax

    @staticmethod
    def plot_multiple_diagrams(
        moment_interaction_results: list[MomentInteractionResults],
        labels: list[str],
        n_scale: float = 1e-3,
        m_scale: float = 1e-6,
        moment: str = "m_x",
        fmt: str = "o-",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots multiple moment interaction diagrams.

        Args:
            moment_interaction_results: List of moment interaction results objects
            labels: List of labels for each moment interaction diagram
            n_scale: Scaling factor to apply to axial force
            m_scale: Scaling factor to apply to bending moment
            moment: Which moment to plot, acceptable values are ``"m_x"``, ``"m_y"`` or
                ``"m_xy"``
            fmt: Plot format string
            kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        Returns:
            Matplotlib axes object
        """
        # create plot and setup the plot
        with plotting_context(title="Moment Interaction Diagram", **kwargs) as (
            fig,
            ax,
        ):
            assert ax
            idx = 0

            # for each M-N curve
            for idx, mi_result in enumerate(moment_interaction_results):
                n_list, m_list = mi_result.get_results_lists(moment=moment)

                # scale results
                forces = np.array(n_list) * n_scale
                moments = np.array(m_list) * m_scale

                ax.plot(moments, forces, fmt, label=labels[idx])

            plt.xlabel("Bending Moment")
            plt.ylabel("Axial Force")
            plt.grid(True)

            # if there is more than one curve show legend
            if idx > 0:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        return ax

    def point_in_diagram(
        self,
        n: float,
        m: float,
        moment: str = "m_x",
    ) -> bool:
        """Determines whether or not the design point lies within the diagram.

        Args:
            n: Axial force
            m: Bending moment
            moment: Which moment to analyse, acceptable values are ``"m_x"``, ``"m_y"``
                or ``"m_xy"``

        Returns:
            True, if combination of axial force and moment is within the diagram
        """
        # get results
        n_list, m_list = self.get_results_lists(moment=moment)

        # create a polygon from points on diagram
        poly_points = []

        for idx, mom in enumerate(m_list):
            poly_points.append((mom, n_list[idx]))

        poly = Polygon(poly_points)
        point = Point(m, n)

        return poly.contains(point)


@dataclass
class BiaxialBendingResults:
    """Class for storing biaxial bending results.

    Args:
        n: Net axial force
        results: List of ultimate bending result objects
    """

    n: float
    results: list[UltimateBendingResults] = field(default_factory=list)

    def get_results_lists(
        self,
    ) -> tuple[list[float], list[float]]:
        """Returns a list and moments about the ``x`` and ``y`` axes.

        Returns:
            List of axial forces and moments (``mx``, ``my``)
        """
        # build list of results
        m_x_list = []
        m_y_list = []

        for result in self.results:
            m_x_list.append(result.m_x)
            m_y_list.append(result.m_y)

        return m_x_list, m_y_list

    def plot_diagram(
        self,
        m_scale: float = 1e-6,
        fmt: str = "o-",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots a biaxial bending diagram.

        Args:
            m_scale: Scaling factor to apply to bending moment
            fmt: Plot format string
            kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        Returns:
            Matplotlib axes object
        """
        m_x_list, m_y_list = self.get_results_lists()

        # create plot and setup the plot
        with plotting_context(
            title=f"Biaxial Bending Diagram, $N = {self.n:.3e}$", **kwargs
        ) as (fig, ax):
            assert ax

            # scale results
            m_x = np.array(m_x_list) * m_scale
            m_y = np.array(m_y_list) * m_scale

            ax.plot(m_x, m_y, fmt)

            plt.xlabel("Bending Moment $M_x$")
            plt.ylabel("Bending Moment $M_y$")
            plt.grid(True)

        return ax

    @staticmethod
    def plot_multiple_diagrams_2d(
        biaxial_bending_results: list[BiaxialBendingResults],
        labels: list[str] | None = None,
        m_scale: float = 1e-6,
        fmt: str = "o-",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots multiple biaxial bending diagrams in a 2D plot.

        Args:
            biaxial_bending_results: List of biaxial bending results objects
            labels: List of labels for each biaxial bending diagram, if not provided
                labels are axial forces
            m_scale: Scaling factor to apply to bending moment
            fmt: Plot format string
            kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        Returns:
            Matplotlib axes object
        """
        # create plot and setup the plot
        with plotting_context(title="Biaxial Bending Diagram", **kwargs) as (fig, ax):
            assert ax
            idx = 0

            # generate default labels
            if labels is None:
                labels = []
                default_labels = True
            else:
                default_labels = False

            # for each M-N curve
            for idx, bb_result in enumerate(biaxial_bending_results):
                m_x_list, m_y_list = bb_result.get_results_lists()

                # scale results
                m_x_list = np.array(m_x_list) * m_scale
                m_y_list = np.array(m_y_list) * m_scale

                # generate default labels
                if default_labels:
                    labels.append(f"N = {bb_result.n:.3e}")

                ax.plot(m_x_list, m_y_list, fmt, label=labels[idx])

            plt.xlabel("Bending Moment $M_x$")
            plt.ylabel("Bending Moment $M_y$")
            plt.grid(True)

            # if there is more than one curve show legend
            if idx > 0:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        return ax

    @staticmethod
    def plot_multiple_diagrams_3d(
        biaxial_bending_results: list[BiaxialBendingResults],
        n_scale: float = 1e-3,
        m_scale: float = 1e-6,
        fmt: str = "-",
    ) -> matplotlib.axes.Axes:
        """Plots multiple biaxial bending diagrams in a 3D plot.

        Args:
            biaxial_bending_results: List of biaxial bending results objects
            n_scale: Scaling factor to apply to axial force
            m_scale: Scaling factor to apply to bending moment
            fmt: Plot format string

        Returns:
            Matplotlib axes object
        """
        # make 3d plot
        plt.figure()
        ax = plt.axes(projection="3d")

        # for each curve
        for bb_result in biaxial_bending_results:
            m_x_list, m_y_list = bb_result.get_results_lists()

            # scale results
            n_list = bb_result.n * n_scale * np.ones(len(m_x_list))
            m_x_list = np.array(m_x_list) * m_scale
            m_y_list = np.array(m_y_list) * m_scale

            ax.plot3D(m_x_list, m_y_list, n_list, fmt)

        ax.set_xlabel("Bending Moment $M_x$")
        ax.set_ylabel("Bending Moment $M_y$")
        ax.set_zlabel("Axial Force $N$")
        plt.show()

        return ax

    def point_in_diagram(
        self,
        m_x: float,
        m_y: float,
    ) -> bool:
        """Determines whether or not the design point lies within the biaxial diagram.

        Args:
            m_x: Bending moment about the x-axis
            m_y: Bending moment about the y-axis

        Returns:
            True, if combination of bendings moments is within the diagram
        """
        # create a polygon from points on diagram
        poly_points = []

        for ult_res in self.results:
            poly_points.append((ult_res.m_x, ult_res.m_y))

        poly = Polygon(poly_points)
        point = Point(m_x, m_y)

        return poly.contains(point)


@dataclass
class StressResult:
    """Class for storing stress results.

    The lever arm is computed to the elastic centroid.

    Args:
        concrete_analysis_sections: List of concrete analysis section objects
            present in the stress analysis, which can be visualised by calling the
            :meth:`~concreteproperties.analysis_section.AnalysisSection.plot_mesh` or
            :meth:`~concreteproperties.analysis_section.AnalysisSection.plot_shape`
        concrete_stresses: List of concrete stresses at the nodes of each concrete
            analysis section
        concrete_forces: List of net forces for each concrete analysis section and
            its lever arm (``force``, ``d_x``, ``d_y``)
        meshed_reinforcement_sections: List of meshed reinforcement section objects
            present in the stress analysis
        meshed_reinforcement_stresses: List of meshed reinforcement stresses at the
            nodes of each meshed reinforcement analysis section
        meshed_reinforcement_forces: List of net forces for each meshed reinforcement
            analysis section and its lever arm (``force``, ``d_x``, ``d_y``)
        lumped_reinforcement_geometries: List of lumped reinforcement geometry
            objects present in the stress analysis
        lumped_reinforcement_stresses: List of lumped reinforcement stresses for
            each lumped geometry
        lumped_reinforcement_strains: List of lumped reinforcement strains for each
            lumped geometry
        lumped_reinforcement_forces: List of net forces for each lumped reinforcement
            geometry and its lever arm (``force``, ``d_x``, ``d_y``)
        strand_geometries: List of strand geometry objects present in the stress
            analysis
        strand_stresses: List of strand stresses for each strand
        strand_strains: List of strand strains for each strand
        strand_forces: List of net forces for each strand geometry and its lever arm
            (``force``, ``d_x``, ``d_y``)
    """

    concrete_section: ConcreteSection
    concrete_analysis_sections: list[AnalysisSection]
    concrete_stresses: list[np.ndarray]
    concrete_forces: list[tuple[float, float, float]]
    meshed_reinforcement_sections: list[AnalysisSection]
    meshed_reinforcement_stresses: list[np.ndarray]
    meshed_reinforcement_forces: list[tuple[float, float, float]]
    lumped_reinforcement_geometries: list[CPGeom]
    lumped_reinforcement_stresses: list[float]
    lumped_reinforcement_strains: list[float]
    lumped_reinforcement_forces: list[tuple[float, float, float]]
    strand_geometries: list[CPGeom] = field(default_factory=list)
    strand_stresses: list[float] = field(default_factory=list)
    strand_strains: list[float] = field(default_factory=list)
    strand_forces: list[tuple[float, float, float]] = field(default_factory=list)
    _m_net: float | None = field(default=None, repr=False)

    def plot_stress(
        self,
        title: str = "Stress",
        conc_cmap: str = "RdGy",
        reinf_cmap: str = "bwr",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots concrete and steel stresses on a concrete section.

        Args:
            title: Plot title
            conc_cmap: Colour map for the concrete stress
            reinf_cmap: Colour map for the reinforcement stress
            kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        Returns:
            Matplotlib axes object
        """
        with plotting_context(
            title=title,
            aspect=True,
            **dict(
                kwargs, nrows=1, ncols=3, gridspec_kw={"width_ratios": [1, 0.08, 0.08]}
            ),
        ) as (fig, ax):
            assert fig
            assert ax

            # plot background
            self.concrete_section.plot_section(
                background=True, **dict(kwargs, ax=fig.axes[0])
            )

            # set up the colormaps
            cmap_conc = matplotlib.colormaps.get_cmap(cmap=conc_cmap)
            cmap_reinf = matplotlib.colormaps.get_cmap(cmap=reinf_cmap)

            # determine minimum and maximum stress values for the contour list
            # add tolerance for plotting stress blocks
            conc_sig_min = min([min(x) for x in self.concrete_stresses]) - 1e-12
            conc_sig_max = max([max(x) for x in self.concrete_stresses]) + 1e-12

            # if there is meshed reinforcement, calculate min and max
            if self.meshed_reinforcement_stresses:
                meshed_reinf_sig_min = (
                    min([min(x) for x in self.meshed_reinforcement_stresses]) - 1e-12
                )
                meshed_reinf_sig_max = (
                    max([max(x) for x in self.meshed_reinforcement_stresses]) + 1e-12
                )
            else:
                meshed_reinf_sig_min = None
                meshed_reinf_sig_max = None

            # if there is lumped reinforcement, calculate min and max
            if self.lumped_reinforcement_stresses or self.strand_stresses:
                lumped_reinf_sig_min = min(
                    self.lumped_reinforcement_stresses + self.strand_stresses
                )
                lumped_reinf_sig_max = max(
                    self.lumped_reinforcement_stresses + self.strand_stresses
                )
            else:
                lumped_reinf_sig_min = None
                lumped_reinf_sig_max = None

            # determine min and max reinforcement stresess
            if (
                meshed_reinf_sig_min
                and meshed_reinf_sig_max
                and lumped_reinf_sig_min
                and lumped_reinf_sig_max
            ):
                reinf_sig_min = min(meshed_reinf_sig_min, lumped_reinf_sig_min)
                reinf_sig_max = max(meshed_reinf_sig_max, lumped_reinf_sig_max)
            elif meshed_reinf_sig_min and meshed_reinf_sig_max:
                reinf_sig_min = meshed_reinf_sig_min
                reinf_sig_max = meshed_reinf_sig_max
            elif lumped_reinf_sig_min and lumped_reinf_sig_max:
                reinf_sig_min = lumped_reinf_sig_min
                reinf_sig_max = lumped_reinf_sig_max
            else:
                reinf_sig_min = 0
                reinf_sig_max = 0

            # set up ticks
            v_conc = np.linspace(conc_sig_min, conc_sig_max, 15, endpoint=True)
            v_reinf = np.linspace(reinf_sig_min, reinf_sig_max, 15, endpoint=True)

            if np.isclose(v_conc[0], v_conc[-1], atol=1e-12):
                v_conc = 15
                ticks_conc = None
            else:
                ticks_conc = v_conc

            if np.isclose(v_reinf[0], v_reinf[-1], atol=1e-12):
                ticks_reinf = None
                reinf_tick_same = True
            else:
                ticks_reinf = v_reinf
                reinf_tick_same = False

            # plot the concrete stresses
            for idx, sig in enumerate(self.concrete_stresses):
                # check region has a force
                if abs(self.concrete_forces[idx][0]) > 1e-8:
                    # create triangulation
                    triang_conc = tri.Triangulation(
                        self.concrete_analysis_sections[idx].mesh_nodes[:, 0],
                        self.concrete_analysis_sections[idx].mesh_nodes[:, 1],
                        self.concrete_analysis_sections[idx].mesh_elements[:, 0:3],
                    )

                    # plot the filled contour
                    trictr_conc = fig.axes[0].tricontourf(
                        triang_conc, sig, v_conc, cmap=cmap_conc, norm=CenteredNorm()
                    )

                    # plot a zero stress contour, supressing warning
                    with warnings.catch_warnings():
                        msg = "No contour levels were found within the data range."
                        warnings.filterwarnings("ignore", message=msg)

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

                        fig.axes[0].tricontour(
                            triang_conc,
                            sig,
                            [zero_level],
                            linewidths=1,
                            linestyles="dashed",
                        )

            # plot the meshed reinforcement stresses
            trictr_reinf = None

            for idx, sig in enumerate(self.meshed_reinforcement_stresses):
                # check region has a force
                if abs(self.meshed_reinforcement_forces[idx][0]) > 1e-8:
                    # create triangulation
                    triang_reinf = tri.Triangulation(
                        self.meshed_reinforcement_sections[idx].mesh_nodes[:, 0],
                        self.meshed_reinforcement_sections[idx].mesh_nodes[:, 1],
                        self.meshed_reinforcement_sections[idx].mesh_elements[:, 0:3],
                    )

                    # plot the filled contour
                    trictr_reinf = fig.axes[0].tricontourf(
                        triang_reinf, sig, v_reinf, cmap=cmap_reinf, norm=CenteredNorm()
                    )

                    # plot a zero stress contour, supressing warning
                    with warnings.catch_warnings():
                        msg = "No contour levels were found within the data range."
                        warnings.filterwarnings("ignore", message=msg)

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

                        fig.axes[0].tricontour(
                            triang_reinf,
                            sig,
                            [zero_level],
                            linewidths=1,
                            linestyles="dashed",
                        )

            # plot the lumped reinforcement stresses
            lumped_reinf_patches = []
            colours = []

            for idx, sig in enumerate(self.lumped_reinforcement_stresses):
                lumped_geom = self.lumped_reinforcement_geometries[idx].geom
                lumped_reinf_patches.append(
                    mpatches.Polygon(xy=list(lumped_geom.exterior.coords))
                )
                colours.append(sig)

            for idx, sig in enumerate(self.strand_stresses):
                lumped_reinf_patches.append(
                    mpatches.Polygon(
                        xy=list(self.strand_geometries[idx].geom.exterior.coords)
                    )
                )
                colours.append(sig)

            patch = PatchCollection(lumped_reinf_patches, cmap=cmap_reinf)
            patch.set_array(colours)
            if reinf_tick_same:
                patch.set_clim(vmin=0.99 * v_reinf[0], vmax=1.01 * v_reinf[-1])
            else:
                patch.set_clim(vmin=v_reinf[0], vmax=v_reinf[-1])
            fig.axes[0].add_collection(patch)

            # add the colour bars
            fig.colorbar(
                trictr_conc,
                label="Concrete Stress",
                format="%.2e",
                ticks=ticks_conc,
                cax=fig.axes[1],
            )

            if trictr_reinf:
                mappable = trictr_reinf
            else:
                mappable = patch

            fig.colorbar(
                mappable,
                label="Reinforcement Stress",
                format="%.2e",
                ticks=ticks_reinf,
                cax=fig.axes[2],
            )

        return ax

    def sum_forces(self) -> float:
        """Returns the sum of the internal forces.

        Returns:
            Sum of internal forces
        """
        force_sum = 0

        # sum concrete forces
        for conc_force in self.concrete_forces:
            force_sum += conc_force[0]

        # sum meshed reinf forces
        for meshed_reinf_force in self.meshed_reinforcement_forces:
            force_sum += meshed_reinf_force[0]

        # sum lumped reinf forces
        for lumped_reinf_force in self.lumped_reinforcement_forces:
            force_sum += lumped_reinf_force[0]

        # sum strand forces
        for strand_force in self.strand_forces:
            force_sum += strand_force[0]

        return force_sum

    def sum_moments(self) -> tuple[float, float, float]:
        """Returns the sum of the internal moments.

        Returns:
            Sum of internal moments about each axis and resultant moment (``m_x``,
            ``m_y``, ``m``)
        """
        moment_sum_x = 0
        moment_sum_y = 0

        # sum concrete moments
        for conc_force in self.concrete_forces:
            moment_sum_x += conc_force[0] * conc_force[2]
            moment_sum_y += conc_force[0] * conc_force[1]

        # sum meshed reinf moments
        for meshed_reinf_force in self.meshed_reinforcement_forces:
            moment_sum_x += meshed_reinf_force[0] * meshed_reinf_force[2]
            moment_sum_y += meshed_reinf_force[0] * meshed_reinf_force[1]

        # sum lumped reinf moments
        for lumped_reinf_force in self.lumped_reinforcement_forces:
            moment_sum_x += lumped_reinf_force[0] * lumped_reinf_force[2]
            moment_sum_y += lumped_reinf_force[0] * lumped_reinf_force[1]

        # sum strand moments
        for strand_force in self.strand_forces:
            moment_sum_x += strand_force[0] * strand_force[2]
            moment_sum_y += strand_force[0] * strand_force[1]

        moment_sum = np.sqrt(moment_sum_x * moment_sum_x + moment_sum_y * moment_sum_y)

        return moment_sum_x, moment_sum_y, moment_sum

    def get_concrete_stress_limits(self) -> tuple[float, float]:
        """Returns the minimum and maximum concrete stress.

        Returns:
            Minimum concrete stress, maximum concrete stress
        """
        min_stress = 0
        max_stress = 0

        for idx, stress_list in enumerate(self.concrete_stresses):
            if idx == 0:
                min_stress = stress_list.min()
                max_stress = stress_list.max()
            else:
                min_stress = min(min_stress, stress_list.min())
                max_stress = max(max_stress, stress_list.max())

        return min_stress, max_stress
