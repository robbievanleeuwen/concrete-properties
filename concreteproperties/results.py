from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, TYPE_CHECKING
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.tri as tri
import matplotlib.cm as cm
from matplotlib.colors import CenteredNorm
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits import mplot3d
from scipy.interpolate import interp1d
from rich.console import Console
from rich.table import Table
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from concreteproperties.post import plotting_context
from sectionproperties.pre.geometry import CompoundGeometry

if TYPE_CHECKING:
    import matplotlib
    from concreteproperties.concrete_section import ConcreteSection
    from concreteproperties.analysis_section import AnalysisSection
    from sectionproperties.pre.geometry import Geometry


@dataclass
class ConcreteProperties:
    """Class for storing gross concrete section properties.

    All properties with an `e_` preceding the property are multiplied by the elastic
    modulus. In order to obtain transformed properties, call the
    :meth:`~concreteproperties.concrete_section.ConcreteSection.get_transformed_gross_properties`
    method.
    """

    # section areas
    total_area: float = 0
    concrete_area: float = 0
    steel_area: float = 0
    e_a: float = 0

    # section mass
    mass: float = 0

    # section perimeter
    perimeter: float = 0

    # first moments of area
    e_qx: float = 0
    e_qy: float = 0

    # centroids
    cx: float = 0
    cy: float = 0

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

    # plastic properties
    squash_load: float = 0
    tensile_load: float = 0
    axial_pc_x: float = 0
    axial_pc_y: float = 0
    conc_ultimate_strain: float = 0

    def print_results(
        self,
        fmt: Optional[str] = "8.6e",
    ):
        """Prints the gross concrete section properties to the terminal.

        :param fmt: Number format
        :type fmt: Optional[str]
        """

        table = Table(title="Gross Concrete Section Properties")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        table.add_row("Total Area", "{:>{fmt}}".format(self.total_area, fmt=fmt))
        table.add_row("Concrete Area", "{:>{fmt}}".format(self.concrete_area, fmt=fmt))
        table.add_row("Steel Area", "{:>{fmt}}".format(self.steel_area, fmt=fmt))
        table.add_row("Axial Rigidity (EA)", "{:>{fmt}}".format(self.e_a, fmt=fmt))
        table.add_row("Mass (per unit length)", "{:>{fmt}}".format(self.mass, fmt=fmt))
        table.add_row("Perimeter", "{:>{fmt}}".format(self.perimeter, fmt=fmt))
        table.add_row("E.Qx", "{:>{fmt}}".format(self.e_qx, fmt=fmt))
        table.add_row("E.Qy", "{:>{fmt}}".format(self.e_qy, fmt=fmt))
        table.add_row("x-Centroid", "{:>{fmt}}".format(self.cx, fmt=fmt))
        table.add_row("y-Centroid", "{:>{fmt}}".format(self.cy, fmt=fmt))
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
        table.add_row("Squash Load", "{:>{fmt}}".format(self.squash_load, fmt=fmt))
        table.add_row("Tensile Load", "{:>{fmt}}".format(self.tensile_load, fmt=fmt))
        table.add_row(
            "x-Axial Plastic Centroid", "{:>{fmt}}".format(self.axial_pc_x, fmt=fmt)
        )
        table.add_row(
            "y-Axial Plastic Centroid", "{:>{fmt}}".format(self.axial_pc_y, fmt=fmt)
        )
        table.add_row(
            "Ultimate Concrete Strain",
            "{:>{fmt}}".format(self.conc_ultimate_strain, fmt=fmt),
        )

        console = Console()
        console.print(table)


@dataclass
class TransformedConcreteProperties:
    """Class for storing transformed gross concrete section properties.

    :param concrete_properties: Concrete properties object
    :type concrete_properties:
        :class:`~concreteproperties.concrete_section.ConcreteProperties`
    :param float elastic_modulus: Reference elastic modulus
    """

    concrete_properties: ConcreteProperties = field(repr=False)
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

    def __post_init__(
        self,
    ):
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
        fmt: Optional[str] = "8.6e",
    ):
        """Prints the transformed gross concrete section properties to the terminal.

        :param fmt: Number format
        :type fmt: Optional[str]
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

    All properties with an `e_` preceding the property are multiplied by the elastic
    modulus. In order to obtain transformed properties, call the
    :meth:`~concreteproperties.results.CrackedResults.calculate_transformed_properties`
    method.

    :param float theta: Angle (in radians) the neutral axis makes with the horizontal
        axis (:math:`-\pi \leq \theta \leq \pi`)
    """

    theta: float
    m_cr: float = 0
    d_nc: float = 0
    cracked_geometries: List[Geometry] = field(default_factory=list, repr=False)
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
    elastic_modulus_ref: float = None
    a_cr: float = None
    qx_cr: float = None
    qy_cr: float = None
    ixx_g_cr: float = None
    iyy_g_cr: float = None
    ixy_g_cr: float = None
    ixx_c_cr: float = None
    iyy_c_cr: float = None
    ixy_c_cr: float = None
    iuu_cr: float = None
    i11_cr: float = None
    i22_cr: float = None

    def calculate_transformed_properties(
        self,
        elastic_modulus: float,
    ):
        """Calculates and stores transformed cracked properties using a reference
        elastic modulus.

        :param float elastic_modulus: Reference elastic modulus
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
        title: Optional[str] = "Cracked Geometries",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots the geometries that remain (are in compression or are steel) after a
        cracked analysis.

        :param title: Plot title
        :type title: Optional[str]
        :param kwargs: Passed to
            :meth:`~sectionproperties.pre.geometry.CompoundGeometry.plot_geometry`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes.Axes`
        """

        return CompoundGeometry(self.cracked_geometries).plot_geometry(
            title=title, **kwargs
        )

    def print_results(
        self,
        fmt: Optional[str] = "8.6e",
    ):
        """Prints the cracked concrete section properties to the terminal.

        :param fmt: Number format
        :type fmt: Optional[str]
        """

        table = Table(title="Cracked Concrete Section Properties")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        table.add_row("theta", "{:>{fmt}}".format(self.theta, fmt=fmt))

        if self.elastic_modulus_ref:
            table.add_row(
                "E_ref", "{:>{fmt}}".format(self.elastic_modulus_ref, fmt=fmt)
            )

        table.add_row("M_cr", "{:>{fmt}}".format(self.m_cr, fmt=fmt))
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

    :param float theta: Angle (in radians) the neutral axis makes with the horizontal
        axis (:math:`-\pi \leq \theta \leq \pi`)
    :var kappa: List of curvatures
    :vartype kappa: List[float]
    :var moment: List of bending moments
    :vartype moment: List[float]
    :var failure_geometry: Geometry object of the region of the cross-section that
        failed, ending the moment curvature analysis
    :vartype failure_geometry: :class:`sectionproperties.pre.geometry.Geometry`
    """

    # results
    theta: float
    kappa: List[float] = field(default_factory=list)
    moment: List[float] = field(default_factory=list)
    failure_geometry: Geometry = field(init=False)

    # for analysis
    _n_i: float = field(default=0, repr=False)
    _m_i: float = field(default=0, repr=False)
    _failure: bool = field(default=False, repr=False)

    def __post_init__(
        self,
    ):
        self.kappa.append(0)
        self.moment.append(0)

    def plot_results(
        self,
        m_scale: Optional[float] = 1e-6,
        fmt: Optional[str] = "o-",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots the moment curvature results.

        :param m_scale: Scaling factor to apply to bending moment
        :type m_scale: Optional[float]
        :param fmt: Plot format string
        :type fmt: Optional[str]
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes.Axes`
        """

        # scale moments
        moments = np.array(self.moment) * m_scale

        # create plot and setup the plot
        with plotting_context(title="Moment-Curvature", **kwargs) as (
            fig,
            ax,
        ):
            ax.plot(self.kappa, moments, fmt)
            plt.xlabel("Curvature")
            plt.ylabel("Moment")
            plt.grid(True)

        return ax

    @staticmethod
    def plot_multiple_results(
        moment_curvature_results: List[MomentCurvatureResults],
        labels: List[str],
        m_scale: Optional[float] = 1e-6,
        fmt: Optional[str] = "o-",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots multiple moment curvature results.

        :param moment_curvature_results: List of moment curvature results objects
        :type moment_interaction_results:
            List[:class:`~concreteproperties.results.MomentCurvatureResults`]
        :param labels: List of labels for each moment curvature diagram
        :type labels: List[str]
        :param float m_scale: Scaling factor to apply to bending moment
        :type m_scale: Optional[float]
        :param fmt: Plot format string
        :type fmt: Optional[str]
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes.Axes`
        """

        # create plot and setup the plot
        with plotting_context(title="Moment-Curvature", **kwargs) as (
            fig,
            ax,
        ):
            # for each M-k curve
            for idx, mk_result in enumerate(moment_curvature_results):
                # scale results
                kappas = np.array(mk_result.kappa)
                moments = np.array(mk_result.moment) * m_scale

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
        title: Optional[str] = "Failure Geometry",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots the geometry that fails in the moment curvature analysis.

        :param title: Plot title
        :type title: Optional[str]
        :param kwargs: Passed to
            :meth:`~sectionproperties.pre.geometry.CompoundGeometry.plot_geometry`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes.Axes`
        """

        return self.failure_geometry.plot_geometry(title=title, **kwargs)

    def get_curvature(
        self,
        moment: float,
    ) -> float:
        """Given a moment, uses the moment-curvature results to interpolate a curvature.

        :param float moment: Bending moment at which to obtain curvature

        :raises ValueError: If supplied moment is outside bounds of moment-curvature
            results.

        :return: Curvature
        :rtype: float
        """

        # check moment is within bounds of results
        m_min = min(self.moment)
        m_max = max(self.moment)

        if moment > m_max or moment < m_min:
            raise ValueError(
                "moment must be within the bounds of the moment-curvature results."
            )

        f_kappa = interp1d(
            x=self.moment,
            y=self.kappa,
            kind="linear",
        )

        return float(f_kappa(moment))


@dataclass
class UltimateBendingResults:
    r"""Class for storing ultimate bending results.

    :param float theta: Angle (in radians) the neutral axis makes with the horizontal
        axis (:math:`-\pi \leq \theta \leq \pi`)
    :var float d_n: Ultimate neutral axis depth
    :var float k_u: Neutral axis parameter *(d_n / d)*
    :var float n: Resultant axial force
    :var float m_x: Resultant bending moment about the x-axis
    :var float m_y: Resultant bending moment about the y-axis
    :var float m_u: Resultant bending moment about the u-axis
    """

    # bending angle
    theta: float

    # ultimate neutral axis depth
    d_n: float = None
    k_u: float = None

    # resultant actions
    n: float = None
    m_x: float = None
    m_y: float = None
    m_u: float = None

    def print_results(
        self,
        fmt: Optional[str] = "8.6e",
    ):
        """Prints the ultimate bending results to the terminal.

        :param fmt: Number format
        :type fmt: Optional[str]
        """

        table = Table(title="Ultimate Bending Results")
        table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        table.add_row("Bending Angle - theta", "{:>{fmt}}".format(self.theta, fmt=fmt))
        table.add_row("Neutral Axis Depth - d_n", "{:>{fmt}}".format(self.d_n, fmt=fmt))
        table.add_row(
            "Neutral Axis Parameter- k_u", "{:>{fmt}}".format(self.k_u, fmt=fmt)
        )
        table.add_row("Axial Force", "{:>{fmt}}".format(self.n, fmt=fmt))
        table.add_row("Bending Capacity - m_x", "{:>{fmt}}".format(self.m_x, fmt=fmt))
        table.add_row("Bending Capacity - m_y", "{:>{fmt}}".format(self.m_y, fmt=fmt))
        table.add_row("Bending Capacity - m_u", "{:>{fmt}}".format(self.m_u, fmt=fmt))

        console = Console()
        console.print(table)


@dataclass
class MomentInteractionResults:
    """Class for storing moment interaction results.

    :var results: List of ultimate bending result objects
    :vartype results: List[:class:`~concreteproperties.results.UltimateBendingResults`]
    :var results_neg: List of ultimate bending result objects (for negative bending)
    :vartype results: List[:class:`~concreteproperties.results.UltimateBendingResults`]
    """

    results: List[UltimateBendingResults] = field(default_factory=list)
    results_neg: List[UltimateBendingResults] = field(default_factory=list)

    def get_results_lists(
        self,
        neg=False,
    ) -> Tuple[List[float]]:
        """Returns a list of axial forces and moments.

        :param bool neg: If True, gets the negative bending results

        :return: List of axial forces and moments *(n, m)*
        :rtype: Tuple[List[float]]
        """

        # build list of results
        n_list = []
        m_list = []

        if neg:
            results_list = self.results_neg
        else:
            results_list = self.results

        for result in results_list:
            n_list.append(result.n)
            m_list.append(result.m_u)

        return n_list, m_list

    def plot_diagram(
        self,
        n_scale: Optional[float] = 1e-3,
        m_scale: Optional[float] = 1e-6,
        fmt: Optional[str] = "o-",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots a moment interaction diagram.

        :param n_scale: Scaling factor to apply to axial force
        :type n_scale: Optional[float]
        :param n_scale: Scaling factor to apply to axial force
        :type m_scale: Optional[float]
        :param fmt: Plot format string
        :type fmt: Optional[str]
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes.Axes`
        """

        # create plot and setup the plot
        with plotting_context(title="Moment Interaction Diagram", **kwargs) as (
            fig,
            ax,
        ):
            # get results
            n_list, m_list = self.get_results_lists()

            # scale results
            forces = np.array(n_list) * n_scale
            moments = np.array(m_list) * m_scale

            # if negative results
            if len(self.results_neg) > 0:
                # get results
                n_list, m_list = self.get_results_lists(neg=True)

                # scale results
                forces = np.hstack((forces, np.flip(np.array(n_list) * n_scale)))
                moments = np.hstack((moments, np.flip(np.array(m_list) * m_scale)))

            ax.plot(moments, forces, fmt)

            plt.xlabel("Bending Moment")
            plt.ylabel("Axial Force")
            plt.grid(True)

        return ax

    @staticmethod
    def plot_multiple_diagrams(
        moment_interaction_results: List[MomentInteractionResults],
        labels: List[str],
        n_scale: Optional[float] = 1e-3,
        m_scale: Optional[float] = 1e-6,
        fmt: Optional[str] = "o-",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots multiple moment interaction diagrams.

        :param moment_interaction_results: List of moment interaction results objects
        :type moment_interaction_results:
            List[:class:`~concreteproperties.results.MomentInteractionResults`]
        :param labels: List of labels for each moment interaction diagram
        :type labels: List[str]
        :param float n_scale: Scaling factor to apply to axial force
        :type n_scale: Optional[float]
        :param float m_scale: Scaling factor to apply to bending moment
        :type m_scale: Optional[float]
        :param fmt: Plot format string
        :type fmt: Optional[str]
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes.Axes`
        """

        # create plot and setup the plot
        with plotting_context(title="Moment Interaction Diagram", **kwargs) as (
            fig,
            ax,
        ):
            # for each M-N curve
            for idx, mi_result in enumerate(moment_interaction_results):
                n_list, m_list = mi_result.get_results_lists()

                # scale results
                forces = np.array(n_list) * n_scale
                moments = np.array(m_list) * m_scale

                # if negative results
                if len(mi_result.results_neg) > 0:
                    # get results
                    n_list, m_list = mi_result.get_results_lists(neg=True)

                    # scale results
                    forces = np.hstack((forces, np.flip(np.array(n_list) * n_scale)))
                    moments = np.hstack((moments, np.flip(np.array(m_list) * m_scale)))

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
    ) -> bool:
        """Determines whether or not the combination of axial force and moment lies
        within the moment interaction diagram.

        :param float n: Axial force
        :param float m: Bending moment

        :returns: True, if combination of axial force and moment is within the diagram
        :rtype: bool
        """

        # create a polygon from points on diagram
        poly_points = []

        for ult_res in self.results:
            poly_points.append((ult_res.m_u, ult_res.n))

        for ult_res in self.results_neg:
            poly_points.append((ult_res.m_u, ult_res.n))

        poly = Polygon(poly_points)
        point = Point(m, n)

        return poly.contains(point)


@dataclass
class BiaxialBendingResults:
    """Class for storing biaxial bending results.

    :param float n: Net axial force
    :var results: List of ultimate bending result objects
    :vartype results: List[:class:`~concreteproperties.results.UltimateBendingResults`]
    """

    n: float
    results: List[UltimateBendingResults] = field(default_factory=list)

    def get_results_lists(
        self,
    ) -> Tuple[List[float]]:
        """Returns a list and moments about the ``x`` and ``y`` axes.

        :return: List of axial forces and moments *(mx, my)*
        :rtype: Tuple[List[float]]
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
        m_scale: Optional[float] = 1e-6,
        fmt: Optional[str] = "o-",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots a biaxial bending diagram.

        :param m_scale: Scaling factor to apply to bending moment
        :type m_scale: Optional[float]
        :param fmt: Plot format string
        :type fmt: Optional[str]
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes.Axes`
        """

        m_x_list, m_y_list = self.get_results_lists()

        # create plot and setup the plot
        with plotting_context(
            title=f"Biaxial Bending Diagram, $N = {self.n:.3e}$", **kwargs
        ) as (
            fig,
            ax,
        ):
            # scale results
            m_x = np.array(m_x_list) * m_scale
            m_y = np.array(m_y_list) * m_scale

            ax.plot(m_x, m_y, fmt)

            plt.xlabel("Bending Moment $M_x$")
            plt.ylabel("Bending Moment $M_y$")
            plt.grid(True)

        return ax

    @staticmethod
    def plot_multiple_diagrams(
        biaxial_bending_results: List[BiaxialBendingResults],
        n_scale: Optional[float] = 1e-3,
        m_scale: Optional[float] = 1e-6,
        fmt: Optional[str] = "-",
    ) -> matplotlib.axes.Axes:
        """Plots multiple biaxial bending diagrams in a 3D plot.

        :param biaxial_bending_results: List of biaxial bending results objects
        :type biaxial_bending_results:
            List[:class:`~concreteproperties.results.BiaxialBendingResults`]
        :param float n_scale: Scaling factor to apply to axial force
        :type n_scale: Optional[float]
        :param float m_scale: Scaling factor to apply to bending moment
        :type m_scale: Optional[float]
        :param fmt: Plot format string
        :type fmt: Optional[str]

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes.Axes`
        """

        # make 3d plot
        fig = plt.figure()
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
        """Determines whether or not the combination of bending moments lies within the
        biaxial bending diagram.

        :param float m_x: Bending moment about the x-axis
        :param float m_y: Bending moment about the y-axis

        :returns: True, if combination of bendings moments is within the diagram
        :rtype: bool
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

    For service and ultimate stress analyses, the lever arm is stored in the ``d_x``
    variable and is the perpendicular distance to the neutral axis.

    :var concrete_analysis_sections: List of concrete analysis section objects
        present in the stress analysis, which can be visualised by calling the
        :meth:`~concreteproperties.analysis_section.AnalysisSection.plot_mesh` or
        :meth:`~concreteproperties.analysis_section.AnalysisSection.plot_shape`
    :vartype concrete_analysis_sections:
        List[:class:`~concreteproperties.analysis_section.AnalysisSection`]
    :var concrete_stresses: List of concrete stresses at the nodes of each concrete
        analysis section
    :vartype concrete_stresses: List[:class:`numpy.ndarray`]
    :var concrete_forces: List of net forces for each concrete analysis section and its
        lever arm to the neutral axis (``force``, ``d_x``, ``d_y``)
    :vartype concrete_forces: List[Tuple[float]]
    :var steel_geometries: List of steel geometry objects present in the stress analysis
    :vartype steel_geometries: List[:class:`sectionproperties.pre.geometry.Geometry`]
    :var steel_stresses: List of steel stresses for each steel geometry
    :vartype steel_stresses: List[float]
    :var steel_strains: List of steel strains for each steel geometry
    :vartype steel_strains: List[float]
    :var steel_forces: List of net forces for each steel geometry and its lever arm to
        the neutral axis (``force``, ``d_x``, ``d_y``)
    :vartype steel_forces: List[Tuple[float]]
    """

    concrete_section: ConcreteSection
    concrete_analysis_sections: List[AnalysisSection]
    concrete_stresses: List[np.ndarray]
    concrete_forces: List[Tuple[float]]
    steel_geometries: List[Geometry]
    steel_stresses: List[float]
    steel_strains: List[float]
    steel_forces: List[Tuple[float]]

    def plot_stress(
        self,
        title: Optional[str] = "Stress",
        conc_cmap: Optional[str] = "RdGy",
        steel_cmap: Optional[str] = "bwr",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots concrete and steel stresses on a concrete section.

        :param title: Plot title
        :type title: Optional[str]
        :param conc_cmap: Colour map for the concrete stress
        :type conc_cmap: Optional[str]
        :param steel_cmap: Colour map for the steel stress
        :type steel_cmap: Optional[str]
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes.Axes`
        """

        with plotting_context(
            title=title,
            **dict(
                kwargs, nrows=1, ncols=3, gridspec_kw={"width_ratios": [1, 0.08, 0.08]}
            ),
        ) as (fig, ax):
            # plot background
            self.concrete_section.plot_section(
                background=True, **dict(kwargs, ax=fig.axes[0])
            )

            # set up the colormaps
            cmap_conc = cm.get_cmap(name=conc_cmap)
            cmap_steel = cm.get_cmap(name=steel_cmap)

            # determine minimum and maximum stress values for the contour list
            # add tolerance for plotting stress blocks
            conc_sig_min = min([min(x) for x in self.concrete_stresses]) - 1e-12
            conc_sig_max = max([max(x) for x in self.concrete_stresses]) + 1e-12
            steel_sig_min = min(self.steel_stresses)
            steel_sig_max = max(self.steel_stresses)

            # set up ticks
            v_conc = np.linspace(conc_sig_min, conc_sig_max, 15, endpoint=True)
            v_steel = np.linspace(steel_sig_min, steel_sig_max, 15, endpoint=True)

            if np.isclose(v_conc[0], v_conc[-1], atol=1e-12):
                v_conc = 15
                ticks_conc = None
            else:
                ticks_conc = v_conc

            if np.isclose(v_steel[0], v_steel[-1], atol=1e-12):
                ticks_steel = None
                steel_tick_same = True
            else:
                ticks_steel = v_steel
                steel_tick_same = False

            # plot the concrete stresses
            for idx, sig in enumerate(self.concrete_stresses):
                # check region has a force
                if abs(self.concrete_forces[idx][0]) > 1e-8:
                    # create triangulation
                    triang = tri.Triangulation(
                        self.concrete_analysis_sections[idx].mesh_nodes[:, 0],
                        self.concrete_analysis_sections[idx].mesh_nodes[:, 1],
                        self.concrete_analysis_sections[idx].mesh_elements[:, 0:3],
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

            for idx, sig in enumerate(self.steel_stresses):
                steel_patches.append(
                    mpatches.Polygon(
                        xy=list(
                            self.concrete_section.steel_geometries[
                                idx
                            ].geom.exterior.coords
                        )
                    )
                )
                colours.append(sig)

            patch = PatchCollection(steel_patches, cmap=cmap_steel)
            patch.set_array(colours)
            if steel_tick_same:
                patch.set_clim([0.99 * v_steel[0], 1.01 * v_steel[-1]])
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

    def sum_forces(
        self,
    ) -> float:
        """Returns the sum of the internal forces.

        :return: Sum of internal forces
        :rtype: float
        """

        force_sum = 0

        # sum concrete forces
        for conc_force in self.concrete_forces:
            force_sum += conc_force[0]

        # sum steel forces
        for steel_force in self.steel_forces:
            force_sum += steel_force[0]

        return force_sum

    def sum_moments(
        self,
    ) -> Tuple[float]:
        """Returns the sum of the internal moments.

        :return: Sum of internal moments about each axis and resultant moment
            (``m_x``, ``m_y``, ``m``)
        :rtype: Tuple[float]
        """

        moment_sum_x = 0
        moment_sum_y = 0

        # sum concrete forces
        for conc_force in self.concrete_forces:
            moment_sum_x += conc_force[0] * conc_force[2]
            moment_sum_y += conc_force[0] * conc_force[1]

        # sum steel forces
        for steel_force in self.steel_forces:
            moment_sum_x += steel_force[0] * steel_force[2]
            moment_sum_y += steel_force[0] * steel_force[1]

        moment_sum = np.sqrt(moment_sum_x * moment_sum_x + moment_sum_y * moment_sum_y)

        return moment_sum_x, moment_sum_y, moment_sum
