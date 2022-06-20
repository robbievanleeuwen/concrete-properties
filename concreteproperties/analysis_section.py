from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np
from matplotlib.colors import ListedColormap
import triangle

import concreteproperties.utils as utils
from concreteproperties.post import plotting_context

import sectionproperties.analysis.fea as sp_fea

if TYPE_CHECKING:
    import matplotlib.axes
    from concreteproperties.material import Concrete
    from sectionproperties.pre.geometry import Geometry

from rich.pretty import pprint


class AnalysisSection:
    """Class for an analysis section to perform a fast analysis on concrete sections."""

    def __init__(
        self,
        geometry: Geometry,
    ):
        """Inits the AnalysisSection class.

        :param geometry: Geometry object
        :type geometry: :class:`sectionproperties.pre.geometry.Geometry`
        """

        self.geometry = geometry

        # create simple mesh
        tri = {}  # create tri dictionary
        tri["vertices"] = geometry.points  # set point
        tri["segments"] = geometry.facets  # set facets

        if geometry.holes:
            tri["holes"] = geometry.holes  # set holes

        self.mesh = triangle.triangulate(tri, "p")

        # extract mesh data
        self.mesh_nodes = np.array(self.mesh["vertices"], dtype=np.dtype(float))
        self.mesh_elements = np.array(self.mesh["triangles"], dtype=np.dtype(int))

        # build elements
        self.elements = []

        for idx, node_ids in enumerate(self.mesh_elements):
            x1 = self.mesh_nodes[node_ids[0]][0]
            y1 = self.mesh_nodes[node_ids[0]][1]
            x2 = self.mesh_nodes[node_ids[1]][0]
            y2 = self.mesh_nodes[node_ids[1]][1]
            x3 = self.mesh_nodes[node_ids[2]][0]
            y3 = self.mesh_nodes[node_ids[2]][1]

            # create a list containing the vertex coordinates
            coords = np.array([[x1, x2, x3], [y1, y2, y3]])

            # add tri elements to the mesh
            self.elements.append(
                Tri3(
                    el_id=idx,
                    coords=coords,
                    node_ids=node_ids,
                    conc_material=self.geometry.material,
                )
            )

    def service_stress_analysis(
        self,
        point_na: Tuple[float],
        d_n: float,
        theta: float,
        kappa: float,
        na_local: float,
    ) -> Tuple[float]:
        """Performs an ultimate stress analysis on the section.

        :param point_na: Point on the neutral axis
        :type point_na: Tuple[float]
        :param float d_n: Depth of the neutral axis from the extreme compression fibre
        :param float theta: Angle the neutral axis makes with the horizontal axis
        :param float kappa: Curvature
        :param float na_local: y-location of the neutral axis in local coordinates

        :return: Axial force and resultant moment
        :rtype: Tuple[float]
        """

        # initialise section actions
        n = 0
        mv = 0

        for el in self.elements:
            el_n, el_mv = el.calculate_service_actions(
                point_na=point_na,
                d_n=d_n,
                theta=theta,
                kappa=kappa,
                na_local=na_local,
            )

            n += el_n
            mv += el_mv

        return n, mv

    def ultimate_stress_analysis(
        self,
        point_na: Tuple[float],
        d_n: float,
        theta: float,
        ultimate_strain: float,
        pc_local: float,
    ) -> Tuple[float]:
        """Performs an ultimate stress analysis on the section.

        :param point_na: Point on the neutral axis
        :type point_na: Tuple[float]
        :param float d_n: Depth of the neutral axis from the extreme compression fibre
        :param float theta: Angle the neutral axis makes with the horizontal axis
        :param float ultimate_strain: Strain at the extreme compression fibre
        :param float pc_local: y-location of the plastic centroid in local coordinates

        :return: Axial force and resultant moment
        :rtype: Tuple[float]
        """

        # initialise section actions
        n = 0
        mv = 0

        for el in self.elements:
            el_n, el_mv = el.calculate_ultimate_actions(
                point_na=point_na,
                d_n=d_n,
                theta=theta,
                ultimate_strain=ultimate_strain,
                pc_local=pc_local,
            )

            n += el_n
            mv += el_mv

        return n, mv

    def plot_mesh(
        self,
        alpha=0.5,
        title="Finite Element Mesh",
        **kwargs,
    ) -> matplotlib.axes._subplots.AxesSubplot:
        """Plots the finite element mesh.

        :param float alpha: Transparency of the mesh outlines
        :param string title: Plot title
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes._subplots.AxesSubplot`
        """

        with plotting_context(title=title, **kwargs) as (fig, ax):
            colour_array = []
            c = []  # Indices of elements for mapping colours

            # create an array of finite element colours
            for idx, element in enumerate(self.elements):
                colour_array.append(element.conc_material.colour)
                c.append(idx)

            cmap = ListedColormap(colour_array)  # custom colourmap

            # plot the mesh colours
            ax.tripcolour(
                self.mesh_nodes[:, 0],
                self.mesh_nodes[:, 1],
                self.mesh_elements[:, 0:3],
                c,
                cmap=cmap,
            )

            # plot the mesh
            ax.triplot(
                self.mesh_nodes[:, 0],
                self.mesh_nodes[:, 1],
                self.mesh_elements[:, 0:3],
                lw=0.5,
                colour="black",
                alpha=alpha,
            )

            ax.set_aspect("equal", anchor="C")

        return ax


@dataclass
class Tri3:
    """Class for a three noded linear triangular element.

    :param int el_id: Unique element id
    :param coords: A 2 x 3 array of the coordinates of the tri-3 nodes.
    :type coords: :class:`numpy.ndarray`
    :param node_ids: A list of the global node ids for the current element
    :type node_ids: List[int]
    :param conc_material: Material object for the current finite element.
    :type conc_material: :class:`~concreteproperties.material.Concrete`
    """

    el_id: int
    coords: np.ndarray
    node_ids: List[int]
    conc_material: Concrete

    def second_moments_of_area(
        self,
    ) -> Tuple[float]:
        """Calculates the second moments of area for the current finite element.

        :return: Modulus weighted second moments of area *(e_ixx, e_iyy, e_ixy)*
        :rtype: Tuple[float]
        """

        # initialise properties
        e_ixx = 0
        e_iyy = 0
        e_ixy = 0

        # get points for 3 point Gaussian integration
        gps = utils.gauss_points(n=3)

        # loop through each gauss point
        for gp in gps:
            # determine shape function and jacobian
            N, j = utils.shape_function(coords=self.coords, gauss_point=gp)

            e_ixx += (
                self.conc_material.elastic_modulus
                * gp[0]
                * np.dot(N, np.transpose(self.coords[1, :])) ** 2
                * j
            )
            e_iyy += (
                self.conc_material.elastic_modulus
                * gp[0]
                * np.dot(N, np.transpose(self.coords[0, :])) ** 2
                * j
            )
            e_ixy += (
                self.conc_material.elastic_modulus
                * gp[0]
                * np.dot(N, np.transpose(self.coords[1, :]))
                * np.dot(N, np.transpose(self.coords[0, :]))
                * j
            )

        return e_ixx, e_iyy, e_ixy

    def calculate_service_actions(
        self,
        point_na: Tuple[float],
        d_n: float,
        theta: float,
        kappa: float,
        na_local: float,
    ) -> Tuple[float]:
        """Calculates ultimate actions for the current finite element.

        :param point_na: Point on the neutral axis
        :type point_na: Tuple[float]
        :param float d_n: Depth of the neutral axis from the extreme compression fibre
        :param float theta: Angle the neutral axis makes with the horizontal axis
        :param float kappa: Curvature
        :param float na_local: y-location of the neutral axis in local coordinates

        :return: Axial force and resultant moment
        :rtype: Tuple[float]
        """

        # initialise element results
        area_e = 0
        qx_e = 0
        qy_e = 0
        force_e = 0

        # get points for 1 point Gaussian integration
        gps = utils.gauss_points(n=1)

        # loop through each gauss point
        for gp in gps:
            # determine shape function and jacobian
            N, j = utils.shape_function(coords=self.coords, gauss_point=gp)

            # get coordinates of the gauss point
            x = np.dot(N, np.transpose(self.coords[0, :]))
            y = np.dot(N, np.transpose(self.coords[1, :]))

            # calculate area properties
            area_e += gp[0] * j
            qx_e += gp[0] * y * j
            qy_e += gp[0] * x * j

            # get strain at gauss point
            strain = utils.get_service_strain(
                point=(x, y),
                point_na=point_na,
                theta=theta,
                kappa=kappa,
            )

            # get stress at gauss point
            stress = self.conc_material.stress_strain_profile.get_stress(strain=strain)

            # calculate force (stress * area)
            force_e += gp[0] * stress * j

        # calculate element centroid
        cx_e, cy_e = qy_e / area_e, qx_e / area_e

        # convert centroid to local coordinates
        _, c_v = sp_fea.principal_coordinate(phi=theta * 180 / np.pi, x=cx_e, y=cy_e)

        # calculate moment
        mv = force_e * (c_v - na_local)

        return force_e, mv

    def calculate_ultimate_actions(
        self,
        point_na: Tuple[float],
        d_n: float,
        theta: float,
        ultimate_strain: float,
        pc_local: float,
    ) -> Tuple[float]:
        """Calculates ultimate actions for the current finite element.

        :param point_na: Point on the neutral axis
        :type point_na: Tuple[float]
        :param float d_n: Depth of the neutral axis from the extreme compression fibre
        :param float theta: Angle the neutral axis makes with the horizontal axis
        :param float ultimate_strain: Strain at the extreme compression fibre
        :param float pc_local: y-location of the plastic centroid in local coordinates

        :return: Axial force and resultant moment
        :rtype: Tuple[float]
        """

        # initialise element results
        area_e = 0
        qx_e = 0
        qy_e = 0
        force_e = 0

        # get points for 1 point Gaussian integration
        gps = utils.gauss_points(n=1)

        # loop through each gauss point
        for gp in gps:
            # determine shape function and jacobian
            N, j = utils.shape_function(coords=self.coords, gauss_point=gp)

            # get coordinates of the gauss point
            x = np.dot(N, np.transpose(self.coords[0, :]))
            y = np.dot(N, np.transpose(self.coords[1, :]))

            # calculate area properties
            area_e += gp[0] * j
            qx_e += gp[0] * y * j
            qy_e += gp[0] * x * j

            # get strain at gauss point
            strain = utils.get_ultimate_strain(
                point=(x, y),
                point_na=point_na,
                d_n=d_n,
                theta=theta,
                ultimate_strain=ultimate_strain,
            )

            # get stress at gauss point
            stress = self.conc_material.ultimate_stress_strain_profile.get_stress(
                strain=strain
            )

            # calculate force (stress * area)
            force_e += gp[0] * stress * j

        # calculate element centroid
        cx_e, cy_e = qy_e / area_e, qx_e / area_e

        # convert centroid to local coordinates
        _, c_v = sp_fea.principal_coordinate(phi=theta * 180 / np.pi, x=cx_e, y=cy_e)

        # calculate moment
        mv = force_e * (c_v - pc_local)

        return force_e, mv
