from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np
import triangle

from concreteproperties.utils import get_strain, gauss_points, shape_function
from concreteproperties.post import plotting_context

import sectionproperties.analysis.fea as sp_fea

if TYPE_CHECKING:
    from concreteproperties.material import Concrete
    from sectionproperties.pre.geometry import Geometry

from rich.pretty import pprint


class AnalysisSection:
    """Class for an analysis section to perform a fast analysis on concrete sections."""

    def __init__(
        self,
        geometry: Geometry,
        order: int = 1,
    ):
        """Inits the AnalysisSection class.

        :param geometry: Geometry object
        :type geometry: :class:`sectionproperties.pre.geometry.Geometry`
        :param int order: Mesh element order, linear (n = 1) or quadratic (n = 2)
        """

        self.geometry = geometry

        # create simple mesh
        tri = {}  # create tri dictionary
        tri["vertices"] = geometry.points  # set point
        tri["segments"] = geometry.facets  # set facets

        if geometry.holes:
            tri["holes"] = geometry.holes  # set holes

        if order == 1:
            self.mesh = triangle.triangulate(tri, "pq")
        elif order == 2:
            self.mesh = triangle.triangulate(tri, "po2")

        # extract mesh data
        self.mesh_nodes = np.array(self.mesh["vertices"], dtype=np.dtype(float))
        self.mesh_elements = np.array(self.mesh["triangles"], dtype=np.dtype(int))

        if order == 2:
            # swap mid-node order to retain node ordering consistency
            self.mesh_elements[:, [3, 4, 5]] = self.mesh_elements[:, [5, 3, 4]]

        # build elements
        self.elements = []

        for idx, node_ids in enumerate(self.mesh_elements):
            x1 = self.mesh_nodes[node_ids[0]][0]
            y1 = self.mesh_nodes[node_ids[0]][1]
            x2 = self.mesh_nodes[node_ids[1]][0]
            y2 = self.mesh_nodes[node_ids[1]][1]
            x3 = self.mesh_nodes[node_ids[2]][0]
            y3 = self.mesh_nodes[node_ids[2]][1]

            if order == 2:
                x4 = self.mesh_nodes[node_ids[3]][0]
                y4 = self.mesh_nodes[node_ids[3]][1]
                x5 = self.mesh_nodes[node_ids[4]][0]
                y5 = self.mesh_nodes[node_ids[4]][1]
                x6 = self.mesh_nodes[node_ids[5]][0]
                y6 = self.mesh_nodes[node_ids[5]][1]

            # create a list containing the vertex coordinates
            if order == 1:
                coords = np.array([[x1, x2, x3], [y1, y2, y3]])
            elif order == 2:
                coords = coords = np.array([[x1, x2, x3, x4, x5, x6], [y1, y2, y3, y4, y5, y6]])

            # add tri elements to the mesh
            if order == 1:
                new_el = Tri3(
                    el_id=idx,
                    coords=coords,
                    node_ids=node_ids,
                    conc_material=self.geometry.material,
                )
            elif order == 2:
                new_el = Tri6(
                    el_id=idx,
                    coords=coords,
                    node_ids=node_ids,
                    conc_material=self.geometry.material,
                )

            self.elements.append(new_el)
            self.order = order

    def ultimate_stress_analysis(
        self,
        point_na,
        d_n,
        theta,
        ultimate_strain,
        pc_local,
    ):
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
    ):
        """Plots the finite element mesh.

        :param float alpha: Transparency of the mesh outlines
        :param string title: Plot title
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes`
        """

        with plotting_context(title=title, **kwargs) as (fig, ax):
            # plot the mesh
            ax.triplot(
                self.mesh_nodes[:, 0],
                self.mesh_nodes[:, 1],
                self.mesh_elements[:, 0:3],
                lw=0.5,
                color="black",
                alpha=alpha,
            )

            ax.set_aspect("equal", anchor="C")

        return ax


@dataclass
class Tri3:
    """xxx

    xxx
    """

    el_id: int
    coords: np.ndarray
    node_ids: List[int]
    conc_material: Concrete

    def second_moments_of_area(
        self,
    ):
        """x

        x
        """

        # initialise properties
        e_ixx = 0
        e_iyy = 0
        e_ixy = 0

        # get points for 3 point Gaussian integration
        gps = gauss_points(n=3)

        # loop through each gauss point
        for gp in gps:
            # determine shape function and jacobian
            N, j = shape_function(coords=self.coords, gauss_point=gp)

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

    def calculate_ultimate_actions(
        self,
        point_na,
        d_n,
        theta,
        ultimate_strain,
        pc_local,
    ):
        """x

        x
        """

        # initialise element results
        area_e = 0
        qx_e = 0
        qy_e = 0
        force_e = 0

        # get points for 1 point Gaussian integration
        gps = gauss_points(n=1)

        # loop through each gauss point
        for gp in gps:
            # determine shape function and jacobian
            N, j = shape_function(coords=self.coords, gauss_point=gp)

            # get coordinates of the gauss point
            x = np.dot(N, np.transpose(self.coords[0, :]))
            y = np.dot(N, np.transpose(self.coords[1, :]))

            # calculate area properties
            area_e += gp[0] * j
            qx_e += gp[0] * y * j
            qy_e += gp[0] * x * j

            # get strain at gauss point
            strain = get_strain(
                point=(x, y),
                point_na=point_na,
                d_n=d_n,
                theta=theta,
                ultimate_strain=ultimate_strain,
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
        mv = force_e * (c_v - pc_local)

        return force_e, mv


@dataclass
class Tri6:
    """xxx

    xxx
    """

    el_id: int
    coords: np.ndarray
    node_ids: List[int]
    conc_material: Concrete

    def second_moments_of_area(
        self,
    ):
        """x

        x
        """

        # initialise properties
        e_ixx = 0
        e_iyy = 0
        e_ixy = 0

        # get points for 6 point Gaussian integration
        gps = sp_fea.gauss_points(n=6)

        # loop through each gauss point
        for gp in gps:
            # determine shape function and jacobian
            N, _, j = sp_fea.shape_function(coords=self.coords, gauss_point=gp)

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
