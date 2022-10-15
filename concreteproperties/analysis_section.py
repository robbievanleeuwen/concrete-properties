from __future__ import annotations

from dataclasses import dataclass
from math import isinf
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import triangle
from matplotlib.colors import ListedColormap

import concreteproperties.utils as utils
from concreteproperties.material import Concrete
from concreteproperties.post import plotting_context

if TYPE_CHECKING:
    import matplotlib

    from concreteproperties.material import Material
    from concreteproperties.pre import CPGeom


class AnalysisSection:
    """Class for an analysis section to perform a fast analysis on meshed sections."""

    def __init__(
        self,
        geometry: CPGeom,
    ):
        """Inits the AnalysisSection class.

        :param geometry: Geometry object
        """

        self.geometry = geometry
        self.material = geometry.material

        # create simple mesh
        tri = {}  # create tri dictionary
        tri["vertices"] = geometry.points  # set point
        tri["segments"] = geometry.facets  # set facets

        if geometry.holes:
            tri["holes"] = geometry.holes  # set holes

        # coarse mesh
        self.mesh = triangle.triangulate(tri, "p")

        # extract mesh data
        self.mesh_nodes = np.array(self.mesh["vertices"], dtype=np.dtype(float))
        try:
            self.mesh_elements = np.array(self.mesh["triangles"], dtype=np.dtype(int))
        except KeyError:
            # if there are no triangles
            self.mesh_elements = []

        # build elements
        self.elements: List[Tri3] = []

        for node_ids in self.mesh_elements:
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
                    coords=coords,
                    node_ids=node_ids,
                    material=self.material,
                )
            )

    def calculate_meshed_area(self) -> float:
        """Calculates the area of the analysis section based on the generated mesh.

        :return: Meshed area (un-weighted by elastic modulus)
        """

        area = 0

        for el in self.elements:
            area += el.calculate_area()

        return area

    def get_elastic_stress(
        self,
        n: float,
        m_x: float,
        m_y: float,
        e_a: float,
        cx: float,
        cy: float,
        e_ixx: float,
        e_iyy: float,
        e_ixy: float,
    ) -> Tuple[np.ndarray, float, float, float]:
        r"""Given section actions and section propreties, calculates elastic stresses.

        :param n: Axial force
        :param m_x: Bending moment about the x-axis
        :param m_y: Bending moment about the y-axis
        :param e_a: Axial rigidity
        :param cx: x-Centroid
        :param cy: y-Centroid
        :param e_ixx: Flexural rigidity about the x-axis
        :param e_iyy: Flexural rigidity about the y-axis
        :param e_ixy: Flexural rigidity about the xy-axis

        :return: Elastic stresses, net force and distance from neutral axis to point of
            force action
        """

        # intialise stress results
        sig = np.zeros(len(self.mesh_nodes))

        # loop through nodes
        for idx, node in enumerate(self.mesh_nodes):
            x = node[0] - cx
            y = node[1] - cy

            # axial stress
            sig[idx] += n * self.material.elastic_modulus / e_a

            # bending moment
            sig[idx] += self.material.elastic_modulus * (
                -(e_ixy * m_x) / (e_ixx * e_iyy - e_ixy**2) * x
                + (e_iyy * m_x) / (e_ixx * e_iyy - e_ixy**2) * y
            )
            sig[idx] += self.material.elastic_modulus * (
                +(e_ixx * m_y) / (e_ixx * e_iyy - e_ixy**2) * x
                - (e_ixy * m_y) / (e_ixx * e_iyy - e_ixy**2) * y
            )

        # initialise section actions
        n_sec = 0
        m_x_sec = 0
        m_y_sec = 0

        for el in self.elements:
            el_n, el_m_x, el_m_y = el.calculate_elastic_actions(
                n=n,
                m_x=m_x,
                m_y=m_y,
                e_a=e_a,
                cx=cx,
                cy=cy,
                e_ixx=e_ixx,
                e_iyy=e_iyy,
                e_ixy=e_ixy,
            )

            n_sec += el_n
            m_x_sec += el_m_x
            m_y_sec += el_m_y

        # calculate point of action
        if n_sec == 0:
            d_x = 0
            d_y = 0
        else:
            d_x = m_y_sec / n_sec
            d_y = m_x_sec / n_sec

        return sig, n_sec, d_x, d_y

    def service_analysis(
        self,
        point_na: Tuple[float, float],
        theta: float,
        kappa: float,
        centroid: Tuple[float, float],
    ) -> Tuple[float, float, float, float, float]:
        r"""Performs a service stress analysis on the section.

        :param point_na: Point on the neutral axis
        :param d_n: Depth of the neutral axis from the extreme compression fibre
        :param theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)
        :param kappa: Curvature
        :param centroid: Centroid about which to take moments

        :return: Axial force, section moments and min/max strain
        """

        # initialise section actions
        n_sec = 0
        m_x_sec = 0
        m_y_sec = 0
        min_strain = 0
        max_strain = 0

        for el in self.elements:
            (
                el_n,
                el_m_x,
                el_m_y,
                el_min_strain,
                el_max_strain,
            ) = el.calculate_service_actions(
                point_na=point_na,
                theta=theta,
                kappa=kappa,
                centroid=centroid,
            )
            min_strain = min(min_strain, el_min_strain)
            max_strain = max(max_strain, el_max_strain)

            n_sec += el_n
            m_x_sec += el_m_x
            m_y_sec += el_m_y

        return n_sec, m_x_sec, m_y_sec, min_strain, max_strain

    def get_service_stress(
        self,
        d_n: float,
        kappa: float,
        point_na: Tuple[float, float],
        theta: float,
        centroid: Tuple[float, float],
    ) -> Tuple[np.ndarray, float, float, float]:
        r"""Given the neutral axis depth `d_n` and curvature `kappa` determines the
        service stresses within the section.

        :param d_n: Neutral axis depth
        :param kappa: Curvature
        :param point_na: Point on the neutral axis
        :param theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)
        :param centroid: Centroid about which to take moments

        :return: Service stresses, net force and distance from centroid to point of
            force action
        """

        # intialise stress results
        sig = np.zeros(len(self.mesh_nodes))

        # loop through nodes and calculate stress at nodes
        for idx, node in enumerate(self.mesh_nodes):
            # get strain at node
            strain = utils.get_service_strain(
                point=(node[0], node[1]),
                point_na=point_na,
                theta=theta,
                kappa=kappa,
            )

            # get stress at node
            sig[idx] = self.material.stress_strain_profile.get_stress(strain=strain)

        # calculate total force
        n_sec, m_x_sec, m_y_sec, _, _ = self.service_analysis(
            point_na=point_na,
            theta=theta,
            kappa=kappa,
            centroid=centroid,
        )

        # calculate point of action
        if n_sec == 0:
            d_x = 0
            d_y = 0
        else:
            d_x = m_y_sec / n_sec
            d_y = m_x_sec / n_sec

        return sig, n_sec, d_x, d_y

    def ultimate_analysis(
        self,
        point_na: Tuple[float, float],
        d_n: float,
        theta: float,
        ultimate_strain: float,
        centroid: Tuple[float, float],
    ) -> Tuple[float, float, float]:
        r"""Performs an ultimate stress analysis on the section.

        :param point_na: Point on the neutral axis
        :param d_n: Depth of the neutral axis from the extreme compression fibre
        :param theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)
        :param ultimate_strain: Concrete strain at failure
        :param centroid: Centroid about which to take moments

        :return: Axial force and resultant moments about the global axes
        """

        # initialise section actions
        n_sec = 0
        m_x_sec = 0
        m_y_sec = 0

        for el in self.elements:
            el_n, el_m_x, el_m_y = el.calculate_ultimate_actions(
                point_na=point_na,
                d_n=d_n,
                theta=theta,
                ultimate_strain=ultimate_strain,
                centroid=centroid,
            )

            n_sec += el_n
            m_x_sec += el_m_x
            m_y_sec += el_m_y

        return n_sec, m_x_sec, m_y_sec

    def get_ultimate_stress(
        self,
        d_n: float,
        point_na: Tuple[float, float],
        theta: float,
        ultimate_strain: float,
        centroid: Tuple[float, float],
    ) -> Tuple[np.ndarray, float, float, float]:
        r"""Given the neutral axis depth `d_n` and ultimate strain, determines the
        ultimate stresses with the section.

        :param d_n: Neutral axis depth
        :param point_na: Point on the neutral axis
        :param theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)
        :param ultimate_strain: Concrete strain at failure
        :param centroid: Centroid about which to take moments

        :return: Ultimate stresses net force and distance from neutral axis to point of
            force action
        """

        # intialise stress results
        sig = np.zeros(len(self.mesh_nodes))

        # loop through nodes
        for idx, node in enumerate(self.mesh_nodes):
            # get strain at node
            if isinf(d_n):
                strain = ultimate_strain
            else:
                strain = utils.get_ultimate_strain(
                    point=(node[0], node[1]),
                    point_na=point_na,
                    d_n=d_n,
                    theta=theta,
                    ultimate_strain=ultimate_strain,
                )

            # get stress at node
            if isinstance(self.material, Concrete):
                sig[idx] = self.material.ultimate_stress_strain_profile.get_stress(
                    strain=strain
                )
            else:
                sig[idx] = self.material.stress_strain_profile.get_stress(strain=strain)

        # calculate total force
        n_sec, m_x_sec, m_y_sec = self.ultimate_analysis(
            point_na=point_na,
            d_n=d_n,
            theta=theta,
            ultimate_strain=ultimate_strain,
            centroid=centroid,
        )

        # calculate point of action
        if n_sec == 0:
            d_x = 0
            d_y = 0
        else:
            d_x = m_y_sec / n_sec
            d_y = m_x_sec / n_sec

        return sig, n_sec, d_x, d_y

    def plot_mesh(
        self,
        alpha: float = 0.5,
        title: str = "Finite Element Mesh",
        **kwargs,
    ) -> matplotlib.axes.Axes:  # type: ignore
        """Plots the finite element mesh.

        :param alpha: Transparency of the mesh outlines
        :param title: Plot title
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        """

        with plotting_context(title=title, aspect=True, **kwargs) as (fig, ax):
            colour_array = []
            c = []  # Indices of elements for mapping colours

            # create an array of finite element colours
            for idx, el in enumerate(self.elements):
                colour_array.append(el.material.colour)
                c.append(idx)

            cmap = ListedColormap(colour_array)  # type: ignore

            # plot the mesh colours
            ax.tripcolor(  # type: ignore
                self.mesh_nodes[:, 0],
                self.mesh_nodes[:, 1],
                self.mesh_elements[:, 0:3],  # type: ignore
                c,
                cmap=cmap,
            )

            # plot the mesh
            ax.triplot(  # type: ignore
                self.mesh_nodes[:, 0],
                self.mesh_nodes[:, 1],
                self.mesh_elements[:, 0:3],  # type: ignore
                lw=0.5,
                color="black",
                alpha=alpha,
            )

        return ax

    def plot_shape(
        self,
        ax: matplotlib.axes.Axes,  # type: ignore
    ):
        """Plots the coloured shape of the mesh with no outlines on `ax`.

        :param ax: Matplotlib axes object
        """

        colour_array = []
        c = []  # Indices of elements for mapping colours

        # create an array of finite element colours
        for idx, el in enumerate(self.elements):
            colour_array.append(el.material.colour)
            c.append(idx)

        cmap = ListedColormap(colour_array)  # type: ignore

        # plot the mesh colours
        ax.tripcolor(
            self.mesh_nodes[:, 0],
            self.mesh_nodes[:, 1],
            self.mesh_elements[:, 0:3],  # type: ignore
            c,
            cmap=cmap,
        )


@dataclass
class Tri3:
    """Class for a three noded linear triangular element.

    :param coords: A 2 x 3 array of the coordinates of the tri-3 nodes
    :param node_ids: A list of the global node ids for the current element
    :param material: Material object for the current finite element
    """

    coords: np.ndarray
    node_ids: List[int]
    material: Material

    def calculate_area(self) -> float:
        """Calculates the area of the finite element.

        :return: Element area
        """

        area = 0

        # get points for 1 point Gaussian integration
        gps = utils.gauss_points(n=1)

        # loop through each gauss point
        for gp in gps:
            # determine shape function and jacobian
            _, j = utils.shape_function(coords=self.coords, gauss_point=gp)

            area += gp[0] * j

        return area

    def second_moments_of_area(self) -> Tuple[float, float, float]:
        """Calculates the second moments of area of the finite element.

        :return: Modulus weighted second moments of area *(e_ixx, e_iyy, e_ixy)*
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
                self.material.elastic_modulus
                * gp[0]
                * np.dot(N, np.transpose(self.coords[1, :])) ** 2
                * j
            )
            e_iyy += (
                self.material.elastic_modulus
                * gp[0]
                * np.dot(N, np.transpose(self.coords[0, :])) ** 2
                * j
            )
            e_ixy += (
                self.material.elastic_modulus
                * gp[0]
                * np.dot(N, np.transpose(self.coords[1, :]))
                * np.dot(N, np.transpose(self.coords[0, :]))
                * j
            )

        return e_ixx, e_iyy, e_ixy

    def calculate_elastic_actions(
        self,
        n: float,
        m_x: float,
        m_y: float,
        e_a: float,
        cx: float,
        cy: float,
        e_ixx: float,
        e_iyy: float,
        e_ixy: float,
    ) -> Tuple[float, float, float]:
        """Calculates elastic actions for the current finite element.

        :param n: Axial force
        :param m_x: Bending moment about the x-axis
        :param m_y: Bending moment about the y-axis
        :param e_a: Axial rigidity
        :param cx: x-Centroid
        :param cy: y-Centroid
        :param e_ixx: Flexural rigidity about the x-axis
        :param e_iyy: Flexural rigidity about the y-axis
        :param e_ixy: Flexural rigidity about the xy-axis

        :return: Elastic force and resultant moments
        """

        # initialise element results
        force_e = 0
        m_x_e = 0
        m_y_e = 0

        # get points for 3 point Gaussian integration
        gps = utils.gauss_points(n=3)

        # loop through each gauss point
        for gp in gps:
            # determine shape function and jacobian
            N, j = utils.shape_function(coords=self.coords, gauss_point=gp)

            # get coordinates (wrt NA) of the gauss point
            x = np.dot(N, np.transpose(self.coords[0, :])) - cx
            y = np.dot(N, np.transpose(self.coords[1, :])) - cy

            # axial force
            force_gp = 0
            force_gp += gp[0] * n * self.material.elastic_modulus / e_a * j

            # bending moment
            force_gp += (
                gp[0]
                * self.material.elastic_modulus
                * (
                    -(e_ixy * m_x) / (e_ixx * e_iyy - e_ixy**2) * x
                    + (e_iyy * m_x) / (e_ixx * e_iyy - e_ixy**2) * y
                )
                * j
            )
            force_gp += (
                gp[0]
                * self.material.elastic_modulus
                * (
                    +(e_ixx * m_y) / (e_ixx * e_iyy - e_ixy**2) * x
                    - (e_ixy * m_y) / (e_ixx * e_iyy - e_ixy**2) * y
                )
                * j
            )

            # add force and moment
            force_e += force_gp
            m_x_e += force_gp * y
            m_y_e += force_gp * x

        return force_e, m_x_e, m_y_e

    def calculate_service_actions(
        self,
        point_na: Tuple[float, float],
        theta: float,
        kappa: float,
        centroid: Tuple[float, float],
    ) -> Tuple[float, float, float, float, float]:
        r"""Calculates service actions for the current finite element.

        :param point_na: Point on the neutral axis
        :param theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)
        :param kappa: Curvature
        :param centroid: Centroid about which to take moments

        :return: Axial force, moments and min/max strain
        """

        # initialise element results
        force_e = 0
        m_x_e = 0
        m_y_e = 0
        min_strain_e = 0
        max_strain_e = 0

        # get points for 3 point Gaussian integration
        gps = utils.gauss_points(n=3)

        # loop through each gauss point
        for gp in gps:
            # determine shape function and jacobian
            N, j = utils.shape_function(coords=self.coords, gauss_point=gp)

            # get coordinates of the gauss point
            x = np.dot(N, np.transpose(self.coords[0, :]))
            y = np.dot(N, np.transpose(self.coords[1, :]))

            # get strain at gauss point
            strain = utils.get_service_strain(
                point=(x, y),
                point_na=point_na,
                theta=theta,
                kappa=kappa,
            )
            min_strain_e = min(min_strain_e, strain)
            max_strain_e = max(max_strain_e, strain)

            # get stress at gauss point
            stress = self.material.stress_strain_profile.get_stress(strain=strain)

            # calculate force (stress * area)
            force_gp = gp[0] * stress * j

            # add force and moment
            force_e += force_gp
            m_x_e += force_gp * (y - centroid[1])
            m_y_e += force_gp * (x - centroid[0])

        return force_e, m_x_e, m_y_e, min_strain_e, max_strain_e

    def calculate_ultimate_actions(
        self,
        point_na: Tuple[float, float],
        d_n: float,
        theta: float,
        ultimate_strain: float,
        centroid: Tuple[float, float],
    ) -> Tuple[float, float, float]:
        r"""Calculates ultimate actions for the current finite element.

        :param point_na: Point on the neutral axis
        :param d_n: Depth of the neutral axis from the extreme compression fibre
        :param theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)
        :param ultimate_strain: Concrete strain at failure
        :param centroid: Centroid about which to take moments

        :return: Axial force and resultant moments about the global axes
        """

        # initialise element results
        force_e = 0
        m_x_e = 0
        m_y_e = 0

        # get points for 3 point Gaussian integration
        gps = utils.gauss_points(n=3)

        # loop through each gauss point
        for gp in gps:
            # determine shape function and jacobian
            N, j = utils.shape_function(coords=self.coords, gauss_point=gp)

            # get coordinates of the gauss point
            x = np.dot(N, np.transpose(self.coords[0, :]))
            y = np.dot(N, np.transpose(self.coords[1, :]))

            # get strain at gauss point
            if isinf(d_n):
                strain = ultimate_strain
            else:
                strain = utils.get_ultimate_strain(
                    point=(x, y),
                    point_na=point_na,
                    d_n=d_n,
                    theta=theta,
                    ultimate_strain=ultimate_strain,
                )

            # get stress at gauss point
            if isinstance(self.material, Concrete):
                stress = self.material.ultimate_stress_strain_profile.get_stress(
                    strain=strain
                )
            else:
                stress = self.material.stress_strain_profile.get_stress(strain=strain)

            # calculate force (stress * area)
            force_gp = gp[0] * stress * j

            # add force and moment
            force_e += force_gp
            m_x_e += force_gp * (y - centroid[1])
            m_y_e += force_gp * (x - centroid[0])

        return force_e, m_x_e, m_y_e
