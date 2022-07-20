from __future__ import annotations

from typing import List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np
from matplotlib.colors import ListedColormap
import triangle

import concreteproperties.utils as utils
from concreteproperties.post import plotting_context

import sectionproperties.analysis.fea as sp_fea

if TYPE_CHECKING:
    import matplotlib
    from concreteproperties.material import Concrete
    from sectionproperties.pre.geometry import Geometry


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
        try:
            self.mesh_elements = np.array(self.mesh["triangles"], dtype=np.dtype(int))
        except KeyError:
            # if there are no triangles
            self.mesh_elements = []

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

        :param float n: Axial force
        :param float m_x: Bending moment about the x-axis
        :param float m_y: Bending moment about the y-axis
        :param float e_a: Axial rigidity
        :param float cx: x-Centroid
        :param float cy: y-Centroid
        :param float e_ixx: Flexural rigidity about the x-axis
        :param float e_iyy: Flexural rigidity about the y-axis
        :param float e_ixy: Flexural rigidity about the xy-axis

        :return: Elastic stresses, net force and distance from neutral axis to point of
            force action
        :rtype: Tuple[:class:`numpy.ndarray`, float, float, float]
        """

        # intialise stress results
        sig = np.zeros(len(self.mesh_nodes))

        # loop through nodes
        for idx, node in enumerate(self.mesh_nodes):
            x = node[0] - cx
            y = node[1] - cy

            # axial stress
            sig[idx] += n * self.geometry.material.elastic_modulus / e_a

            # bending moment
            sig[idx] += self.geometry.material.elastic_modulus * (
                -(e_ixy * m_x) / (e_ixx * e_iyy - e_ixy**2) * x
                + (e_iyy * m_x) / (e_ixx * e_iyy - e_ixy**2) * y
            )
            sig[idx] += self.geometry.material.elastic_modulus * (
                +(e_ixx * m_y) / (e_ixx * e_iyy - e_ixy**2) * x
                - (e_ixy * m_y) / (e_ixx * e_iyy - e_ixy**2) * y
            )

        # initialise section actions
        n_conc = 0
        m_x_conc = 0
        m_y_conc = 0

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

            n_conc += el_n
            m_x_conc += el_m_x
            m_y_conc += el_m_y

        # calculate point of action
        if n_conc == 0:
            d_x = 0
            d_y = 0
        else:
            d_x = m_y_conc / n_conc
            d_y = m_x_conc / n_conc

        return sig, n_conc, d_x, d_y

    def service_stress_analysis(
        self,
        point_na: Tuple[float],
        d_n: float,
        theta: float,
        kappa: float,
        na_local: float,
    ) -> Tuple[float]:
        r"""Performs a service stress analysis on the section.

        :param point_na: Point on the neutral axis
        :type point_na: Tuple[float]
        :param float d_n: Depth of the neutral axis from the extreme compression fibre
        :param float theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)
        :param float kappa: Curvature
        :param float na_local: y-location of the neutral axis in local coordinates

        :return: Axial force, resultant moment and max strain
        :rtype: Tuple[float]
        """

        # initialise section actions
        n = 0
        m_u = 0
        max_strain = 0

        for el in self.elements:
            el_n, el_m_u, el_max_strain = el.calculate_service_actions(
                point_na=point_na,
                d_n=d_n,
                theta=theta,
                kappa=kappa,
                na_local=na_local,
            )
            max_strain = max(max_strain, el_max_strain)

            n += el_n
            m_u += el_m_u

        return n, m_u, max_strain

    def get_service_stress(
        self,
        d_n: float,
        kappa: float,
        point_na: Tuple[float],
        theta: float,
        na_local: float,
    ) -> Tuple[np.ndarray, float, float]:
        r"""Given the neutral axis depth `d_n` and curvature `kappa` determines the
        service stresses within the section.

        :param float d_n: Neutral axis depth
        :param float kappa: Curvature
        :param point_na: Point on the neutral axis
        :type point_na: Tuple[float]
        :param float theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)
        :param float na_local: y-location of the neutral axis in local coordinates

        :return: Service stresses, net force and distance from neutral axis to point of
            force action
        :rtype: Tuple[:class:`numpy.ndarray`, float, float]
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

            # get stress at gauss point
            sig[idx] = self.geometry.material.stress_strain_profile.get_stress(
                strain=strain
            )

        # calculate total force
        n, m_u, _ = self.service_stress_analysis(
            point_na=point_na,
            d_n=d_n,
            theta=theta,
            kappa=kappa,
            na_local=na_local,
        )

        # calculate point of action
        if n == 0:
            d = 0
        else:
            d = m_u / n

        return sig, n, d

    def ultimate_stress_analysis(
        self,
        point_na: Tuple[float],
        d_n: float,
        theta: float,
        ultimate_strain: float,
        pc_local: float,
    ) -> Tuple[float]:
        r"""Performs an ultimate stress analysis on the section.

        :param point_na: Point on the neutral axis
        :type point_na: Tuple[float]
        :param float d_n: Depth of the neutral axis from the extreme compression fibre
        :param float theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)
        :param float ultimate_strain: Concrete strain at failure
        :param float pc_local: y-location of the plastic centroid in local coordinates

        :return: Axial force and resultant moment
        :rtype: Tuple[float]
        """

        # initialise section actions
        n = 0
        m_u = 0

        for el in self.elements:
            el_n, el_m_u = el.calculate_ultimate_actions(
                point_na=point_na,
                d_n=d_n,
                theta=theta,
                ultimate_strain=ultimate_strain,
                pc_local=pc_local,
            )

            n += el_n
            m_u += el_m_u

        return n, m_u

    def get_ultimate_stress(
        self,
        d_n: float,
        point_na: Tuple[float],
        theta: float,
        ultimate_strain: float,
        pc_local: float,
    ) -> Tuple[np.ndarray, float, float]:
        r"""Given the neutral axis depth `d_n` and ultimate strain, determines the
        ultimate stresses with the section.

        :param float d_n: Neutral axis depth
        :param point_na: Point on the neutral axis
        :type point_na: Tuple[float]
        :param float theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)
        :param float ultimate_strain: Concrete strain at failure
        :param float pc_local: y-location of the plastic centroid in local coordinates

        :return: Ultimate stresses net force and distance from neutral axis to point of
            force action
        :rtype: Tuple[:class:`numpy.ndarray`, float, float]
        """

        # intialise stress results
        sig = np.zeros(len(self.mesh_nodes))

        # loop through nodes
        for idx, node in enumerate(self.mesh_nodes):
            # get strain at node
            strain = utils.get_ultimate_strain(
                point=(node[0], node[1]),
                point_na=point_na,
                d_n=d_n,
                theta=theta,
                ultimate_strain=ultimate_strain,
            )

            # get stress at gauss point
            sig[idx] = self.geometry.material.ultimate_stress_strain_profile.get_stress(
                strain=strain
            )

        # calculate total force
        n, m_u = self.ultimate_stress_analysis(
            point_na=point_na,
            d_n=d_n,
            theta=theta,
            ultimate_strain=ultimate_strain,
            pc_local=pc_local,
        )

        # calculate point of action
        if n == 0:
            d = 0
        else:
            # get principal coordinates of neutral axis
            na_local = utils.principal_coordinate(
                phi=theta * 180 / np.pi, x=point_na[0], y=point_na[1]
            )

            d = m_u / n + pc_local - na_local[1]

        return sig, n, d

    def plot_mesh(
        self,
        alpha: Optional[float] = 0.5,
        title: Optional[str] = "Finite Element Mesh",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots the finite element mesh.

        :param alpha: Transparency of the mesh outlines
        :type alpha: Optional[float]
        :param title: Plot title
        :type title: Optional[str]
        :param kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        :return: Matplotlib axes object
        :rtype: :class:`matplotlib.axes.Axes`
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
            ax.tripcolor(
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
                color="black",
                alpha=alpha,
            )

            ax.set_aspect("equal", anchor="C")

        return ax

    def plot_shape(
        self,
        ax: matplotlib.axes.Axes,
    ):
        """Plots the coloured shape of the mesh with no outlines on `ax`.

        :param ax: Matplotlib axes object
        :type ax: :class:`matplotlib.axes.Axes`
        """

        colour_array = []
        c = []  # Indices of elements for mapping colours

        # create an array of finite element colours
        for idx, element in enumerate(self.elements):
            colour_array.append(element.conc_material.colour)
            c.append(idx)

        cmap = ListedColormap(colour_array)  # custom colourmap

        # plot the mesh colours
        ax.tripcolor(
            self.mesh_nodes[:, 0],
            self.mesh_nodes[:, 1],
            self.mesh_elements[:, 0:3],
            c,
            cmap=cmap,
        )


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
    ) -> Tuple[float]:
        r"""Calculates elastic actions for the current finite element.

        :param float n: Axial force
        :param float m_x: Bending moment about the x-axis
        :param float m_y: Bending moment about the y-axis
        :param float e_a: Axial rigidity
        :param float cx: x-Centroid
        :param float cy: y-Centroid
        :param float e_ixx: Flexural rigidity about the x-axis
        :param float e_iyy: Flexural rigidity about the y-axis
        :param float e_ixy: Flexural rigidity about the xy-axis

        :return: Elastic force and resultant moment
        :rtype: Tuple[float]
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
            force_gp += gp[0] * n * self.conc_material.elastic_modulus / e_a * j

            # bending moment
            force_gp += (
                gp[0]
                * self.conc_material.elastic_modulus
                * (
                    -(e_ixy * m_x) / (e_ixx * e_iyy - e_ixy**2) * x
                    + (e_iyy * m_x) / (e_ixx * e_iyy - e_ixy**2) * y
                )
                * j
            )
            force_gp += (
                gp[0]
                * self.conc_material.elastic_modulus
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
        point_na: Tuple[float],
        d_n: float,
        theta: float,
        kappa: float,
        na_local: float,
    ) -> Tuple[float]:
        r"""Calculates service actions for the current finite element.

        :param point_na: Point on the neutral axis
        :type point_na: Tuple[float]
        :param float d_n: Depth of the neutral axis from the extreme compression fibre
        :param float theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)
        :param float kappa: Curvature
        :param float na_local: y-location of the neutral axis in local coordinates

        :return: Axial force, resultant moment and maximum strain
        :rtype: Tuple[float]
        """

        # initialise element results
        force_e = 0
        m_u_e = 0
        max_strain_e = 0

        # get points for 1 point Gaussian integration
        gps = utils.gauss_points(n=1)

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
            max_strain_e = max(max_strain_e, strain)

            # get stress at gauss point
            stress = self.conc_material.stress_strain_profile.get_stress(strain=strain)

            # calculate force (stress * area)
            force_gp = gp[0] * stress * j

            # convert gauss point to local coordinates
            _, c_v = sp_fea.principal_coordinate(phi=theta * 180 / np.pi, x=x, y=y)

            # add force and moment
            force_e += force_gp
            m_u_e += force_gp * (c_v - na_local)

        return force_e, m_u_e, max_strain_e

    def calculate_ultimate_actions(
        self,
        point_na: Tuple[float],
        d_n: float,
        theta: float,
        ultimate_strain: float,
        pc_local: float,
    ) -> Tuple[float]:
        r"""Calculates ultimate actions for the current finite element.

        :param point_na: Point on the neutral axis
        :type point_na: Tuple[float]
        :param float d_n: Depth of the neutral axis from the extreme compression fibre
        :param float theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)
        :param float ultimate_strain: Concrete strain at failure
        :param float pc_local: y-location of the plastic centroid in local coordinates

        :return: Axial force and resultant moment
        :rtype: Tuple[float]
        """

        # initialise element results
        force_e = 0
        m_u_e = 0

        # get points for 1 point Gaussian integration
        gps = utils.gauss_points(n=1)

        # loop through each gauss point
        for gp in gps:
            # determine shape function and jacobian
            N, j = utils.shape_function(coords=self.coords, gauss_point=gp)

            # get coordinates of the gauss point
            x = np.dot(N, np.transpose(self.coords[0, :]))
            y = np.dot(N, np.transpose(self.coords[1, :]))

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
            force_gp = gp[0] * stress * j

            # convert gauss point to local coordinates
            _, c_v = sp_fea.principal_coordinate(phi=theta * 180 / np.pi, x=x, y=y)

            # add force and moment
            force_e += force_gp
            m_u_e += force_gp * (c_v - pc_local)

        return force_e, m_u_e
