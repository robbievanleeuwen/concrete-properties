"""Regression test for degenerate slivers produced by CPGeom.split_section.

See https://github.com/robbievanleeuwen/concrete-properties/issues/153. For
some load angles/curvatures the neutral axis passes so close to an existing
vertex that, after coordinate rounding, the resulting sliver collapses to
fewer than three distinct points. Passing such a sliver on to
``AnalysisSection`` used to raise a ``ValueError`` from the triangulation
library instead of being silently discarded.
"""

import math

import numpy as np
import sectionproperties.pre.library.concrete_sections as sp_cs
from sectionproperties.pre.library.primitive_sections import rectangular_section

from concreteproperties.concrete_section import ConcreteSection
from concreteproperties.material import Concrete, SteelBar
from concreteproperties.stress_strain_profile import (
    EurocodeNonLinear,
    EurocodeParabolicUltimate,
    SteelProfile,
)


def test_moment_curvature_analysis_does_not_crash_on_degenerate_split():
    fck = 30
    fy = 500
    width = 250
    depth = 250

    elastic_modulus = 5600 * fck**0.5

    concrete = Concrete(
        name="Concrete",
        density=2.5e-6,
        stress_strain_profile=EurocodeNonLinear(
            elastic_modulus=elastic_modulus,
            ultimate_strain=0.0035,
            compressive_strength=fck,
            compressive_strain=0.002,
            tensile_strength=1e-3,
            tension_softening_stiffness=1,
        ),
        ultimate_stress_strain_profile=EurocodeParabolicUltimate(
            compressive_strength=fck,
            compressive_strain=0.002,
            ultimate_strain=0.0035,
            n=2,
        ),
        flexural_tensile_strength=0,
        colour="grey",
    )

    steel = SteelBar(
        name="Bar",
        density=7.85e-6,
        stress_strain_profile=SteelProfile(
            strains=[-0.05, -0.03, -0.02, -fy / 200e3, 0, 1],
            stresses=[-fy, -fy, -fy, -fy, 0, 100],
            yield_strength=fy,
            elastic_modulus=200e3,
            fracture_strain=0.05,
        ),
        colour="orange",
    )

    geom = rectangular_section(d=width, b=depth, material=concrete)
    for dia, x, y in [(22.225, 35, 35), (22.225, 215, 215), (22.225, 35, 215), (22.225, 215, 35)]:
        geom = sp_cs.add_bar(
            geometry=geom,
            area=math.pi * dia**2 * 0.25,
            material=steel,
            x=x,
            y=y,
            n=6,
        )

    conc_section = ConcreteSection(geom)

    inertia = conc_section.get_transformed_gross_properties(
        elastic_modulus=elastic_modulus
    )

    # angle/curvature combination that used to hit a degenerate split geometry
    mx, mz = -25e6, 1e6
    kappa_x = mx * inertia.ixx_c + mz * inertia.ixy_c
    kappa_y = mz * inertia.iyy_c + mx * inertia.ixy_c
    theta = np.arctan2(kappa_y, kappa_x)

    # should complete without raising
    conc_section.moment_curvature_analysis(theta=theta, n=0, progress_bar=False)
