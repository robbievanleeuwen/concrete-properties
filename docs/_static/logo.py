"""Script to generate the concreteproperties logo."""

from sectionproperties.analysis.section import Section
from sectionproperties.pre.geometry import Geometry
from sectionproperties.pre.library.primitive_sections import circular_section
from sectionproperties.pre.pre import Material

concrete = Material(
    name="Concrete",
    elastic_modulus=30.1e3,
    poissons_ratio=0.2,
    yield_strength=32,
    density=2.4e-6,
    color="lightgrey",
)
steel = Material(
    name="Steel",
    elastic_modulus=200e3,
    poissons_ratio=0.3,
    yield_strength=500,
    density=7.85e-6,
    color="grey",
)

t = 150
d = 500
b = 2400
c = 15

points = [
    (0, c),
    (0, d - c),
    (c, d),
    (b - c, d),
    (b, d - c),
    (b, c),
    (b - c, 0),
    (b - t + c, 0),
    (b - t, c),
    (b - t, d - t - c),
    (b - t - c, d - t),
    (t + c, d - t),
    (t, d - t - c),
    (t, c),
    (t - c, 0),
    (c, 0),
]
facets = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (15, 0),
]
control_points = [(0.5 * t, 0.5 * d)]

beam = Geometry.from_points(points, facets, control_points, material=concrete)
beam.material = concrete

n_top = 14
n_wall = 3

for idx in range(n_top):
    bar = circular_section(d=20, n=16, material=steel).shift_section(
        x_offset=0.5 * t + idx * (b - t) / (n_top - 1), y_offset=d - 0.5 * t
    )
    beam = (beam - bar) + bar

for idx in range(n_wall):
    bar = circular_section(d=20, n=16, material=steel).shift_section(
        x_offset=0.5 * t, y_offset=0.5 * t + idx * (d - t) / n_wall
    )
    beam = (beam - bar) + bar

for idx in range(n_wall):
    bar = circular_section(d=20, n=16, material=steel).shift_section(
        x_offset=b - 0.5 * t, y_offset=0.5 * t + idx * (d - t) / n_wall
    )
    beam = (beam - bar) + bar

beam.create_mesh([300.0])

sec = Section(beam)
sec.plot_mesh()
