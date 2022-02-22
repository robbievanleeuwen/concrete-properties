from sectionproperties.pre.library.concrete_sections import concrete_tee_section
from sectionproperties.pre.pre import Material
from sectionproperties.analysis.section import Section

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

geometry = concrete_tee_section(
    b=450,
    d=900,
    b_f=1200,
    d_f=250,
    dia=24,
    n_bar=5,
    n_circle=24,
    cover=30,
    conc_mat=concrete,
    steel_mat=steel,
)
geometry.create_mesh(mesh_sizes=[500])

section = Section(geometry)
section.plot_mesh()
