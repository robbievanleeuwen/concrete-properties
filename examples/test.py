from sectionproperties.pre.library.concrete_sections import concrete_tee_section
from sectionproperties.analysis.section import Section
from concreteproperties.material import Concrete, Steel
from concreteproperties.concrete_section import ConcreteSection

concrete = Concrete(
    name="32 MPa Concrete",
    elastic_modulus=30.1e3,
    compressive_strength=32,
    density=2.4e-6,
)

steel = Steel(
    name="500 MPa Steel",
    elastic_modulus=200e3,
    yield_strength=500,
    density=7.85e-6,
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

conc_sec = ConcreteSection(section)
conc_sec.concrete_section.plot_mesh()
