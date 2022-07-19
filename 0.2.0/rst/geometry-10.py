from sectionproperties.pre.library.primitive_sections import rectangular_section
from sectionproperties.pre.library.bridge_sections import super_t_girder_section

conc_precast = None  # define your concrete material properties here
conc_insitu = None  # define your concrete material properties here
beam = super_t_girder_section(girder_type=5, material=conc_precast)
slab = rectangular_section(
  d=180, b=2100, material=conc_insitu
).shift_section(-1050, 75)
geom = beam + slab
geom.plot_geometry()