from sectionproperties.pre.library.primitive_sections import rectangular_section

concrete = None  # define your concrete material properties here
slab = rectangular_section(d=150, b=800, material=concrete)
beam = rectangular_section(
  d=600, b=300, material=concrete
).align_to(other=slab, on="bottom").align_to(other=slab, on="left", inner=True)
geom = slab + beam
geom.plot_geometry(labels=[], cp=False, legend=False)