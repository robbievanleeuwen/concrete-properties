from sectionproperties.pre.library.primitive_sections import rectangular_section

concrete = None  # define your concrete material properties here
outer = rectangular_section(d=800, b=600, material=concrete)
inner = rectangular_section(d=600, b=400).align_center(align_to=outer)
geom = outer - inner
geom.plot_geometry()