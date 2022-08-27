from sectionproperties.pre.library.primitive_sections import rectangular_section

concrete = None  # define your concrete material properties here
geom = rectangular_section(d=600, b=300, material=concrete)
geom.plot_geometry(labels=[], cp=False, legend=False)