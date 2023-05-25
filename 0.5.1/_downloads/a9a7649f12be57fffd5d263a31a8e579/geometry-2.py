from sectionproperties.pre.library.primitive_sections import circular_section

concrete = None  # define your concrete material properties here
geom = circular_section(d=600, n=32, material=concrete)
geom.plot_geometry(labels=[], cp=False, legend=False)