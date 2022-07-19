from sectionproperties.pre.library.bridge_sections import super_t_girder_section

concrete = None  # define your concrete material properties here
geom = super_t_girder_section(girder_type=5, material=concrete)
geom.plot_geometry(labels=[], cp=False, legend=False)