from sectionproperties.pre.library.bridge_sections import super_t_girder_section

concrete = None  # define your concrete material properties here
steel = None  # define your steel material properties here
beam = super_t_girder_section(girder_type=5, material=concrete)
geom = add_bar(geometry=beam, area=5000, material=steel, x=0, y=-1550)
geom.plot_geometry(labels=[], cp=False, legend=False)