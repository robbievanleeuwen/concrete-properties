from sectionproperties.pre.library.primitive_sections import rectangular_section

concrete = None  # define your concrete material properties here
steel = None  # define your steel material properties here
beam = rectangular_section(d=500, b=300, material=concrete)
geom = add_bar_rectangular_array(
  geometry=beam, area=310, material=steel, n_x=3, x_s=110, n_y=2, y_s=420,
  anchor=(40, 40)
)
geom.plot_geometry(labels=[], cp=False, legend=False)