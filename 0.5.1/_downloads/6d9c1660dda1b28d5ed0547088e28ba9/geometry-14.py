from sectionproperties.pre.library.primitive_sections import circular_section_by_area

concrete = None  # define your concrete material properties here
steel = None  # define your steel material properties here
circle = circular_section_by_area(area=282.74e3, n=32, material=concrete)
geom = add_bar_circular_array(
  geometry=circle, area=310, material=steel, n_bar=7, r_array=250
)
geom.plot_geometry(labels=[], cp=False, legend=False)