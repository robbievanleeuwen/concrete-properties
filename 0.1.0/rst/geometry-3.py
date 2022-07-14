import math
from sectionproperties.pre.library.primitive_sections import circular_section_by_area

concrete = None  # define your concrete material properties here
geom = circular_section_by_area(area=math.pi * 600 * 600 / 4, n=32, material=concrete)
geom.plot_geometry(labels=[], cp=False, legend=False)