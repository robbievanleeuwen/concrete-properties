import sectionproperties.pre.library.primitive_sections as sp_ps

conc = None  # define your concrete material properties here
steel = None  # define your steel material properties here
steel_col = sp_ps.circular_section_by_area(
  area=np.pi * 323.9**2 / 4,
  n=64,
  material=steel,
)
inner_conc = sp_ps.circular_section_by_area(
  area=np.pi * (323.9 - 2 * 12.7) ** 2 / 4,
  n=64,
  material=conc,
)
geom = steel_col - inner_conc + inner_conc
geom.plot_geometry()