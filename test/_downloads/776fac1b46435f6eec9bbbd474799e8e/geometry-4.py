from sectionproperties.pre.library.concrete_sections import concrete_tee_section

concrete = None  # define your concrete material properties here
steel = None  # define your steel material properties here
geom = concrete_tee_section(
  b=450, d=1200, b_f=1500, d_f=200, dia_top=24, n_top=8, dia_bot=32, n_bot=4,
  n_circle=12, cover=30, conc_mat=concrete, steel_mat=steel
)
geom.plot_geometry(labels=[], cp=False, legend=False)