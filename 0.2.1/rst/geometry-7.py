from sectionproperties.pre.geometry import Geometry

concrete = None  # define your concrete material properties here

pts = [
  [0, 0], [600, 0], [600, 800], [0, 800], [100, 100], [500, 100], [500, 700],
  [100, 700]
]
fcts = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 0]]
cps = [[50, 50]]
hls = [[400, 400]]

geom = Geometry.from_points(
  points=pts, facets=fcts, control_points=cps, holes=hls, material=concrete
)
geom.plot_geometry()