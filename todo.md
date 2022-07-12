# RELEASE ITEMS
- [x] Obtain ultimate stresses (using d_n to work out strains at nodes, then get stress)
- [x] Do the same thing for service stresses
- [x] Stress plot - add tricontour for zero stress only?
- [x] Add testing
  - [x] Stresses for RCB tests
  - [x] Rotation tests - rotate geometry & theta and check results are constant
  - [x] Stress forces - check force equilibrium and desired moments
- [x] FORCE THETA -pi to pi
- [x] Add steel strain to StressResult & ku to UltimateResult (=dn/d)
- [x] Clean up profiles - maybe require that certain profiles may only be used for each material?
- [x] Finish pre.py
- [ ] Documentation!
  - [x] Is there a way to generate output within a docstring??
  - [ ] gif of progressive service stress plot?
  - [ ] Fix mv!!!

# POST-RELEASE ITEMS
- [ ] Speed up
  - [ ] Profiling
  - [ ] Clean up all the na_local, points_na etc. etc.
- [ ] Start to add design codes