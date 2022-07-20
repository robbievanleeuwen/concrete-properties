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
- [x] Documentation!
  - [x] Is there a way to generate output within a docstring??
  - [x] gif of progressive service stress plot?
- [x] Fix mv!!!
- [x] CRACKED STRESS - resolve moments!

# POST-RELEASE ITEMS
- [ ] Speed up
  - [ ] Profiling
  - [ ] Clean up all the na_local, points_na etc. etc.
- [ ] Add prestressing
- [x] Docs version control
- [x] Start to add design codes

# Version Instructions
1. Change version in __init__.py
2. Update switcher.json
3. Push to github & create release
4. Update version_match in conf.py
5. Update directory in docs workflow
6. Push to github to create docs version
7. Change version_match back to latest & workflow back to main directory
