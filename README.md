# concreteproperties

[![Run Tests](https://github.com/robbievanleeuwen/concrete-properties/actions/workflows/tests.yml/badge.svg)](https://github.com/robbievanleeuwen/concrete-properties/actions/workflows/tests.yml) [![Lint with Black](https://github.com/robbievanleeuwen/concrete-properties/actions/workflows/black.yml/badge.svg)](https://github.com/robbievanleeuwen/concrete-properties/actions/workflows/black.yml) [![Documentation Status](https://readthedocs.org/projects/concrete-properties/badge/?version=latest)](https://concrete-properties.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/concreteproperties.svg)](https://badge.fury.io/py/concreteproperties) [![Python versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue?style=flat&logo=python)](https://badge.fury.io/py/concreteproperties) [![GitHub license](https://img.shields.io/github/license/robbievanleeuwen/concrete-properties)](https://github.com/robbievanleeuwen/concrete-properties/blob/master/LICENSE.md)

Calculate section properties for reinforced concrete sections.

```shell
pip install concreteproperties
```

## To do:
- [ ] Expand material properties
  - [ ] Tensile strength
  - [ ] Residual shrinkage tensile stress
- [ ] Add concrete property calculations
  - [ ] Gross second moment of area (I_g)
  - [ ] Cracking moment (M_cr)
  - [ ] Cracked second moment of area (I_cr)
  - [ ] Effective second moment of area (I_ef)
  - [ ] Reporting of k_u
- [ ] Add stress calculations
  - [ ] Uncracked stresses
  - [ ] Cracked stresses
  - [ ] Stresses at ultimate
- [ ] Add code module
  - [ ] Codify calculation of material properties
- [ ] Add visualisation
  - [ ] Stress visualisation
  - [ ] Free body diagrams
- [ ] Expand to include prestressed concrete
- [ ] Exclude holes made by reinforcement in ultimate calculation
