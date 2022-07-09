![Logo Dark](docs/source/_static/cp_logo_dark.png#gh-dark-mode-only)
![Logo Light](docs/source/_static/cp_logo.png#gh-light-mode-only)

[![Run Tests](https://github.com/robbievanleeuwen/concrete-properties/actions/workflows/tests.yml/badge.svg)](https://github.com/robbievanleeuwen/concrete-properties/actions/workflows/tests.yml) [![Lint with Black](https://github.com/robbievanleeuwen/concrete-properties/actions/workflows/black.yml/badge.svg)](https://github.com/robbievanleeuwen/concrete-properties/actions/workflows/black.yml) [![Build Documentation](https://github.com/robbievanleeuwen/concrete-properties/actions/workflows/build_docs.yml/badge.svg)](https://robbievanleeuwen.github.io/concrete-properties/) [![codecov](https://codecov.io/gh/robbievanleeuwen/concrete-properties/branch/master/graph/badge.svg?token=3WXMUQITTD)](https://codecov.io/gh/robbievanleeuwen/concrete-properties) [![PyPI version](https://badge.fury.io/py/concreteproperties.svg)](https://badge.fury.io/py/concreteproperties) [![Python versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue?style=flat&logo=python)](https://badge.fury.io/py/concreteproperties) [![GitHub license](https://img.shields.io/github/license/robbievanleeuwen/concrete-properties)](https://github.com/robbievanleeuwen/concrete-properties/blob/master/LICENSE.md)

A python package to calculate the section properties of arbitrary reinforced concrete
sections.

## Installation:

For more detailed installation instructions, refer to the [documentation](https://robbievanleeuwen.github.io/concrete-properties/rst/installation.html).

```shell
pip install concreteproperties
```

## Documentation:

The documentation for *concreteproperties* is currently under construction. The
documentation can found at [https://robbievanleeuwen.github.io/concrete-properties](https://robbievanleeuwen.github.io/concrete-properties).

## Current Capabilities:

### Material Properties
- [x] Concrete material
  - [x] Service stress-strain profiles
    - [x] Linear profile
    - [x] Linear profile (no tension)
    - [x] Eurocode Non-Linear
  - [x] Ultimate stress-strain profiles
    - [x] Rectangular stress block
    - [x] Bilinear stress-strain profile
    - [x] Eurocode parabolic
  - [x] Flexural tensile strength
- [x] Steel material
  - [x] Stress-strain profiles
    - [x] Elastic-plastic
    - [x] Elastic-plastic (with hardening)

### Gross Section Properties
- [x] Cross-sectional areas (total, concrete, steel)
- [x] Axial rigidity
- [x] Cross-section mass
- [x] Cross-section perimeter
- [x] First moments of area
- [x] Elastic centroid
- [x] Global second moments of area
- [x] Centroidal second moments of area
- [x] Principal axis angle
- [x] Principal second moments of area
- [x] Centroidal section moduli
- [x] Principal section moduli

### Service Analysis
- [x] Cracking moment
- [x] Cracked second moment of area
- [x] Moment-curvature diagram

### Ultimate Analysis
- [x] Ultimate bending capacity
- [x] Squash load
- [x] Tensile load
- [x] Moment interaction diagrams
  - [x] M-N curves
  - [x] Biaxial bending curve

### Stress Analysis
- [x] Uncracked stresses
- [x] Cracked stresses
- [x] Service stresses
- [x] Ultimate stresses

### Design Codes
- [ ] Design code modules
  - [ ] AS3600
  - [ ] AS5100
