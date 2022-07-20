![Logo Dark](docs/source/_static/cp_logo_dark.png#gh-dark-mode-only)
![Logo Light](docs/source/_static/cp_logo.png#gh-light-mode-only)

[![Run Tests](https://github.com/robbievanleeuwen/concrete-properties/actions/workflows/tests.yml/badge.svg)](https://github.com/robbievanleeuwen/concrete-properties/actions/workflows/tests.yml) [![Lint with Black](https://github.com/robbievanleeuwen/concrete-properties/actions/workflows/black.yml/badge.svg)](https://github.com/robbievanleeuwen/concrete-properties/actions/workflows/black.yml) [![Build Documentation](https://github.com/robbievanleeuwen/concrete-properties/actions/workflows/build_docs.yml/badge.svg)](https://robbievanleeuwen.github.io/concrete-properties/) [![codecov](https://codecov.io/gh/robbievanleeuwen/concrete-properties/branch/master/graph/badge.svg?token=3WXMUQITTD)](https://codecov.io/gh/robbievanleeuwen/concrete-properties) [![PyPI version](https://badge.fury.io/py/concreteproperties.svg)](https://badge.fury.io/py/concreteproperties) [![Python versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue?style=flat&logo=python)](https://badge.fury.io/py/concreteproperties) [![GitHub license](https://img.shields.io/github/license/robbievanleeuwen/concrete-properties)](https://github.com/robbievanleeuwen/concrete-properties/blob/master/LICENSE.md) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/robbievanleeuwen/concrete-properties-examples/master)

*concreteproperties* is a python package that can be used to calculate the section
properties of arbitrary reinforced concrete sections. *concreteproperties* can calculate
gross, cracked and ultimate properties. It can perform moment curvature analyses
and generate moment interaction and biaxial bending diagrams. On top of this,
*concreteproperties* can also generate pretty stress plots!

Here's an example of some of the non-linear output *concreteproperties* can generate:

<p align="center">
  <img src="docs/source/_static/anim/anim_compress.gif" width="500"/>
</p>

## Installation:

For more detailed installation instructions, refer to the [documentation](https://robbievanleeuwen.github.io/concrete-properties/rst/installation.html).

```shell
pip install concreteproperties
```

## Documentation:

*concreteproperties* is fully documented including examples and a fully documented API.
The documentation can found at [https://robbievanleeuwen.github.io/concrete-properties](https://robbievanleeuwen.github.io/concrete-properties).

## Contributing

We welcome anyone interested in contributing to *concreteproperties*, whether it be
through submitting bug reports, feature requests or pull requests. Please read the
[contributing guide](.github/CONTRIBUTING.md) prior to contributing.

## Current Capabilities:

### Material Properties
- [x] Concrete material
  - [x] Service stress-strain profiles
    - [x] Linear profile
    - [x] Linear profile (no tension)
    - [x] Eurocode non-linear
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
- [x] Design code modules
  - [x] AS3600
  - [ ] AS5100

## Disclaimer:

*concreteproperties* is an open source engineering tool that continues to benefit from
the collaboration of many contributors. Although efforts have been made to ensure the
that relevant engineering theories have been correctly implemented, it remains the
user's responsibility to confirm and accept the output. Refer to the
[license](LICENSE.md) for clarification of the conditions of use.
