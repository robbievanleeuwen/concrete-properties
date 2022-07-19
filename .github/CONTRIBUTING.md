# Contributing to *concreteproperties*

👍🎉 First off, thanks for taking the time to contribute! 🎉👍

The following is a set of guidelines for contributing to *concreteproperties*. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

*concreteproperties* is intended to be used by structural engineers all over the world in the context many different design codes. *concreteproperties* started with only one design code (Australian standard AS 3600:2018) and it is a goal of *concreteproperties* to implement as many reinforced concrete design codes as possible, which wouldn't be possible without you 🙌

## Code of Conduct

This project and everyone participating in it is governed by the Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to robbie.vanleeuwen@gmail.com.

## Bug Reports

If you think you have identified a bug in *concreteproperties*, please submit a bug report by [raising an issue](https://github.com/robbievanleeuwen/concrete-properties/issues). Please follow the ``bug report template`` to help us understand the report and be able to reproduce the issue.

## Suggesting Enhancements

Features that improve *concreteproperties* can be suggested by [raising an issue](https://github.com/robbievanleeuwen/concrete-properties/issues) and following the ``feature request template``. It's a good idea to first submit the proposal as a feature request prior to submitting a pull request as this allows for the best coordination of efforts by preventing the duplication of work, and allows for feedback on your ideas.

Dependencies for developing with *concreteproperties* can be installed with:

```shell
pip install concreteproperties[dev]
```

## Styleguide

If submitting a pull request, please follow the below style guide to ensure a smooth review and approval process.

### Python

All python code is linted with [black](https://github.com/psf/black). 

All classes, methods and functions must have docstrings capable of being interpreted by sphinx, documenting all parameters and outputs. Use typehints where possible. Refer to the existing code for examples.

All new code should have associated [tests](https://github.com/robbievanleeuwen/concrete-properties/tree/master/concreteproperties/tests) where possible. Code coverage can be viewed [here](https://app.codecov.io/gh/robbievanleeuwen/concrete-properties).

### Documentation

New features in *concreteproperties* should be documented on the relevant pages. For example new stress-strain profiles should be added to the [Materials](https://github.com/robbievanleeuwen/concrete-properties/blob/master/docs/source/rst/materials.rst) page and new design codes should be added to the [Design Codes](https://github.com/robbievanleeuwen/concrete-properties/blob/master/docs/source/rst/design_codes.rst) page. Where possible and relevant, use sphinx directives to make links to classes, functions and methods mentioned in the documentation. Refer to the existing methods and classes to ensure consistent formatting and fluidity of the documentation.

Examples can be added by adding a ``jupyter notebook`` file to the [notebooks](https://github.com/robbievanleeuwen/concrete-properties/tree/master/docs/source/notebooks) folder and referencing the file on the [Examples](https://github.com/robbievanleeuwen/concrete-properties/blob/master/docs/source/rst/examples.rst) page. It is recommended that the documentation be built locally with ``make html`` prior to commiting to the pull request to ensure there are no build errors.
