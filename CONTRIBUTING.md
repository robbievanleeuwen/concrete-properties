# Contributor Guide

Thank you for your interest in improving this project. This project is open-source under the [MIT license] and welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

- [Source Code]
- [Documentation]
- [Issue Tracker]
- [Code of Conduct]

[mit license]: https://opensource.org/licenses/MIT
[source code]: https://github.com/robbievanleeuwen/concrete-properties
[documentation]: https://concrete-properties.readthedocs.io/
[issue tracker]: https://github.com/robbievanleeuwen/concrete-properties/issues

## How to report a bug

Report bugs on the [Issue Tracker].

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case, and/or steps to reproduce the issue.

## How to request a feature

Features that improve `concreteproperties` can be suggested on the [Issue Tracker]. It's a good idea to first submit the proposal as a feature request prior to submitting a pull request as this allows for the best coordination of efforts by preventing the duplication of work, and allows for feedback on your ideas.

## How to set up your development environment

`concreteproperties` uses `uv` for python project management. `uv` can be installed on using the standalone installer:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Installation instructions for other methods and Windows can be found [here](https://docs.astral.sh/uv/getting-started/installation/).

`uv` can then be used to install the latest compatible version of python:

```shell
uv python install 3.12
```

`concreteproperties` and it's development dependencies can be installed with:

```shell
uv sync
```

If you want to build the documentation locally, you will need to install `pandoc`. The [installation method](https://pandoc.org/installing.html) depends on what OS you are running.

To run a script using the development virtual environment, you can run:

```shell
uv run example.py
```

Refer to the `uv` [documentation](https://docs.astral.sh/uv/) for more information relating to using `uv` for project management.

## How to test the project

### Pre-commit

[Pre-commit](https://pre-commit.com/) ensures code quality and consistency by running the `ruff` linter and formatter, stripping out execution cells in jupyter notebooks, and running several pre-commit hooks.

These can be run against all files in the project with:

```shell
uv run pre-commit run --all-files
```

However, the best way to ensure code quality is by installing the git pre-commit hook:

```shell
uv run pre-commit install
```

This will run `pre-commit` against all changed files when attempting to `git commit`. You will need to fix the offending files prior to being able to commit a change unless you run `git commit --no-verify`.

### Type Checking

`concreteproperties` uses `pyright` to ensure type-checking where possible. `pyright` can be run on all files with:

```shell
uv run pyright
```

### Tests

The `concreteproperties` tests are located in the `tests/` directory and are written using the [pytest] testing framework. The test suite can be run with:

```shell
uv run pytest
```

[pytest]: https://pytest.readthedocs.io/

### Documentation

You can build the documentation locally with:

```shell
uv run sphinx-build docs docs/_build
```

Make sure that you have a recent version of `pandoc` installed so that the example notebooks can be generated.

Note that all pull requests also build the documentation on Read the Docs, so building the documentation locally is not required.

## How to submit changes

Open a [pull request] to submit changes to this project.

Your pull request needs to meet the following guidelines for acceptance:

- The test suite, pre-commit and pyright checks must pass without errors and warnings.
- Include unit tests. This project aims for a high code coverage.
- If your changes add functionality, update the documentation accordingly.

It is recommended to open an issue before starting work on anything. This will allow a chance to talk it over with the owners and validate your approach.

[pull request]: https://github.com/robbievanleeuwen/concrete-properties/pulls
[code of conduct]: CODE_OF_CONDUCT.md
