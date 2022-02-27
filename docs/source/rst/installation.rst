Installation
============

These instructions will get you a copy of *concreteproperties* up and running on
your local machine. You will need a working copy of python 3.7, 3.8 or 3.9 on your
machine.

Installing *concreteproperties*
-------------------------------

*concreteproperties* uses `sectionproperties <https://github.com/robbievanleeuwen/section-properties>`_
to generate a reinforced concrete cross-section and conduct various analyses.

*concreteproperties* and all of its dependencies can be installed through the python
package index::

  $ pip install concreteproperties

Testing the Installation
------------------------

Python *pytest* modules are located in the *concreteproperties.tests* package.
To see if your installation is working correctly, install `pytest` and run the
following test::

  $ pytest --pyargs concreteproperties
