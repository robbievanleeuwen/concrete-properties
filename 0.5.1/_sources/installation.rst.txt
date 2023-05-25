Installation
============

These instructions will get you a copy of *concreteproperties* up and running on
your local machine. You will need a working copy of python 3.8, 3.9 or 3.10 on your
machine.

Installing *concreteproperties*
-------------------------------

*concreteproperties* uses `sectionproperties <https://github.com/robbievanleeuwen/section-properties>`_
to generate a reinforced concrete geometry.

*concreteproperties* and all of its dependencies can be installed through the python
package index:

.. code-block:: console

  pip install concreteproperties

Testing the Installation
------------------------

Python *pytest* modules are located in the *concreteproperties.tests* package.
To see if your installation is working correctly, install `pytest` and run the
following test:

.. code-block:: console

  pytest --pyargs concreteproperties
