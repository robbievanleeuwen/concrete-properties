.. _label-results:

Results
=======

After performing an analysis on a reinforced concrete cross-section (see
:ref:`label-analysis`), *concreteproperties* provides the user with a results object
specific to the conducted analysis. These results objects have methods tailored for the
post-processing of analysis results.

Gross Area Properties
---------------------

Gross area properties can be retrieved by calling the
:meth:`~concreteproperties.concrete_section.ConcreteSection.get_gross_properties`
method.

..  automethod:: concreteproperties.concrete_section.ConcreteSection.get_gross_properties
  :noindex:

This method returns a :class:`~concreteproperties.results.ConcreteProperties` object,
which stores all the calculated section properties as class attributes. The gross area
properties can be printed to the terminal by calling the
:meth:`~concreteproperties.results.ConcreteProperties.print_results` method.

..  autoclass:: concreteproperties.results.ConcreteProperties()
  :noindex:
  :members:

Transformed gross area properties can be obtained by calling the
:meth:`~concreteproperties.concrete_section.ConcreteSection.get_transformed_gross_properties`
method.

..  automethod:: concreteproperties.concrete_section.ConcreteSection.get_transformed_gross_properties
  :noindex:

This method returns a :class:`~concreteproperties.results.TransformedConcreteProperties`
object, which stores all the calculated transformed section properties as class
attributes. The transformed gross area properties can be printed to the terminal by
calling the
:meth:`~concreteproperties.results.TransformedConcreteProperties.print_results` method.

..  autoclass:: concreteproperties.results.TransformedConcreteProperties()
  :noindex:
  :members:

.. seealso::
  For an application of the above, see the example
  :ref:`/notebooks/area_properties.ipynb`.


Cracked Area Properties
-----------------------

Performing a cracked analysis with
:meth:`~concreteproperties.concrete_section.ConcreteSection.calculate_cracked_properties`
returns a :class:`~concreteproperties.results.CrackedResults` object.

..  autoclass:: concreteproperties.results.CrackedResults()
  :noindex:
  :members:

Calling
:meth:`~concreteproperties.results.TransformedConcreteProperties.calculate_transformed_properties`
on a :class:`~concreteproperties.results.CrackedResults` object stores the transformed
cracked properties as attributes within the current object.

.. seealso::
  For an application of the above, see the example
  :ref:`/notebooks/cracked_properties.ipynb`.


Moment Curvature Analysis
-------------------------

A :meth:`~concreteproperties.concrete_section.ConcreteSection.moment_curvature_analysis`
returns a :class:`~concreteproperties.results.MomentCurvatureResults` object. This
object can be used to plot moment curvature results.

..  autoclass:: concreteproperties.results.MomentCurvatureResults()
  :noindex:
  :members: plot_results, plot_multiple_results

.. seealso::
  For an application of the above, see the example
  :ref:`/notebooks/moment_curvature.ipynb`.
