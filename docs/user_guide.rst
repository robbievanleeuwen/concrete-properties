User Guide
==========

The ``concreteproperties`` user guide lists the assumptions adopted in
``concreteproperties`` and provides an overview of each step in the
``concreteproperties`` workflow.

.. toctree::
    :caption: Contents
    :maxdepth: 1

    user_guide/materials
    user_guide/geometry
    user_guide/analysis
    user_guide/prestressed_analysis
    user_guide/results
    user_guide/design_codes
    user_guide/assumptions


.. _label-features:

Features
--------

Analysis Types
^^^^^^^^^^^^^^

* ☑ Reinforced Concrete
* ☑ Steel-Concrete Composite
* ☑ Prestressed Concrete

Material Properties
^^^^^^^^^^^^^^^^^^^

* ☑ Concrete material

  * ☑ Service stress-strain profiles

    * ☑ Linear profile
    * ☑ Linear profile (no tension)
    * ☑ Eurocode non-linear
    * ☑ Modified Mander non-linear profile (confined & unconfined concrete)

  * ☑ Ultimate stress-strain profiles

    * ☑ Rectangular stress block
    * ☑ Bilinear stress-strain profile
    * ☑ Eurocode parabolic

  * ☑ Flexural tensile strength

* ☑ Steel material

  * ☑ Stress-strain profiles

    * ☑ Elastic-plastic
    * ☑ Elastic-plastic (with hardening)

* ☑ Strand material

  * ☑ Stress-strain profiles

    * ☑ Elastic-plastic (with hardening)
    * ☑ PCI journal (1992) non-linear

Gross Section Properties
^^^^^^^^^^^^^^^^^^^^^^^^

* ☑ Cross-sectional areas (total, concrete, steel, strand)
* ☑ Axial rigidity
* ☑ Cross-section mass
* ☑ Cross-section perimeter
* ☑ First moments of area
* ☑ Elastic centroid
* ☑ Global second moments of area
* ☑ Centroidal second moments of area
* ☑ Principal axis angle
* ☑ Principal second moments of area
* ☑ Centroidal section moduli
* ☑ Principal section moduli
* ☑ Prestressed Aations

Service Analysis
^^^^^^^^^^^^^^^^

* ☑ Cracking moment
* ☑ Cracked area properties
* ☑ Moment-curvature diagram

Ultimate Analysis
^^^^^^^^^^^^^^^^^

* ☑ Ultimate bending capacity
* ☑ Squash load
* ☑ Tensile load
* ☑ Moment interaction diagrams

  * ☑ M-N curves
  * ☑ Biaxial bending curve

Stress Analysis
^^^^^^^^^^^^^^^

* ☑ Uncracked stresses
* ☑ Cracked stresses
* ☑ Service stresses
* ☑ Ultimate stresses

Design Codes
^^^^^^^^^^^^

* ☑ Design code modules

  * ☑ AS3600
  * ☐ AS5100
  * ☑ NZS3101 & NZSEE C5 Assessment Guidelines
