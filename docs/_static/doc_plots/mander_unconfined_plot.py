"""Plotting function to generate the mander confined plot."""

import matplotlib.pyplot as plt

import concreteproperties.stress_strain_profile as ssp
from concreteproperties.design_codes.nzs3101 import NZS3101


def mander_unconfined_plot(render=False):
    """Plots the confined mander plot.

    Creates a plot for use in the docstring of ModifiedMander class for unconfined
    concrete to generate a plot with stress-strain parameters shown to aid in
    interpreting class variables.

    Args:
        render: Set to True to plot for testing purposes, note will plot
            automatically in a docstring plot directive when set to default of False
    """
    # create unconfined ModifiedMander stress-strain profile
    design_code = NZS3101()
    compressive_strength = 30
    elastic_modulus = design_code.e_conc(compressive_strength)
    concrete_tensile_strength = design_code.concrete_tensile_strength(
        compressive_strength
    )
    tensile_failure_strain = concrete_tensile_strength / elastic_modulus
    stress_strain_profile = ssp.ModifiedMander(
        elastic_modulus=elastic_modulus,
        compressive_strength=compressive_strength,
        tensile_strength=concrete_tensile_strength,
        sect_type="rect",
        conc_tension=True,
        conc_spalling=True,
    )

    # plot unconfined stress-strain relationship
    ax = stress_strain_profile.plot_stress_strain(
        fmt="-k", render=False, linewidth=1, figsize=(8, 6)
    )

    # add fake origin axes lines
    plt.axvline(linewidth=1.5, color="grey")
    plt.axhline(linewidth=1.5, color="grey")

    # add arrows at end of origin axes
    ax.plot(
        (1),
        (0),
        ls="",
        marker=">",
        ms=10,
        color="k",
        transform=ax.get_yaxis_transform(),
        clip_on=False,
    )
    ax.plot(
        (0),
        (1),
        ls="",
        marker="^",
        ms=10,
        color="k",
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )

    # add title and axes labels
    plt.title(label="Modified Mander Unconfined Stress-Strain Profile").set_fontsize(16)
    plt.xlabel("Compressive Strain $\\varepsilon_c$", labelpad=10).set_fontsize(16)
    plt.ylabel("Compressive Strength $f_c$", labelpad=0).set_fontsize(16)

    # define data for annotations
    x = [
        -tensile_failure_strain,
        0.002,
        0.004,
        0.006,
    ]
    y = [
        -concrete_tensile_strength,
        compressive_strength,
        stress_strain_profile.stresses[-3],
        0,
    ]
    x_annotation = [
        "$\\varepsilon_{t}$",
        "$\\varepsilon_{co}$",
        "$2\\varepsilon_{co}$",
        "$\\varepsilon_{sp}$",
    ]
    y_label = [
        -concrete_tensile_strength,
        compressive_strength,
    ]
    y_annotation = [
        "$f'_t$",
        "$f'_{co}$",
    ]

    # add markers
    plt.plot(x, y, "ok", ms=6)

    # add tick labels for control points
    plt.xticks(x, labels=x_annotation, fontsize=16)
    plt.yticks(y_label, labels=y_annotation, fontsize=16)

    # set min axes extent
    xmin, xmax, ymin, ymax = plt.axis()
    ax.axes.set_xlim(xmin)
    ax.axes.set_ylim(ymin)

    # add line to maximum tension strength f_t at esp_t
    plt.plot(
        [xmin, -tensile_failure_strain, -tensile_failure_strain],
        [-concrete_tensile_strength, -concrete_tensile_strength, ymin],
        "k",
        linewidth=0.75,
        dashes=[6, 6],
    )

    # add line to maximum strength f_co at esp_co
    plt.plot(
        [xmin, 0.002, 0.002],
        [compressive_strength, compressive_strength, ymin],
        "k",
        linewidth=0.75,
        dashes=[6, 6],
    )

    # add line to end of non-linear curve at 2*eps_co
    plt.plot(
        [0.004, 0.004],
        [stress_strain_profile.stresses[-3], ymin],
        "k",
        linewidth=0.75,
        dashes=[6, 6],
    )

    # add line at eps_sp
    plt.plot(
        [0.006, 0.006],
        [0, ymin],
        "k",
        linewidth=0.75,
        dashes=[6, 6],
    )

    # add fill
    plt.fill_between(
        stress_strain_profile.strains,
        stress_strain_profile.stresses,
        0,
        alpha=0.15,
        color="grey",
    )

    # Turn off grid
    plt.grid(False)

    # turn off plot border
    plt.box(False)

    # turn on tight layout
    plt.tight_layout()

    # plot if required for testing purposes
    if render:
        plt.show()
