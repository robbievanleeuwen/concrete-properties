"""Plotting function to generate the mander confined plot."""

import matplotlib.pyplot as plt
import numpy as np

import concreteproperties.stress_strain_profile as ssp
from concreteproperties.design_codes.nzs3101 import NZS3101


def mander_confined_plot(render=False):
    """Plots the confined mander plot.

    Creates a plot for use in the docstring of ModifiedMander class for unconfined
    concrete to generate a plot with stress-strain parameters shown to aid in
    interpreting class variables.

    Args:
        render: Set to True to plot for testing purposes, note will plot
            automatically in a docstring plot directive when set to default of False
    """
    # create confined ModifiedMander stress-strain profile
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
        conc_confined=True,
        conc_tension=True,
        d=800,
        b=500,
        long_reinf_area=12 * 314,
        w_dash=[150.0] * 12,
        cvr=30 + 10,
        trans_spacing=125,
        trans_d_b=10,
        trans_num_d=4,
        trans_num_b=4,
        trans_f_y=500,
        eps_su=0.15,
    )

    # add return of stress-strain diagram to zero stress
    stress_strain_profile.stresses = np.append(stress_strain_profile.stresses, 0)
    stress_strain_profile.strains = np.append(
        stress_strain_profile.strains,
        stress_strain_profile.strains[-1],
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
    plt.title(label="Modified Mander Confined Stress-Strain Profile").set_fontsize(16)
    plt.xlabel("Compressive Strain $\\varepsilon_c$", labelpad=10).set_fontsize(16)
    plt.ylabel("Compressive Strength $f_c$", labelpad=0).set_fontsize(16)

    # define data for annotations
    f_cc = max(stress_strain_profile.stresses)
    eps_cc = stress_strain_profile.strains[np.argmax(stress_strain_profile.stresses)]
    eps_cu = max(stress_strain_profile.strains)
    x = [
        -tensile_failure_strain,
        eps_cc,
        max(stress_strain_profile.strains),
    ]
    y = [
        -concrete_tensile_strength,
        f_cc,
        stress_strain_profile.stresses[-2],
    ]
    x_annotation = [
        "$\\varepsilon_{t}$",
        "$\\varepsilon_{cc}$",
        "$\\varepsilon_{cu}$",
    ]
    y_label = [
        -concrete_tensile_strength,
        f_cc,
    ]
    y_annotation = [
        "$f'_t$",
        "$f'_{cc}$",
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

    # add line to maximum strength f_cc at esp_cc
    plt.plot(
        [xmin, eps_cc, eps_cc],
        [f_cc, f_cc, ymin],
        "k",
        linewidth=0.75,
        dashes=[6, 6],
    )

    # add line at eps_cu
    plt.plot(
        [eps_cu, eps_cu],
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
