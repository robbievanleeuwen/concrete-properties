import matplotlib.pyplot as plt
import concreteproperties.stress_strain_profile as ssp


def generic_stress_strain_plot(render=False):
    """Creates a plot for use in the docstring of class StressStrainProfile class,
    generates a plot with stress-strain parameters shown to aid in interpreting class
    variables.

    :param render: Set to True to plot for testing purposes, note will plot
        automatically in a docstring plot directive when set to default of False
    """
    # create class StressStrainProfile stress-strain profile
    strains = [0, 0.75, 1.5, 2.5, 3.8, 5]
    stresses = [0, 0.6, 0.775, 0.875, 0.97, 1]

    stress_strain_profile = ssp.StressStrainProfile(
        strains=strains,
        stresses=stresses,
    )

    # plot design stress-strain relationship
    ax = stress_strain_profile.plot_stress_strain(
        fmt="-r", render=False, linewidth=1, figsize=(8, 6)
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
    plt.title(label="Generic Material Stress-Strain Profile").set_fontsize(16)
    plt.xlabel("Material Strain $\\varepsilon_m$", labelpad=10).set_fontsize(16)
    plt.ylabel("Material Stress $\sigma_m$", labelpad=10).set_fontsize(16)

    # define data for annotations
    x = strains[2:5]
    y = stresses[2:5]
    x.insert(0, 0)
    y.insert(0, 0)
    x_annotation = [
        "$0$",
        "$\\varepsilon_{m(i-1)}$",
        "$\\varepsilon_{m(i)}$",
        "$\\varepsilon_{m(i+1)}$",
    ]

    y_annotation = [
        "$0$",
        "$\sigma_{m(i-1)}$",
        "$\sigma_{m(i)}$",
        "$\sigma_{m(i+1)}$",
    ]

    # add markers
    plt.plot(strains, stresses, "ok", ms=6)

    # add tick labels for control points
    plt.xticks(x, labels=x_annotation, fontsize=16)
    plt.yticks(y, labels=y_annotation, fontsize=16)

    # set min axes extent
    xmin, xmax, ymin, ymax = plt.axis()
    ax.axes.set_xlim(xmin)
    ax.axes.set_ylim(ymin)

    # add line to each stress and strain point
    for x, y in zip(x, y):
        plt.plot(
            [xmin, x, x],
            [y, y, ymin],
            "k",
            linewidth=0.75,
            dashes=[6, 6],
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
