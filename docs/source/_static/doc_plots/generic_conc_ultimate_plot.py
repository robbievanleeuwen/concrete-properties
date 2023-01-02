import matplotlib.pyplot as plt
import concreteproperties.stress_strain_profile as ssp


def generic_conc_ultimate_plot(render=False):
    """Creates a plot for use in the docstring of ConcreteUltimateProfile class,
    generates a plot with stress-strain parameters shown to aid in interpreting class
    variables.

    :param render: Set to True to plot for testing purposes, note will plot
        automatically in a docstring plot directive when set to default of False
    """
    # create ConcreteUltimateProfile stress-strain profile
    compressive_strength = 40
    ultimate_strain = 0.0035

    strains = [0, 0.0005, 0.001, 0.002, 0.00275, ultimate_strain]
    stresses = [
        0,
        0,
        0.5 * compressive_strength,
        0.8 * compressive_strength,
        0.95 * compressive_strength,
        compressive_strength,
    ]

    stress_strain_profile = ssp.ConcreteUltimateProfile(
        compressive_strength=compressive_strength,
        strains=strains,
        stresses=stresses,
    )

    # add return of stress-strain diagram to zero stress
    stress_strain_profile.stresses.append(0)
    stress_strain_profile.strains.append(stress_strain_profile.strains[-1])

    # plot design stress-strain relationship
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
    plt.title(label="Generic Ultimate Stress-Strain Profile").set_fontsize(16)
    plt.xlabel("Concrete Strain $\\varepsilon_c$", labelpad=10).set_fontsize(16)
    plt.ylabel("Concrete Stress $\sigma_c$", labelpad=10).set_fontsize(16)

    # define data for annotations
    x = strains[2:5]
    y = stresses[2:5]
    x.insert(0, 0)
    x.append(ultimate_strain)
    y.insert(0, 0)
    y.append(compressive_strength)

    x_annotation = [
        "$0$",
        "$\\varepsilon_{c(i-1)}$",
        "$\\varepsilon_{c(i)}$",
        "$\\varepsilon_{c(i+1)}$",
        "$\\varepsilon_{u1}$",
    ]

    y_annotation = [
        "$0$",
        "$\sigma_{c(i-1)}$",
        "$\sigma_{c(i)}$",
        "$\sigma_{c(i+1)}$",
        "$f'_c$",
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
