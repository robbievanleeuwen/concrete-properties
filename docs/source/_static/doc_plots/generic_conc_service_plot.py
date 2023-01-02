import matplotlib.pyplot as plt
import concreteproperties.stress_strain_profile as ssp


def generic_conc_service_plot(render=False):
    """Creates a plot for use in the docstring of ConcreteServiceProfile class,
    generates a plot with stress-strain parameters shown to aid in interpreting class
    variables.

    :param render: Set to True to plot for testing purposes, note will plot
        automatically in a docstring plot directive when set to default of False
    """
    # create ConcreteServiceProfile stress-strain profile
    compressive_strength = 40
    ultimate_strain = 0.0035

    strains = [
        -0.00015,
        -0.00010,
        -0.000075,
        0,
        0.00045,
        0.001,
        0.0015,
        0.002,
        ultimate_strain,
    ]
    stresses = [
        0,
        0,
        -0.1 * compressive_strength,
        0,
        0.6 * compressive_strength,
        0.8 * compressive_strength,
        0.925 * compressive_strength,
        compressive_strength,
        compressive_strength,
    ]

    stress_strain_profile = ssp.ConcreteServiceProfile(
        # compressive_strength=compressive_strength,
        strains=strains,
        stresses=stresses,
        ultimate_strain=ultimate_strain,
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
    plt.title(label="Generic Service Stress-Strain Profile").set_fontsize(16)
    plt.xlabel("Concrete Strain $\\varepsilon_c$", labelpad=10).set_fontsize(16)
    plt.ylabel("Concrete Stress $\sigma_c$", labelpad=10).set_fontsize(16)

    # define data for annotations
    x = strains[4:7]
    y = stresses[4:7]
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
