import matplotlib.pyplot as plt
import concreteproperties.stress_strain_profile as ssp


def generic_linear_no_tension_service_plot(render=False):
    """Creates a plot for use in the docstring of ConcreteLinearNoTension class,
    generates a plot with stress-strain parameters shown to aid in
    interpreting class variables.

    :param render: Set to True to plot for testing purposes, note will plot
        automatically in a docstring plot directive when set to default of False
    """
    # create ConcreteLinearNoTension stress-strain profile
    elastic_modulus = 35000
    compressive_strength = 40
    ultimate_strain = 0.0035
    yield_strain = compressive_strength / elastic_modulus

    stress_strain_profile = ssp.ConcreteLinearNoTension(
        elastic_modulus=elastic_modulus,
        compressive_strength=compressive_strength,
        ultimate_strain=ultimate_strain,
    )

    # overide default tension branch strain
    stress_strain_profile.strains[0] = 0

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
    plt.title(label="Generic Linear No Tension Stress-Strain Profile").set_fontsize(16)
    plt.xlabel("Concrete Strain $\\varepsilon_c$", labelpad=10).set_fontsize(16)
    plt.ylabel("Concrete Stress $\sigma_c$", labelpad=10).set_fontsize(16)

    # define data for annotations
    x = [
        0,
        0.33 * compressive_strength / elastic_modulus,
        0.67 * compressive_strength / elastic_modulus,
        yield_strain,
        ultimate_strain,
    ]
    y = [
        0,
        0.33 * compressive_strength,
        0.67 * compressive_strength,
        compressive_strength,
        compressive_strength,
    ]
    x_annotation = [
        "$0$",
        "$\\varepsilon_{c(i)}$",
        "$\\varepsilon_{c(i+1)}$",
        "$\\varepsilon_{y1}$",
        "$\\varepsilon_{u1}$",
    ]
    y_label = [
        0,
        0.33 * compressive_strength,
        0.67 * compressive_strength,
        compressive_strength,
    ]
    y_annotation = [
        "$0$",
        "$\sigma_{c(i)}$",
        "$\sigma_{c(i+1)}$",
        "$f'_c$",
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

    # add line to each stress and strain point
    for x, y in zip(x, y_label):
        plt.plot(
            [xmin, x, x],
            [y, y, ymin],
            "k",
            linewidth=0.75,
            dashes=[6, 6],
        )

    # add line to maximum strength f_c at eps_u1
    plt.plot(
        [ultimate_strain, ultimate_strain],
        [compressive_strength, ymin],
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
