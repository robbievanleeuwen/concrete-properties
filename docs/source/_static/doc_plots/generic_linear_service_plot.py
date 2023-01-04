import matplotlib.pyplot as plt
import concreteproperties.stress_strain_profile as ssp


def generic_linear_service_plot(render=False):
    """Creates a plot for use in the docstring of ConcreteLinear class,
    generates a plot with stress-strain parameters shown to aid in
    interpreting class variables.

    :param render: Set to True to plot for testing purposes, note will plot
        automatically in a docstring plot directive when set to default of False
    """
    # create ConcreteLinear stress-strain profile
    elastic_modulus = 1

    stress_strain_profile = ssp.ConcreteLinear(
        elastic_modulus=elastic_modulus,
    )

    # add return of stress-strain diagram to zero stress at max tension/compression
    stress_strain_profile.stresses.append(0)
    stress_strain_profile.strains.append(stress_strain_profile.strains[-1])
    stress_strain_profile.stresses.insert(0, 0)
    stress_strain_profile.strains.insert(0, stress_strain_profile.strains[0])

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
    plt.title(
        label="Generic Linear Stress-Strain Profile",
        pad=20,
    ).set_fontsize(16)
    plt.xlabel("Concrete Strain $\\varepsilon_c$", labelpad=10).set_fontsize(16)
    plt.ylabel("Concrete Stress $\sigma_c$", labelpad=10).set_fontsize(16)

    # define data for annotations
    x = [-0.001, 0.00033, 0.00067, 0.001]
    y = [-0.001, 0.00033, 0.00067, 0.001]

    # add markers
    plt.plot(x, y, "ok", ms=6)

    # labels
    x_label = [
        0.00033,
        0.00067,
    ]
    y_label = [
        0.00033,
        0.00067,
    ]

    x_annotation = [
        "$\\varepsilon_{c(i)}$",
        "$\\varepsilon_{c(i+1)}$",
    ]
    y_annotation = [
        "$\sigma_{c(i)}$",
        "$\sigma_{c(i+1)}$",
    ]

    # add tick labels for control points
    plt.xticks(x_label, labels=x_annotation, fontsize=16)
    plt.yticks(y_label, labels=y_annotation, fontsize=16)

    # set min axes extent
    xmin, xmax, ymin, ymax = plt.axis()
    ax.axes.set_xlim(xmin)
    ax.axes.set_ylim(ymin)

    # add line to each stress and strain point
    for x, y in zip(x_label, y_label):
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
