import matplotlib.pyplot as plt
import concreteproperties.stress_strain_profile as ssp


def generic_parabolic_ultimate_plot(render=False):
    """Creates a plot for use in the docstring of ParabolicStressStrain class,
    generates a plot with stress-strain parameters shown to aid in interpreting class
    variables.

    :param render: Set to True to plot for testing purposes, note will plot
        automatically in a docstring plot directive when set to default of False
    """
    # create ParabolicStressStrain stress-strain profile
    compressive_strength = 40
    compressive_strain = 0.0025
    ultimate_strain = 0.0035
    n_points = 50

    stress_strain_profile = ssp.ParabolicStressStrain(
        compressive_strength=compressive_strength,
        compressive_strain=compressive_strain,
        ultimate_strain=ultimate_strain,
        n_exp=2,
        n_points=n_points,
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
    plt.title(label="Generic Parabolic Stress-Strain Profile").set_fontsize(16)
    plt.xlabel("Concrete Strain $\\varepsilon_c$", labelpad=10).set_fontsize(16)
    plt.ylabel("Concrete Stress $\sigma_c$", labelpad=10).set_fontsize(16)

    # define data for annotations
    x = [
        0,
        compressive_strain,
        ultimate_strain,
    ]
    y = [
        0,
        compressive_strength,
        compressive_strength,
    ]
    x_annotation = [
        "$0$",
        "$\\varepsilon_{1}$",
        "$\\varepsilon_{u1}$",
    ]
    y_label = [
        0,
        compressive_strength,
    ]
    y_annotation = [
        "$0$",
        "$\\alpha f'_c$",
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

    # add line to maximum strength at eps_1
    plt.plot(
        [xmin, compressive_strain, compressive_strain],
        [compressive_strength, compressive_strength, ymin],
        "k",
        linewidth=0.75,
        dashes=[6, 6],
    )

    # add line to maximum strength f_cd at eps_u1
    plt.plot(
        [xmin, ultimate_strain, ultimate_strain],
        [compressive_strength, compressive_strength, ymin],
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
