import matplotlib.pyplot as plt
import concreteproperties.stress_strain_profile as ssp


def generic_rect_ultimate_plot(render=False):
    """Creates a plot for use in the docstring of RectangularStressBlock class,
    generates a plot with stress-strain parameters shown to aid in
    interpreting class variables.

    :param render: Set to True to plot for testing purposes, note will plot
        automatically in a docstring plot directive when set to default of False
    """
    # create RectangularStressBlock stress-strain profile
    compressive_strength = 50
    alpha = 0.85
    gamma = 0.69
    ultimate_strain = 0.003
    f_c = compressive_strength
    alpha_f_c = alpha * f_c

    stress_strain_profile = ssp.RectangularStressBlock(
        compressive_strength=compressive_strength,
        alpha=alpha,
        gamma=gamma,
        ultimate_strain=ultimate_strain,
    )

    # create nominal curve stress & strain values
    nom_strains = [
        stress_strain_profile.strains[2],
        stress_strain_profile.strains[2],
        stress_strain_profile.strains[3],
        stress_strain_profile.strains[3],
    ]
    nom_stresses = [
        stress_strain_profile.stresses[2],
        stress_strain_profile.stresses[2] / alpha,
        stress_strain_profile.stresses[3] / alpha,
        stress_strain_profile.stresses[3],
    ]

    # overide default tension branch strain
    stress_strain_profile.strains[0] = 0

    # add return of stress-strain diagram to zero stress
    stress_strain_profile.stresses.append(0)
    stress_strain_profile.strains.append(stress_strain_profile.strains[-1])

    # plot design stress-strain relationship
    ax = stress_strain_profile.plot_stress_strain(
        fmt="-k", render=False, linewidth=1, figsize=(8, 6)
    )

    # plot nominal stress-strain relationship
    ax.plot(
        nom_strains,
        nom_stresses,
        color="k",
        lw=1.25,
        ls="--",
        dashes=[12, 6],
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
    plt.title(label="Rectangular Stress-Strain Profile").set_fontsize(16)
    plt.xlabel("Concrete Strain $\\varepsilon_c$", labelpad=10).set_fontsize(16)
    plt.ylabel("Concrete Stress $\sigma_c$", labelpad=10).set_fontsize(16)

    # define data for annotations
    eps_1, eps_u1 = ultimate_strain * (1 - gamma), ultimate_strain
    x = [
        0,
        eps_1,
        eps_1,
        eps_u1,
        eps_1,
        eps_u1,
    ]
    y = [
        0,
        0,
        f_c,
        f_c,
        alpha_f_c,
        alpha_f_c,
    ]
    x_annotation = [
        "$0$",
        "",
        "",
        "",
        "$\\varepsilon_{1}$",
        "$\\varepsilon_{u1}$",
    ]
    y_label = [
        0,
        alpha_f_c,
        f_c,
    ]
    y_annotation = [
        "$0$",
        "$\\alpha f'_{c}$",
        "$f'_{c}$",
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

    # add line to maximum strength alpha*f'_c at eps_1
    plt.plot(
        [xmin, eps_1, eps_1],
        [alpha_f_c, alpha_f_c, ymin],
        "k",
        linewidth=0.75,
        dashes=[6, 6],
    )

    # add line to maximum strength f'_c at eps_1
    plt.plot(
        [xmin, eps_1],
        [f_c, f_c],
        "k",
        linewidth=0.75,
        dashes=[6, 6],
    )

    # add line to maximum strength alpha*f'_c at eps_u1
    plt.plot(
        [xmin, eps_u1, eps_u1],
        [alpha_f_c, alpha_f_c, ymin],
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
