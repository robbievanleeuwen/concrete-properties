import numpy as np
import matplotlib.pyplot as plt
from concreteproperties.design_codes.nzs3101 import NZS3101
import concreteproperties.stress_strain_profile as ssp
from concreteproperties.material import Concrete
from concreteproperties.stress_strain_profile import ModifiedMander

# this file creates images for docs for modified mander stress-strain profile
# create unconfined concrete
design_code = NZS3101()
compressive_strength = 30
elastic_modulus = design_code.e_conc(compressive_strength)
flexural_tensile_strength = 0.55 * np.sqrt(30)
tensile_failure_strain = flexural_tensile_strength / elastic_modulus
concrete = Concrete(
    name="Mander Concrete Unconfined",
    density=2300,
    stress_strain_profile=ssp.ModifiedMander(
        elastic_modulus=elastic_modulus,
        compressive_strength=compressive_strength,
        tensile_strength=flexural_tensile_strength,
        sect_type="rect",
        conc_tension=True,
        conc_spalling=True,
    ),
    ultimate_stress_strain_profile=ssp.RectangularStressBlock(
        compressive_strength=compressive_strength,
        alpha=design_code.alpha_1(compressive_strength),
        gamma=design_code.beta_1(compressive_strength),
        ultimate_strain=0.003,
    ),
    flexural_tensile_strength=flexural_tensile_strength,
    colour="lightgrey",
)

# plot unconfined stress-strain relationship
ax = concrete.stress_strain_profile.plot_stress_strain(
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
plt.ylabel("Compressive Stength $f_c$", labelpad=0).set_fontsize(16)

# define data for annotations
x = [
    -tensile_failure_strain,
    0.002,
    0.004,
    0.006,
]
y = [
    -flexural_tensile_strength,
    compressive_strength,
    concrete.stress_strain_profile.stresses[-2],
    0,
]
x_annotation = [
    "$\\varepsilon_{t}$",
    "$\\varepsilon_{co}$",
    "$2\\varepsilon_{co}$",
    "$\\varepsilon_{sp}$",
]
y_label = [
    -flexural_tensile_strength,
    compressive_strength,
]
y_annotation = [
    "$f'_t$",
    "$f'_{co}$",
]
xy_list = zip(x, y)

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
    [-flexural_tensile_strength, -flexural_tensile_strength, ymin],
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
    [concrete.stress_strain_profile.stresses[-2], ymin],
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
    concrete.stress_strain_profile.strains,
    concrete.stress_strain_profile.stresses,
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

# save plot
plt.savefig(
    ".\docs\source\_static\mander_unconfined.png",
    dpi="figure",
    format="png",
    pad_inches=0.1,
    facecolor="auto",
    edgecolor="auto",
)

# create confined concrete
compressive_strength = 30
elastic_modulus = design_code.e_conc(compressive_strength)
flexural_tensile_strength = 0.55 * np.sqrt(30)
tensile_failure_strain = flexural_tensile_strength / elastic_modulus
concrete = Concrete(
    name="Mander Concrete Confined",
    density=2300,
    stress_strain_profile=ssp.ModifiedMander(
        elastic_modulus=elastic_modulus,
        compressive_strength=compressive_strength,
        tensile_strength=flexural_tensile_strength,
        sect_type="rect",
        conc_confined=True,
        conc_tension=True,
        d=800,
        b=500,
        long_reinf_area=12 * 314,
        w_dash=[150] * 12,
        cvr=30 + 10,
        trans_spacing=125,
        trans_d_b=10,
        trans_num_d=4,
        trans_num_b=4,
        trans_f_y=500,
        eps_su=0.15,
    ),
    ultimate_stress_strain_profile=ssp.RectangularStressBlock(
        compressive_strength=compressive_strength,
        alpha=design_code.alpha_1(compressive_strength),
        gamma=design_code.beta_1(compressive_strength),
        ultimate_strain=0.003,
    ),
    flexural_tensile_strength=flexural_tensile_strength,
    colour="lightgrey",
)

# add return of stress-strain diagram to zero stress
concrete.stress_strain_profile.stresses = np.append(
    concrete.stress_strain_profile.stresses, 0
)
concrete.stress_strain_profile.strains = np.append(
    concrete.stress_strain_profile.strains, concrete.stress_strain_profile.strains[-1]
)

# plot unconfined stress-strain relationship
ax = concrete.stress_strain_profile.plot_stress_strain(
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
plt.ylabel("Compressive Stength $f_c$", labelpad=0).set_fontsize(16)

# define data for annotations
f_cc = max(concrete.stress_strain_profile.stresses)
eps_cc = concrete.stress_strain_profile.strains[
    np.argmax(concrete.stress_strain_profile.stresses)
]
eps_cu = max(concrete.stress_strain_profile.strains)
x = [
    -tensile_failure_strain,
    eps_cc,
    max(concrete.stress_strain_profile.strains),
]
y = [
    -flexural_tensile_strength,
    f_cc,
    concrete.stress_strain_profile.stresses[-2],
]
x_annotation = [
    "$\\varepsilon_{t}$",
    "$\\varepsilon_{cc}$",
    "$\\varepsilon_{cu}$",
]
y_label = [
    -flexural_tensile_strength,
    f_cc,
]
y_annotation = [
    "$f'_t$",
    "$f'_{cc}$",
]
xy_list = zip(x, y)

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
    [-flexural_tensile_strength, -flexural_tensile_strength, ymin],
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
    concrete.stress_strain_profile.strains,
    concrete.stress_strain_profile.stresses,
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

# save plot
plt.savefig(
    ".\docs\source\_static\mander_confined.png",
    dpi="figure",
    format="png",
    pad_inches=0.1,
    facecolor="auto",
    edgecolor="auto",
)
