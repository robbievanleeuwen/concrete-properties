import numpy as np
import sectionproperties.pre.library.primitive_sections as sp_ps


def add_bar(geometry, area, material, x, y, n=4):
    bar = sp_ps.circular_section_by_area(
        area=area, n=n, material=material
    ).shift_section(x_offset=x, y_offset=y)

    return (geometry - bar) + bar


def add_bar_rectangular_array(
    geometry,
    area,
    material,
    n_x,
    x_s,
    n_y=1,
    y_s=0,
    anchor=(0, 0),
    exterior_only=False,
    n=4,
):
    for j_idx in range(n_y):
        for i_idx in range(n_x):
            # check to see if we are adding a bar
            if exterior_only:
                if i_idx != 0 and i_idx != n_x - 1 and j_idx != 0 and j_idx != n_y - 1:
                    add_bar = False
                else:
                    add_bar = True
            else:
                add_bar = True

            if add_bar:
                bar = sp_ps.circular_section_by_area(area=area, n=n, material=material)
                x = anchor[0] + i_idx * x_s
                y = anchor[1] + j_idx * y_s
                bar = bar.shift_section(x_offset=x, y_offset=y)
                geometry = (geometry - bar) + bar

    return geometry


def add_bar_circular_array(
    geometry, area, material, n_bar, r_array, theta_0=0, ctr=(0, 0), n=4
):
    d_theta = 2 * np.pi / n_bar

    for idx in range(n_bar):
        bar = sp_ps.circular_section_by_area(area=area, n=n, material=material)
        theta = theta_0 + idx * d_theta
        x = ctr[0] + r_array * np.cos(theta)
        y = ctr[1] + r_array * np.sin(theta)
        bar = bar.shift_section(x_offset=x, y_offset=y)
        geometry = (geometry - bar) + bar

    return geometry
