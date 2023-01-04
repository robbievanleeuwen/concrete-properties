import ec2_bilinear_ultimate_plot as ec2_bi
import ec2_parabolic_ultimate_plot as ec2_par
import generic_bilinear_ultimate_plot as gen_bi
import generic_conc_service_plot as gen_ser
import generic_conc_ultimate_plot as gen_ult
import generic_linear_service_plot as gen_lin
import generic_parabolic_ultimate_plot as gen_par
import generic_rect_ultimate_plot as gen_rec
import generic_stress_strain_plot as gen
import mander_confined_plot as man_c
import mander_unconfined_plot as man_uc


def main():
    """Function to show all docs plots.

    Add other plot methods as they are developed/added.

    """
    # display all concrete plots
    ec2_bi.ec2_bilinear_ultimate_plot()
    ec2_par.ec2_parabolic_ultimate_plot()
    gen_bi.generic_bilinear_ultimate_plot()
    gen_ser.generic_conc_service_plot()
    gen_ult.generic_conc_ultimate_plot()
    gen_lin.generic_linear_service_plot()
    gen_par.generic_parabolic_ultimate_plot()
    gen_rec.generic_rect_ultimate_plot()
    gen.generic_stress_strain_plot()
    man_c.mander_confined_plot()
    man_uc.mander_unconfined_plot(True)


if __name__ == "__main__":
    main()
