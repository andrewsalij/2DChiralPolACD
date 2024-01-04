import albano_params as ap
import python_util

otp_cis = python_util.open_object_file("otp_cis_solvent.pkl")
otp_trans = python_util.open_object_file("otp_trans_solvent.pkl")

print("Oligothiophene cis info:")
otp_cis.print_info()

print("Oligothiophene trans info:")
otp_trans.print_info()

