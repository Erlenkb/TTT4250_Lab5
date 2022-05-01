import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat

# import numpy.random.standard_t as t
x_axis_pta = [500, 1000, 2000, 4000, 8000]
x_axis_newt = [125, 250, 500, 1000, 2000, 4000, 8000]
x_ticks_pta = ["500", "1k", "2k", "4k", "8k"]
x_ticks_newt = ["125", "250", "500", "1k", "2k", "4k", "8k"]

molded_earplugs_person8 = [13, 24, 33, 33, 41, 31.5, 39]

filename_pta = "pta.csv"
filename_newt = "newt.csv"


def read_csv(filename):
    return np.genfromtxt(filename, skip_header=1, delimiter=';')


def plått_simple_short(array, title, name_of_file):
    fig, ax = plt.subplots()
    ax.semilogx(x_axis_pta, array)
    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title(title)
    ax.set_xticks(x_axis_pta)
    ax.set_xticklabels(x_ticks_pta)
    plt.savefig(name_of_file)
    plt.show()


def plått_simple_long_unc(array, title, name_of_file, exp_unc):
    fig, ax = plt.subplots()
    ax.semilogx(x_axis_newt, array, marker="o")
    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title(title)
    ax.set_xticks(x_axis_newt)
    ax.set_xticklabels(x_ticks_newt)
    plt.errorbar(x_axis_newt, array, yerr=exp_unc)
    plt.savefig(name_of_file)
    plt.show()


def plått_simple_long(array, title, name_of_file):
    fig, ax = plt.subplots()
    ax.semilogx(x_axis_newt, array, marker="o")
    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title(title)
    ax.set_xticks(x_axis_newt)
    ax.set_xticklabels(x_ticks_newt)
    plt.savefig(name_of_file)
    plt.show()


def plått_multi_short(n_m_array, title, name_of_file, lengend_array):
    assert len(n_m_array.shape) == 2  # hvis denne failer har du en funky array
    fig, ax = plt.subplots()

    for i in range(n_m_array.shape[0]):
        ax.semilogx(x_axis_pta, n_m_array[i],
                    label=f"{lengend_array[i]}", marker="o")

    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title(title)
    ax.set_xticks(x_axis_pta)
    ax.set_xticklabels(x_ticks_pta)
    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    plt.savefig(name_of_file)
    plt.show()


def plått_multi_long(n_m_array, title, name_of_file, lengend_array):
    assert len(n_m_array.shape) == 2  # hvis denne failer har du en funky array
    fig, ax = plt.subplots()

    for i in range(n_m_array.shape[0]):
        ax.semilogx(x_axis_newt, n_m_array[i],
                    label=f"{lengend_array[i]}", marker="o")

    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title(title)
    ax.set_xticks(x_axis_newt)
    ax.set_xticklabels(x_ticks_newt)
    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    plt.savefig(name_of_file)
    plt.show()


def plått_multi_long_unc(n_m_array, title, name_of_file, lengend_array, exp_unc, average):
    assert len(n_m_array.shape) == 2  # hvis denne failer har du en funky array
    fig, ax = plt.subplots()

    for i in range(n_m_array.shape[0]):
        ax.semilogx(x_axis_newt, n_m_array[i],
                    label=f"{lengend_array[i]}", marker="o")

    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title(title)
    ax.set_xticks(x_axis_newt)
    ax.set_xticklabels(x_ticks_newt)
    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    plt.errorbar(x_axis_newt, average, yerr=exp_unc, color='0')
    plt.savefig(name_of_file)
    plt.show()


def plått_multi_short_unc(n_m_array, title, name_of_file, lengend_array, exp_unc, average):
    assert len(n_m_array.shape) == 2  # hvis denne failer har du en funky array
    fig, ax = plt.subplots()

    for i in range(n_m_array.shape[0]):
        ax.semilogx(x_axis_pta, n_m_array[i],
                    label=f"{lengend_array[i]}", marker="o")

    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title(title)
    ax.set_xticks(x_axis_pta)
    ax.set_xticklabels(x_ticks_pta)
    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    plt.errorbar(x_axis_pta, average, yerr=exp_unc)
    plt.savefig(name_of_file)
    plt.show()


def attenuation(data_with, data_without):
    return data_with - data_without


def db_to_pressure(measurements):
    return 10 ** (measurements / 10)


def pressure_to_db(measurements):
    return 10 * np.log10(measurements)


def average_spl(stacked_vec):
    pressure_vec = db_to_pressure(stacked_vec)
    avg_vec = np.average(pressure_vec, axis=0)
    db_vec = pressure_to_db(avg_vec)
    return db_vec


###########################
# Reading the files
###########################

data_newt_raw = read_csv(filename_newt)
data_newt = data_newt_raw[:, 2:]

data_pta_raw = read_csv(filename_pta)
data_pta = data_pta_raw[:, 1:]

#######################
# Group results
#######################

############################################
# Comparison of NEWT without HPD and PTA
############################################
NEWT_without_person7 = data_newt[11, :]
NEWT_without_person8 = data_newt[14, :]

NEWT_without_person7_short = data_newt[11, 2:]
NEWT_without_person8_short = data_newt[14, 2:]

PTA_person7_right = data_pta[4, :5]
PTA_person7_left = data_pta[4, 5:]

PTA_person8_right = data_pta[5, :5]
PTA_person8_left = data_pta[5, 5:]

group_results_stack = np.vstack((NEWT_without_person7_short, NEWT_without_person8_short, PTA_person7_right,
                                 PTA_person7_left, PTA_person8_right, PTA_person8_left))

# plått_multi_short(group_results_stack,title="Results for test subject 7 and 8", name_of_file="group_results.pdf", lengend_array=["NEWT test subject 8", "NEWT test subject 7", "PTA test subject 7 right","PTA test subject 7 left","PTA test subject 8 right","PTA test subject 8 left"])


person7 = np.vstack((NEWT_without_person7_short, PTA_person7_right, PTA_person7_left))
person8 = np.vstack((NEWT_without_person8_short, PTA_person8_right, PTA_person8_left))

plått_multi_short(person8, title= "NEWT and PTA for test subject 8", name_of_file="results_person8.pdf", lengend_array=["NEWT test subject 8", "PTA test subject 8 right", "PTA test subject 8 left"])
plått_multi_short(person7, title= "NEWT and PTA for test subject 7", name_of_file="results_person7.pdf", lengend_array=["NEWT test subject 7", "PTA test subject 7 right", "PTA test subject 7 left"])


########################################
# Attenuation of HPD
########################################

# Making av vector with all the NEWT results without HPD
NEWT_without_HPD = data_newt[[0, 2, 4, 6, 8, 11, 14, 17, 19], :]
NEWT_without_earplugs = data_newt[[0, 4, 6, 8, 11, 14, 17], :]
NEWT_without_earmuffs = data_newt[[2, 8, 11, 14, 19], :]

# Making a vector with all the NEWT results with earmuffs
NEWT_with_earmuffs = data_newt[[3, 9, 12, 15, 20], :]

# Making av vector with all the NEWT results with earplugs
NEWT_with_earplugs = data_newt[[1, 5, 7, 10, 13, 16, 18], :]

# Person 7
NEWT_with_earmuffs_person7 = data_newt[12, :]
NEWT_with_earplugs_person7 = data_newt[13, :]

attenuation_earmuffs_person7 = attenuation(NEWT_with_earmuffs_person7, NEWT_without_person7)
attenuation_earplugs_person7 = attenuation(NEWT_with_earplugs_person7, NEWT_without_person7)

# Person 8
NEWT_with_earmuffs_person8 = data_newt[15, :]
NEWT_with_earplugs_person8 = data_newt[16, :]

attenuation_earmuffs_person8 = attenuation(NEWT_with_earmuffs_person8, NEWT_without_person8)
attenuation_earplugs_person8 = attenuation(NEWT_with_earplugs_person8, NEWT_without_person8)

att_earmuffs_7_8 = np.vstack((attenuation_earmuffs_person7, attenuation_earmuffs_person8))
att_earplugs_7_8 = np.vstack((attenuation_earplugs_person7, attenuation_earplugs_person8))

# Molded earplugs
attenuation_molded_earplugs_person8 = attenuation(molded_earplugs_person8, NEWT_without_person8)
plått_simple_long(attenuation_molded_earplugs_person8,title="Attenuation for molded earplugs person 8", name_of_file="att_molded_person8.pdf")
# plått_simple_long(PTA_person8_right, title="Hearing treshold person 8", name_of_file="hearing_tresh_pers8.pdf")

# plått_multi_long(att_earmuffs_7_8, title="Attenuation for earmuffs for person 7 and 8",name_of_file="att_earmuffs_group.pdf", lengend_array=["Person 7", "Person 8"])
# plått_multi_long(att_earplugs_7_8, title="Attenuation for earplugs for person 7 and 8",name_of_file="att_earplugs_group.pdf", lengend_array=["Person 7", "Person 8"])

##########################
# All results
##########################


attenuation_earplugs = attenuation(NEWT_with_earplugs, NEWT_without_earplugs)
attenuation_earmuffs = attenuation(NEWT_with_earmuffs, NEWT_without_earmuffs)
print("NEWT with earmuffs", NEWT_with_earmuffs)
print("Newt without hpd", NEWT_without_HPD)

# plått_multi_long(attenuation_earmuffs, title="Attenuation for earmuffs", name_of_file="att_earmuffs_all.pdf", lengend_array=["1", "2", "3", "4", "5", "6"])

# plått_multi_long(attenuation_earplugs, title="Attenuation for earplugs", name_of_file="att_earmuffs_all.pdf", lengend_array=["1", "2", "3", "4", "5", "6"])

avg_att_earmuffs = average_spl(attenuation_earmuffs)
avg_att_earplugs = average_spl(attenuation_earplugs)
att_plugs_muffs = np.vstack((avg_att_earplugs, avg_att_earmuffs))
# plått_simple_long(avg_att_earmuffs, title="Average attenuation for earmuffs", name_of_file="att_earmuffs_avgerage.pdf")
# plått_simple_long(avg_att_earplugs, title="Average attenuation for earplugs", name_of_file="att_earplugs_avgerage.pdf")
earplugs_earmuffs_molded = np.vstack((avg_att_earplugs, avg_att_earmuffs, attenuation_molded_earplugs_person8))
# plått_multi_long(earplugs_earmuffs_molded, title="Attenuation for earplugs, earmuffs and molded earplugs", name_of_file="att_earplug_earmuff_mlded.pdf", lengend_array=["1","2","3"])
plått_multi_long(att_plugs_muffs, title="Average attenuation for earplugs and earmuffs", name_of_file="avg_earplug_earmuffs.pdf",lengend_array=["Earplugs", "Earmuffs"])
#############################
# Expanded uncertainty
#############################

# Coverage factor k
#k = 2.447  # for a 95% confidence interval of the Students t-distribution for 6 samples
k_earplugs = 2.365 #for 7 samples
k_earmuffs = 2.571  # for 5 samples


# Standard deviation for a normal distribution
std_earplugs_pressure = np.std(attenuation_earplugs, axis=0)
std_earplugs_db = np.std(attenuation_earplugs, axis=0)
std_uncertainty_earplugs = std_earplugs_db / std_earplugs_pressure.shape[0]
std_earmuffs_pressure = np.std(db_to_pressure(attenuation_earmuffs), axis=0)
std_earmuffs_db = np.std(attenuation_earmuffs, axis=0)
std_uncertainty_earmuffs = std_earmuffs_db / std_earmuffs_pressure.shape[0]


uncertainty_earplugs = std_earplugs_db / np.sqrt(7)
expanded_uncertainty_earplugs_db = k_earplugs * uncertainty_earplugs


uncertainty_earmuffs = std_earmuffs_db / np.sqrt(5)
expanded_uncertainty_earmuffs_db = k_earmuffs * uncertainty_earmuffs

# plått_multi_long(np.vstack((avg_att_earmuffs, avg_att_earplugs)), title="Average attenuation for earplugs and earmuffs",
#                  name_of_file="avg_att_earplug_earmuff.pdf", lengend_array=["Earmuffs", "Earplugs"])


plått_multi_long_unc(attenuation_earmuffs, title="Attenuation for earmuffs and expanded uncertainty",
                     name_of_file="att_earmuffs_all_unc.pdf", lengend_array=["Person 2", "Person 6", "Person 7", "Person 8", "Person 11"],
                     exp_unc=expanded_uncertainty_earmuffs_db, average=avg_att_earmuffs)


plått_multi_long_unc(attenuation_earplugs, title="Attenuation for earplugs and expanded uncertainty",
                     name_of_file="att_earplugs_all_unc.pdf", lengend_array=["Person 1", "Person 3", "Person 5", "Person 6", "Person 7", "Person 8", "Person 10", "Average and expanded uncertainty"],
                     exp_unc=expanded_uncertainty_earplugs_db, average=avg_att_earplugs)


avg_expanded_earplugs = pressure_to_db(np.average(db_to_pressure(expanded_uncertainty_earplugs_db)))
print("avg exp earplugs", avg_expanded_earplugs)

avg_expanded_earmuffs = pressure_to_db(np.average(db_to_pressure(expanded_uncertainty_earmuffs_db)))
print("AVG exp earmuffs", avg_expanded_earmuffs)

avg_earplugs_sum = pressure_to_db(np.sum(db_to_pressure(avg_att_earplugs)))
print(avg_earplugs_sum)

avg_earmuffs_sum = pressure_to_db(np.sum(db_to_pressure(avg_att_earmuffs)))
print(avg_earmuffs_sum)