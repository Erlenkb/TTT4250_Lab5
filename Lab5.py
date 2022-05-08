import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

without_HPD = [0,3,6,9,12,15,18,21,24,27,30]
earmuffs = [4, 10, 16, 19, 22, 28, 31]
earplugs = [2, 8, 14, 17, 20, 23, 26]

####### Global Parameters #############

x_ticks_pta = [500, 1000, 2000, 4000, 8000]
x_ticks_newt = [125, 250, 500, 1000, 2000, 4000, 8000]
x_ticks_pta_label = ["500", "1k", "2k", "4k", "8k"]
x_ticks_newt_label = ["125","250","500", "1k", "2k", "4k", "8k"]

pta_rows = [3,5,8]
newt_rows = [10,11,12,25,26,27]
k_earplugs = 2.365  # for 7 samples
k_earmuffs = 2.365 # for 5 samples

test_subject1 = 4  #### Define what test person to plot 1
test_subject2 = 9  #### Define what test person to plot 2

#### Font details #######################################
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
#########################################################

def _calculate_SPL(array):
    temp = 0
    for i in array:
        temp = temp + 10**(0.1*i)
    return round((10*np.log10(temp)),1)


def _read_csv(filename, type):
    if type == "pta":
        df = pd.read_csv(filename, sep=";") #, skiprows=pta_rows)
        return df.to_numpy().astype(float)

    elif type == "newt":
        df = pd.read_csv(filename, sep=";", usecols=[str(x) for x in x_ticks_newt]) #,skiprows=newt_rows)
        return df.to_numpy().astype(float)
    else:
        df = pd.read_csv(filename, sep=";")
        return df.to_numpy().astype(float)

def _attenuation(arr_with,arr_without):
    return arr_with - arr_without

def _calculate_log_mean(lst):
    avg = 0
    for i in lst:
        avg += 10**(i / 10)
    return round(10*np.log10(avg / len(lst)),1)

def _plot_several(arr,arr2, title,title2, savefile, legend_arr):
    assert arr.ndim == 2, f"Array size weird. Expected 2 dim , got : {arr.ndim}"

    fig = plt.figure(figsize=(17, 6))

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for i, val in enumerate(arr): ax1.semilogx(x_ticks_pta, val,
                                              label="{0}".format(legend_arr[i]),
                                          marker=".")
    for i, val in enumerate(arr2): ax2.semilogx(x_ticks_pta, val,
                                              label="{0}".format(legend_arr[i]),
                                          marker=".")

    ax1.grid(which="major")
    ax1.grid(which="minor", linestyle=":")
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Amplitude [dB]")
    ax1.set_title(title)
    ax1.set_xticks(x_ticks_pta)
    ax1.set_xticklabels(x_ticks_pta_label)

    ax2.grid(which="major")
    ax2.grid(which="minor", linestyle=":")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Amplitude [dB]")
    ax2.set_title(title2)
    ax2.set_xticks(x_ticks_pta)
    ax2.set_xticklabels(x_ticks_pta_label)

    plt.legend(bbox_to_anchor=(1,0.5), loc="center left")
    plt.savefig(savefile)
    plt.tight_layout()
    plt.show()




def _plot_all_uncertainty(arr, title, savefile):
    assert arr.ndim == 2, f"Array size weird. Expected 2 dim , got : {arr.ndim}"
    fig1 = plt.figure(figsize=(10,10))
    fig, ax = plt.subplots(figsize=(10,5))

    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212)


    for i, val in enumerate(earmuffs):
        ax1.semilogx(x_ticks_newt, arr[val] - arr[val-1],
                     label="Person {}".format(1 + (val // 3)), marker=".")
    for i, val in enumerate(earplugs):
        ax2.semilogx(x_ticks_newt, arr[val] - arr[val-2],
                     label="Person {}".format(1 + (val // 3)), marker=".")

    without_earmuffs = [int(x-1) for x in earmuffs]
    without_earplugs = [int(x-2) for x in earplugs]

    attenuation_earmuffs = arr[earmuffs,:] - arr[without_earmuffs,:]
    attenuation_earplugs = arr[earplugs,:] - arr[without_earplugs,:]

    avg_att_earmuffs = []
    avg_att_earplugs = []

    for i in np.transpose(attenuation_earplugs) : avg_att_earplugs.append(_calculate_log_mean(i))
    for i in np.transpose(attenuation_earmuffs) : avg_att_earmuffs.append(_calculate_log_mean(i))

    std_earmuffs = np.std(attenuation_earmuffs, axis=0)
    std_earplugs = np.std(attenuation_earplugs, axis=0)

    expanded_unc_earmuffs = k_earmuffs * (std_earmuffs / np.sqrt(5))
    expanded_unc_earplugs = k_earplugs * (std_earplugs / np.sqrt(7))

    print("Avg expanded unc. earplugs {}".format(round(np.average(expanded_unc_earplugs),1)))
    print("Avg expanded unc. earmuffs {}".format(round(np.average(expanded_unc_earmuffs),1)))

    ax1.errorbar(x_ticks_newt, avg_att_earmuffs, yerr=expanded_unc_earmuffs,
                 linestyle="None", marker="^", ecolor = "black", mec="black",
                 ms=5, mew=2, elinewidth=2, capsize=7)
    ax2.errorbar(x_ticks_newt, avg_att_earplugs, yerr=expanded_unc_earplugs,
                 linestyle="None", marker="^", ecolor = "black", mec="black",
                 ms=5, mew=2, elinewidth=2, capsize=7)

    SPL_earmuffs = _calculate_SPL(avg_att_earmuffs)
    SPL_earplugs = _calculate_SPL(avg_att_earplugs)

    ax.semilogx(x_ticks_newt, avg_att_earmuffs,
                label="Earmuffs", marker=".")

    ax.semilogx(x_ticks_newt, avg_att_earplugs,
                label="Earplugs", marker=".")

    ax1.grid(which="major")
    ax1.grid(which="minor", linestyle=":")
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Amplitude [dB]")
    ax1.set_title("a) Attenuation with earmuffs including expanded uncertainty.\n Average expanded uncertainty: {0}dB".format(round(np.average(expanded_unc_earmuffs),1)))
    ax1.set_xticks(x_ticks_newt)
    ax1.set_xticklabels(x_ticks_newt_label)
    ax1.legend(bbox_to_anchor=(1, 0.5), loc="center left")

    ax2.grid(which="major")
    ax2.grid(which="minor", linestyle=":")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Amplitude [dB]")
    ax2.set_title("b) Attenuation with earplugs including expanded uncertainty.\n Average expanded uncertainty: {0}dB".format(round(np.average(expanded_unc_earplugs),1)))
    ax2.set_xticks(x_ticks_newt)
    ax2.set_xticklabels(x_ticks_newt_label)
    ax2.legend(bbox_to_anchor=(1, 0.5), loc="center left")

    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title("Average attenuation for Earmuffs and Earplugs\n Average attenuation earmuffs: {0}dB\n Average attenuation earplugs: {1}dB".format(SPL_earmuffs,SPL_earplugs))
    ax.set_xticks(x_ticks_newt)
    ax.set_xticklabels(x_ticks_newt_label)
    ax.legend(bbox_to_anchor=(1, 0.5), loc="center left")

    fig.tight_layout()
    fig1.tight_layout()
    fig1.savefig(savefile)
    fig.savefig("Avg_attenuation.png")
    plt.show()




####################################  RUN CODE FROM HERE ##########################################

"""
Create the array for both NEWT and PTA. Slice NEWT into both full and short to work with PTA
"""
file_pta = _read_csv("pta.csv", "pta")
file_newt = _read_csv("newt.csv", "newt")

file_newt_short = file_newt[:,2:]
pta_right = file_pta[:,0:5]
pta_left = file_pta[:,5:]


"""
Create the plot for the two test subjects and the entirety of all test subjects
"""
person1 = test_subject1
person2 = test_subject2

test_person1 = np.vstack((file_newt_short[without_HPD[person1-1]], pta_right[person1-1], pta_left[person1-1]))
test_person2 = np.vstack((file_newt_short[without_HPD[person2-1]], pta_right[person2-1], pta_left[person2-1]))


_plot_several(test_person1, test_person2, title="NEWT and PTA for test subject {0}".format(person1),
              title2="NEWT and PTA for test subject {0}".format(person2),savefile="Results_Person{0}_and_{1}.png".format(person1,person2),
              legend_arr=["NEWT", "PTA right ear", "PTA left ear"])


_plot_all_uncertainty(file_newt,"sumting","expanded_unc.png")



