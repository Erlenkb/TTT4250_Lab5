import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


####### Global Parameters #############

x_ticks_pta = [500, 1000, 2000, 4000, 8000]
x_ticks_newt = [125, 250, 500, 1000, 2000, 4000, 8000]
pta_rows = [3,5,8]
newt_rows = [10,11,12,25,26,27]

#### Font details #######################################
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
#########################################################

def _read_csv(filename, type):
    if type == "pta":
        df = pd.read_csv(filename, sep=";", skiprows=pta_rows)
        return df.to_numpy().astype(float)

    elif type == "newt":
        df = pd.read_csv(filename, sep=";", skiprows=newt_rows, usecols=[str(x) for x in x_ticks_newt])
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

def _plot_PTA_vs_NEWT_single_pers(arr, title, savefile, legend_arr):
    assert arr.ndim == 1, f"Array size to big. Expected 1, got : {arr.ndim}"

    fig, ax = plt.subplots()

    for i, val in enumerate(arr): ax.semilogx(x_ticks_pta, val,
                                              label="{0}".format(legend_arr[i]),
                                              marker=".")

    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title(title)
    ax.set_xticks(x_ticks_pta)
    ax.set_xtickslabels([str(x) for x in x_ticks_pta])
    plt.legend(bbox_to_anchor=(1,0.5), loc="center left")
    plt.savefig(savefile)
    plt.show()




file_pta = _read_csv("pta.csv", "pta")
file_newt = _read_csv("newt.csv", "newt")

newt_without_person7 = file_newt[11, :]

print(newt_without_person7)


