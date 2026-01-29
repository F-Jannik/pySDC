import matplotlib.pyplot as plt
from collections import defaultdict



def file_to_plot(file):
    plt.figure()
    x_axis = "dt"
    y_axis = "error"
    label = ""

#    plt.plot((0.1, 0.0125), (1E-2, 1E-2/(8**1)), color="black", linestyle=":")

    tuples = list()
    with open(file, "r") as f:
        for line in f:
            tuples.append(eval(line))

    if type(tuples[0][0])==str:
        label = tuples[0][0]
        x_axis = tuples[0][1]
        y_axis = tuples[0][2]
        tuples = tuples[1:]

    if len(tuples[0])==3:
        groups = defaultdict(list)
        for t in tuples:
            key = str(t[0])
            value = t[1:]
            groups[key].append(value)

        for key, value in groups.items():
            x, y = zip(*value)
            plt.plot(x, y, marker = "o", label=label+key)
    else:
        x, y = zip(*tuples)
        plt.plot(x, y, marker = "o")

    plt.title(file)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.xscale("log")
    plt.yscale("log")
    plt.gca().invert_xaxis()
    plt.legend()


#file_to_plot("Iter_Errors_noM_control.txt")
#file_to_plot("Iter_Errors_M_control.txt")
#file_to_plot("Iter_Errors_noM_project.txt")
#file_to_plot("Iter_Errors_M_project.txt")
#file_to_plot("Iter_Errors_noM_restrict.txt")
#file_to_plot("Iter_Errors_M_restrict.txt")
#file_to_plot("Iter_Errors_noM_both.txt")
#file_to_plot("Iter_Errors_M_both.txt")
file_to_plot("Iter_Errors_noM.txt")
file_to_plot("Iter_Errors_M.txt")
# file_to_plot("Iter_Errors_noM2.txt")
# file_to_plot("Iter_Errors_M2.txt")
plt.show()
