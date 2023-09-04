import matplotlib.pyplot as plt
from eval_helper import *

# @todo draw all plots together

plots = {
    "flowrate_300_37" : {"settings" : ["300_37_366" , "300_37_683", "300_37_1367"],
                         "legend" : []
                         }
}

settings = ["300_37_366" , "300_37_683", "300_37_1367"]

plot = ReductionPlot("300 mg, 37% H2, 1000 °C", ProgressOverTime, legend_labels=["366 ml/min g", "683 ml/min g", "1367 ml/min g"])
compare_progress(plot, settings, limit=[0, 15])

plot = ReductionPlot("300 mg, 37% H2, 1000 °C", YMasterPlotData, legend_labels=["366 ml/min g", "683 ml/min g", "1367 ml/min g"])

compare_reaction_rate(plot, settings)


plt.show()