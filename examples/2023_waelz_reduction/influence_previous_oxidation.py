from tga.data_loader import *
import matplotlib.pyplot as plt

trials = ["1073_V18", "1076_V21"]
i=0
colors = plt.cm.viridis(np.linspace(0, 1, 8))
def plot_trial(name, ax):
    global i
    f = TgaFile(f"data/{name}.txt")
    trial = Trial(f)


    ax.plot(trial["t"]-trial["t"].iloc[0], trial["m"], label=" ")
    i+=1

def plot_deviation(name, segment, primary_ax, secondary_ax, selector):

    f = TgaFile(f"data/{name}.txt")
    trial = Trial(f)

    trial.apply(selector, SavgolSmoother(101, 1), new_name=f"{selector}_s")

    segment = trial.search_segments_by_gasflow(f"{selector}_s", segment_idx=segment)

    # smoothing
    segment.apply("m", SavgolSmoother(2001, 4), new_name="m_s")
    segment.apply("m_s", SavgolSmoother(1001, 4), new_name="m_s2")

    # deviate
    segment.apply("m_s2", Deviate(1), "dmdt")

    # plot
    primary_ax.plot(segment["t"]-segment["t"].iloc[0], segment["m_s2"], label=" ")

    secondary_ax.plot(segment["t"]-segment["t"].iloc[0], -segment["dmdt"], label=" ")
    secondary_ax.set_ylabel("Reaction Rate [mg/min]")
    secondary_ax.set_xlabel("Time [min]")
    primary_ax.set_ylabel("Mass [mg]")




# do we find any influence from oxidation->reduction circling?
# for "short" circles?

fig, axes = plt.subplots(3,1, figsize=(10,10))
axes[1].set_xlim(0,10)
axes[2].set_xlim(0,10)
for name in ["1049_V8", "1050_V9"]:
    plot_deviation(name, 0, axes[1], axes[2], "gas1")

h, l = axes[1].get_legend_handles_labels()
axes[1].legend(h, ["Direct H2 Reduction", "Prior H2O Oxidation"])
axes[2].legend(h, ["Direct H2 Reduction", "Prior H2O Oxidation"])

#fig, axes = plt.subplots(1,1)
for name in ["1049_V8", "1050_V9"]:
    plot_trial(name, axes[0])

h = axes[0].get_legend_handles_labels()
axes[0].legend(h, ["Direct H2 Reduction", "Prior H2O Oxidation"])
axes[0].set_ylabel("Mass [mg]")
axes[0].set_xlabel("Time [min]")
fig.savefig("influence_prior_oxidation.svg")
plt.show()