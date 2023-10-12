from tga.data_loader import *
import matplotlib.pyplot as plt

trials = ["1073_V18", "1076_V21"]

def plot_trial(name):

    f = TgaFile(f"../data/{name}.txt")
    trial = Trial(f)

    plt.figure()
    plt.plot(trial["t"], trial["m"])

def plot_deviation(name, segment, primary_ax, secondary_ax, selector):

    f = TgaFile(f"../data/{name}.txt")
    trial = Trial(f)

    trial.apply(selector, SavgolSmoother(101, 1), new_name=f"{selector}_s")

    segment = trial.search_segments_by_gasflow(f"{selector}_s", segment_idx=segment)

    # smoothing
    segment.apply("m", SavgolSmoother(2001, 4), new_name="m_s")
    segment.apply("m_s", SavgolSmoother(1001, 4), new_name="m_s2")

    # deviate
    segment.apply("m_s2", Deviate(1), "dmdt")

    # plot
    #primary_ax.plot(segment["t"]-segment["t"].iloc[0], segment["m"])
    primary_ax.plot(segment["t"]-segment["t"].iloc[0], segment["m_s2"])

    secondary_ax.plot(segment["t"]-segment["t"].iloc[0], segment["dmdt"])



# do we find any influence from oxidation->reduction circling?
# for "short" circles?

for name in ["1073_V18"]:

    fig, axes = plt.subplots(2,1)
    fig.suptitle(f"Oxidation {name}")
    primary_ax = axes[0]
    secondary_ax = axes[1]

    for i in range(0,4):
        plot_deviation(name, i, primary_ax, secondary_ax, "h2o")

    primary_ax.set_xlim(0, 5)
    secondary_ax.set_xlim(0, 5)
    fig, axes = plt.subplots(2,1)
    fig.suptitle(f"Reduction {name}")
    primary_ax = axes[0]
    secondary_ax = axes[1]
    primary_ax.set_xlim(0, 5)
    secondary_ax.set_xlim(0, 5)

    for i in range(0,4):
        plot_deviation(name, i, primary_ax, secondary_ax, "gas1")

# for "long" circles?
for name in ["1076_V21"]:

    fig, axes = plt.subplots(2,1)
    fig.suptitle(f"Oxidation {name}")
    primary_ax = axes[0]
    secondary_ax = axes[1]


    for i in range(0, 8):
        plot_deviation(name, i, primary_ax, secondary_ax, "h2o")

    primary_ax.set_xlim(0, 7.5)
    secondary_ax.set_xlim(0, 7.5)

    fig, axes = plt.subplots(2,1)
    fig.suptitle(f"Reduction {name}")
    primary_ax = axes[0]
    secondary_ax = axes[1]

    for i in range(0,8):
        plot_deviation(name, i, primary_ax, secondary_ax, "gas1")

    primary_ax.set_xlim(0, 7.5)
    secondary_ax.set_xlim(0, 7.5)


plot_trial("1076_V21")
plot_trial("1073_V18")

plt.show()