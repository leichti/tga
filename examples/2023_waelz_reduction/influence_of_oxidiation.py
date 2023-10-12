from tga.data_loader import *
import matplotlib.pyplot as plt

test = TgaFile("../data/1072_V17.txt")
trial = Trial(test)

trial.apply("h2o", SavgolSmoother(101, 1), new_name="h2o_s")

oxidation = trial.search_segments_by_gasflow("h2o_s", segment_idx=0)

oxidation.apply("m", SavgolSmoother(3001, 1), new_name="m_s")
oxidation.apply("m_s", SavgolSmoother(3001, 3), new_name="m_s2")

plt.plot(oxidation["t"], oxidation["m_s"], alpha=0.8)
plt.plot(oxidation["t"], oxidation["m_s2"] , alpha=0.8)

oxidation.apply("m_s2", Deviate(1), "dmdt")
plt.gca().twinx()
plt.plot(oxidation["t"], oxidation["dmdt"])

plt.show()