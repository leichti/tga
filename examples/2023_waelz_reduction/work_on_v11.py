import pandas as pd

df = pd.read_csv("data/1053_V11.txt", delimiter=",", skiprows=4, encoding="ISO-8859-1")

import matplotlib.pyplot as plt


plt.plot(df["Time(s)"], df["Delta m(mg)"])


mask = df["Time(s)"]<550
last_val = df.loc[mask, "Delta m(mg)"].iloc[-1]
df.loc[mask, "Delta m(mg)"] -= df.loc[mask, "Delta m(mg)"]
reversed_mask = ~mask
df.loc[reversed_mask, "Delta m(mg)"] -= last_val


plt.plot(df["Time(s)"], df["Gas 1(sccm/min)"], label="H2")
plt.plot(df["Time(s)"], df["Wasser(ml/min)"]*1240, label="H2O")

df.to_csv("10530_V11.txt", index=False, encoding="ISO-8859-1")
plt.legend()

import sys
plt.show()
sys.exit()
# Define the filename
filename = "10530_V11.txt"

# Read the original contents of the file
with open(filename, 'r') as file:
    original_contents = file.read()

# The lines you want to prepend
prepend_lines = """# Export date and time: Mi Sep 6 09:39:16 2023
# Name: V11
# Measurement date and time: Sa Aug 26 21:35:53 2023
# Weight: 1034 mg
"""

# Write the new lines followed by the original contents back to the file
with open(filename, 'w') as file:
    file.write(prepend_lines + original_contents)

plt.show()