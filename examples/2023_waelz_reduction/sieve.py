import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("data/Sieve_Analysis_Waelz_slag_samples.xlsx", sheet_name="bzf3.3")

fig, axes = plt.subplots(2,1, sharex=True)
fig.suptitle("Sieve Analysis Waelz Slag BZF3.3")
axes[0].plot(df["mMaterial [g]"]/df["mMaterial [g]"].sum()*100, color="k")
axes[1].plot(df["Akumuliert"]/df["Akumuliert"].iloc[-1]*100, color="k")
axes[1].set_xticks(df.index)
axes[1].set_xticklabels(df["Siebgröße [mm]:"])
axes[0].set_ylabel("Fraction [%]")
axes[1].set_ylabel("Sum [%]")
axes[1].set_xlabel("Mesh Size in μm")

fig.savefig("sieve_analysis.svg")
plt.show()