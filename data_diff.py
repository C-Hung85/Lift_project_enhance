import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

for root, folder, files in os.walk("/media/belkanwar/SATA_CORE/lifts/result_default/"):
    for file in files:
        data_default = pd.read_csv(os.path.join(root, file))
        data_alter = pd.read_csv(os.path.join(root.replace("result_default", "result_only_vertical"), file))
        data_default = data_default.merge(data_alter, on="second", how="left", suffixes=("", " alter"))
        data_default.loc[:, "delta"] = data_default["vertical_travel_distance (mm)"]-data_default["vertical_travel_distance (mm) alter"]
        
        plt.plot()
        sns.lineplot(data=data_default, x="second", y="delta")
        plt.title(file.replace(".csv", ""))
        plt.savefig(os.path.join("images/", file.replace("csv", "png")))
        plt.close()

