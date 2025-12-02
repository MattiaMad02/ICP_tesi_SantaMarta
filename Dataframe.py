import pandas as pd
import os
import glob
import json
folder = "pointclouds"
project_folder = os.path.dirname(__file__)
with open(os.path.join(folder, "acquisition_times.json"), "r") as f:
    acquisition_times = json.load(f)
all_ply = glob.glob(os.path.join(folder, "*.ply"))
df = pd.DataFrame({
    "file": all_ply,
    "base_name": [os.path.splitext(os.path.basename(f))[0] for f in all_ply],
})
df["time"] = df["base_name"].apply(lambda x: acquisition_times.get(x.replace("waypoint_", ""), float("inf")))
df = df.sort_values("time").reset_index(drop=True)
txt_file = os.path.join(project_folder, "pointclouds_ordered.txt")
df_to_save = df[["base_name", "time"]]
df_to_save.to_csv(txt_file, sep="\t", index=False)
print(f"\nFile con point cloud ordinate e tempi salvato in: {txt_file}")
target_file = df.loc[0, "file"]  # frame con tempo minimo
source_files = df.loc[1:, "file"].tolist()  # resto dei frame
print("Target:", target_file)
print("Source files in ordine crescente:", source_files)

