import pandas as pd
import os
import glob
import json

# Cartella point cloud
folder = "pointclouds"

# Cartella principale del progetto
project_folder = os.path.dirname(__file__)

# Caricamento JSON
with open(os.path.join(folder, "acquisition_times.json"), "r") as f:
    acquisition_times = json.load(f)

# Tutti i file ply
all_ply = glob.glob(os.path.join(folder, "*.ply"))

# --- CREAZIONE DATAFRAME ---
df = pd.DataFrame({
    "file": all_ply,
    "base_name": [os.path.splitext(os.path.basename(f))[0] for f in all_ply],
})

# Aggiunge colonna 'time' leggendo dal JSON (rimuove 'waypoint_' per corrispondenza)
df["time"] = df["base_name"].apply(lambda x: acquisition_times.get(x.replace("waypoint_", ""), float("inf")))

# Ordina per tempo crescente
df = df.sort_values("time").reset_index(drop=True)

# --- SALVATAGGIO FILE TXT CON NOMI E TEMPI ---
txt_file = os.path.join(project_folder, "pointclouds_ordered.txt")
df_to_save = df[["base_name", "time"]]
df_to_save.to_csv(txt_file, sep="\t", index=False)
print(f"\nFile con point cloud ordinate e tempi salvato in: {txt_file}")

# --- TARGET E SOURCE PER ICP ---
target_file = df.loc[0, "file"]  # frame con tempo minimo
source_files = df.loc[1:, "file"].tolist()  # resto dei frame

print("Target:", target_file)
print("Source files in ordine crescente:", source_files)

