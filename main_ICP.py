import open3d as o3d
import numpy as np
import os
import time
import pandas as pd

# TIMER INIZIALE
start_time = time.time()

# Cartella progetto e cartella point cloud
project_folder = os.path.dirname(__file__)
folder = os.path.join(project_folder, "pointclouds")

# --- LETTURA DATAFRAME ORDINATO ---
df_file = os.path.join(project_folder, "pointclouds_ordered.txt")
df = pd.read_csv(df_file, sep="\t")

# Ricostruzione percorsi completi dei file
df["file"] = df["base_name"].apply(lambda x: os.path.join(folder, x + ".ply"))

# --- DEFINIZIONE TARGET E SOURCE ---
target_file = df.loc[0, "file"]       # primo file = tempo minimo
source_files = df.loc[1:, "file"].tolist()  # resto dei file in ordine crescente

print("Target:", target_file)
print("Source files in ordine crescente di tempo:")
for f in source_files:
    print(" -", f)

# --- PARAMETRI ICP ---
voxel_size = 0.02
threshold = 0.1  # aumento threshold per evitare 0.0

# --- CARICAMENTO MAPPA INIZIALE (TARGET) ---
print("\nCaricamento mappa iniziale (target)...")
map_pcd = o3d.io.read_point_cloud(target_file)
map_pcd = map_pcd.voxel_down_sample(voxel_size)
map_pcd.estimate_normals()

# --- CARICAMENTO DELLE ALTRE POINT CLOUD ---
point_clouds = [o3d.io.read_point_cloud(f) for f in source_files]

# --- PREPROCESSING ---
print("\nPreprocessing delle nuvole...")
for i in range(len(point_clouds)):
    pcd = point_clouds[i].voxel_down_sample(voxel_size)
    pcd.estimate_normals()
    point_clouds[i] = pcd

# --- CREAZIONE CARTELLA RISULTATI ---
save_folder = os.path.join(project_folder, "icp_results")
os.makedirs(save_folder, exist_ok=True)

# --- ICP ---
transformations = []
summary_lines = []

print("\nAvvio ICP...\n")

for i, source in enumerate(point_clouds):
    print(f"\nAllineamento nuvola {i+1}/{len(point_clouds)}...")

    reg_icp = o3d.pipelines.registration.registration_icp(
        source, map_pcd, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    print(f"  Fitness: {reg_icp.fitness:.4f}, RMSE: {reg_icp.inlier_rmse:.4f}")

    # Applica trasformazione
    source.transform(reg_icp.transformation)
    transformations.append(reg_icp.transformation)

    # Salva trasformazione
    np.savetxt(os.path.join(save_folder, f"transformation_{i+1}.txt"), reg_icp.transformation)

    # Salva nuvola trasformata
    o3d.io.write_point_cloud(os.path.join(save_folder, f"aligned_{i+1}.ply"), source)

    summary_lines.append(
        f"Frame {i+1}: Fitness={reg_icp.fitness:.4f}, RMSE={reg_icp.inlier_rmse:.4f}\n"
    )

    # --- AGGIORNAMENTO TARGET CUMULATIVO ---
    map_pcd += source
    map_pcd = map_pcd.voxel_down_sample(voxel_size)
    map_pcd.estimate_normals()

# --- CREAZIONE MAPPA COMPLETA ---
print("\nFusione finale delle mappe...")

merged = map_pcd  # la mappa già cumulativa

# SALVATAGGIO MAPPA
output_file = os.path.join(save_folder, "merged_icp_map.ply")
o3d.io.write_point_cloud(output_file, merged)

# TEMPO TOTALE
end_time = time.time()
total_time = end_time - start_time
summary_lines.append(f"\nTempo totale esecuzione: {total_time:.2f} secondi\n")

# SUMMARY FILE
summary_file = os.path.join(save_folder, "icp_summary.txt")
with open(summary_file, "w") as f:
    f.writelines(summary_lines)

print(f"\nICP completato in {total_time:.2f} secondi")
print(f"Risultati salvati in: {save_folder}")

# VISUALIZZAZIONE
o3d.visualization.draw_geometries([map_pcd])
