import open3d as o3d
import numpy as np
import os
import time
import pandas as pd

# ===============================
# TIMER
# ===============================
start_time = time.time()

# ===============================
# PATH
# ===============================
project_folder = os.path.dirname(__file__)
folder = os.path.join(project_folder, "pointclouds")
df_file = os.path.join(project_folder, "pointclouds_ordered.txt")

save_folder = os.path.join(project_folder, "multiscale_icp_robust")
os.makedirs(save_folder, exist_ok=True)

# ===============================
# LETTURA FILE
# ===============================
df = pd.read_csv(df_file, sep="\t")
df["file"] = df["base_name"].apply(lambda x: os.path.join(folder, x + ".ply"))

target_file = df.loc[0, "file"]
source_files = df.loc[1:, "file"].tolist()

print("Target:", target_file)
print("Source files:")
for f in source_files:
    print(" -", f)

# ===============================
# PARAMETRI
# ===============================
voxel_size = 0.02
threshold = 0.1
nb_neighbors = 20
std_ratio = 2.0

# Scala grossa -> fine
voxel_scales = [0.04, 0.03, 0.02]
max_iters = [25, 25, 20]  # iterazioni più alte sulle scale fini

# ===============================
# PREPROCESSING ROBUSTO
# ===============================
def preprocess_frame(pcd, voxel, estimate_normals=True):
    pcd_down = pcd.voxel_down_sample(voxel)
    if estimate_normals:
        pcd_down.estimate_normals()
    return pcd_down

# ===============================
# MULTISCALE ICP ROBUSTO
# ===============================
def multiscale_icp(source, target, threshold, voxel_scales, max_iters):
    T = np.eye(4)

    for i, voxel in enumerate(voxel_scales):
        print(f"    → Scala {i+1}: voxel={voxel}")

        # Preprocessing senza rimuovere outlier ad ogni scala
        src = preprocess_frame(source, voxel)
        tgt = preprocess_frame(target, voxel)

        reg = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            threshold,  # threshold fisso per maggiore stabilità
            T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iters[i])
        )

        T = reg.transformation
        print(f"       fitness={reg.fitness:.4f}, RMSE={reg.inlier_rmse:.4f}")

    return T

# ===============================
# CARICAMENTO MAPPA INIZIALE
# ===============================
print("\nCaricamento mappa iniziale...")
map_pcd = o3d.io.read_point_cloud(target_file)
map_pcd = preprocess_frame(map_pcd, voxel_size)
map_pcd, _ = map_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
map_pcd.estimate_normals()

# ===============================
# CARICAMENTO SOURCE
# ===============================
point_clouds = [
    preprocess_frame(o3d.io.read_point_cloud(f), voxel_size)
    for f in source_files
]

# ===============================
# LOOP MULTISCALE ICP
# ===============================
transformations = []
summary_lines = []

print("\nAvvio MULTISCALE ICP robusto...\n")

for i, source in enumerate(point_clouds):
    print(f"\nAllineamento nuvola {i+1}/{len(point_clouds)}")

    T = multiscale_icp(source, map_pcd, threshold, voxel_scales, max_iters)

    source.transform(T)
    transformations.append(T)

    np.savetxt(os.path.join(save_folder, f"transformation_{i+1}.txt"), T)
    o3d.io.write_point_cloud(os.path.join(save_folder, f"aligned_{i+1}.ply"), source)

    summary_lines.append(f"Frame {i+1}: OK, fitness={np.nan}, RMSE={np.nan}\n")

    # Aggiornamento mappa cumulativa
    map_pcd += source
    map_pcd = map_pcd.voxel_down_sample(voxel_size)
    map_pcd.estimate_normals()

# ===============================
# CLEAN FINALE MAPPA
# ===============================
map_pcd, _ = map_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

# ===============================
# SALVATAGGI FINALI
# ===============================
output_file = os.path.join(save_folder, "merged_multiscale_icp_map_robust.ply")
o3d.io.write_point_cloud(output_file, map_pcd)

end_time = time.time()
total_time = end_time - start_time
summary_lines.append(f"\nTempo totale esecuzione: {total_time:.2f} secondi\n")

summary_file = os.path.join(save_folder, "multiscale_icp_robust_summary.txt")
with open(summary_file, "w") as f:
    f.writelines(summary_lines)

print(f"\nMULTISCALE ICP robusto completato in {total_time:.2f} secondi")
print(f"Risultati salvati in: {save_folder}")

# ===============================
# VISUALIZZAZIONE
# ===============================
o3d.visualization.draw_geometries([map_pcd])
