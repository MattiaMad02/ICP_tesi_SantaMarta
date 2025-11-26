import os
import time
import numpy as np
import open3d as o3d
import pandas as pd

# -------------------------
# Parametri e cartelle
# -------------------------
start_time = time.time()
project_folder = os.path.dirname(__file__)
pc_folder = os.path.join(project_folder, "pointclouds")
df_file = os.path.join(project_folder, "pointclouds_ordered.txt")
out_folder = os.path.join(project_folder, "ransac_results")
os.makedirs(out_folder, exist_ok=True)

# Parametri RANSAC + FPFH più permissivi
voxel_size = 0.012
distance_threshold = voxel_size * 5.0  # aumentato per tollerare frame più distanti
ransac_n = 3                           # meno punti necessari
convergence_criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
    max_iteration=2000000, confidence=0.999
)

# -------------------------
# Lettura dataframe ordinato
# -------------------------
df = pd.read_csv(df_file, sep="\t")
df["file"] = df["base_name"].apply(lambda x: os.path.join(pc_folder, x + ".ply"))

target_file = df.loc[0, "file"]
source_files = df.loc[1:, "file"].tolist()

print("Target:", target_file)
print("Source files in ordine temporale:")
for f in source_files:
    print(" -", f)

# -------------------------
# Funzioni helper
# -------------------------
def preprocess(pcd, voxel):
    """Downsample, calcolo normali e FPFH features"""
    pcd_down = pcd.voxel_down_sample(voxel)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*4, max_nn=100)
    )
    return pcd_down, fpfh

def run_ransac(src_down, tgt_down, src_fpfh, tgt_fpfh):
    """Esegue RANSAC + FPFH tra due nuvole preprocessate"""
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=False,  # più permissivo
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9)
        ],
        criteria=convergence_criteria
    )
    return result

# -------------------------
# Preprocess target
# -------------------------
target_pcd = o3d.io.read_point_cloud(target_file)
target_down, target_fpfh = preprocess(target_pcd, voxel_size)

summary_lines = []
failed_frames = []

# -------------------------
# Loop su tutti i source
# -------------------------
for idx, src_file in enumerate(source_files, start=1):
    print(f"\n[{idx}/{len(source_files)}] Processing {src_file}")
    src_pcd = o3d.io.read_point_cloud(src_file)
    src_down, src_fpfh = preprocess(src_pcd, voxel_size)

    # RANSAC
    result = run_ransac(src_down, target_down, src_fpfh, target_fpfh)
    print(f"  Fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.4f}")

    if result.fitness == 0.0:
        print("  --> RANSAC failed. Frame skipped.")
        failed_frames.append(idx)
        continue

    # Applica trasformazione
    src_pcd.transform(result.transformation)

    # Salvataggio trasformazione e nuvola allineata
    np.savetxt(os.path.join(out_folder, f"transformation_{idx}.txt"), result.transformation)
    o3d.io.write_point_cloud(os.path.join(out_folder, f"aligned_{idx}.ply"), src_pcd)

    summary_lines.append(f"Frame {idx}: Fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.4f}\n")

# -------------------------
# Salvataggi finali
# -------------------------
if failed_frames:
    summary_lines.append("Frames failed: " + ", ".join(map(str, failed_frames)) + "\n")
total_time = time.time() - start_time
summary_lines.append(f"Total execution time: {total_time:.2f} s\n")

with open(os.path.join(out_folder, "ransac_optimized_summary.txt"), "w") as f:
    f.writelines(summary_lines)

print("\nRANSAC optimized test completed.")
if failed_frames:
    print("Frames failed:", failed_frames)
print("Results saved in:", out_folder)

# Visualizzazione finale (solo frame allineati)
aligned_pcds = [target_pcd] + [
    o3d.io.read_point_cloud(os.path.join(out_folder, f"aligned_{i+1}.ply"))
    for i in range(len(source_files)) if (i+1) not in failed_frames
]
o3d.visualization.draw_geometries(aligned_pcds)
