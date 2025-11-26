import os
import time
import numpy as np
import open3d as o3d

# -------------------------
# Parametri e cartelle
# -------------------------
start_time = time.time()
project_folder = os.path.dirname(__file__)
pc_folder = os.path.join(project_folder, "pointclouds")
out_folder = os.path.join(project_folder, "ransac_icp_results")
os.makedirs(out_folder, exist_ok=True)

# Parametri
voxel_size = 0.012
distance_threshold_ransac = voxel_size * 2.0
distance_threshold_icp = 0.05  # per ICP refine su frame vicini
ransac_n = 4
convergence_criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
    max_iteration=1000000,
    confidence=0.999
)

# -------------------------
# Lista file ply nella cartella
# -------------------------
all_files = sorted([f for f in os.listdir(pc_folder) if f.endswith(".ply")])
if len(all_files) < 2:
    raise RuntimeError("La cartella deve contenere almeno due file .ply")

# Primo file = target
target_file = os.path.join(pc_folder, all_files[0])
target_pcd = o3d.io.read_point_cloud(target_file)
# Calcolo delle normali sul target
target_pcd.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30)
)
print("Target:", target_file)

# Tutti gli altri file = source
source_files = [os.path.join(pc_folder, f) for f in all_files[1:]]
print("Sources:")
for f in source_files:
    print(" -", f)

# -------------------------
# Funzioni helper
# -------------------------
def preprocess(pcd, voxel):
    pcd_down = pcd.voxel_down_sample(voxel)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*4, max_nn=100)
    )
    return pcd_down, fpfh

def run_ransac(src_down, tgt_down, src_fpfh, tgt_fpfh):
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold_ransac,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold_ransac),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9)
        ],
        criteria=convergence_criteria
    )
    return result

def run_icp(src, tgt, init_transformation):
    # Calcolo delle normali se mancanti
    if not src.has_normals():
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    if not tgt.has_normals():
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    return o3d.pipelines.registration.registration_icp(
        src, tgt, distance_threshold_icp, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

# -------------------------
# Preprocess target
# -------------------------
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

    # --- RANSAC ---
    result_ransac = run_ransac(src_down, target_down, src_fpfh, target_fpfh)
    if result_ransac.fitness == 0.0:
        print("  RANSAC failed, using identity transformation as initial guess")
        init_T = np.eye(4)
        failed_frames.append(idx)
    else:
        init_T = result_ransac.transformation
        print(f"  RANSAC success: Fitness={result_ransac.fitness:.4f}, RMSE={result_ransac.inlier_rmse:.4f}")

    # --- ICP refine ---
    result_icp = run_icp(src_pcd, target_pcd, init_T)
    print(f"  ICP refine: Fitness={result_icp.fitness:.4f}, RMSE={result_icp.inlier_rmse:.4f}")

    # Applica trasformazione finale
    src_pcd.transform(result_icp.transformation)

    # Salvataggio trasformazione
    np.savetxt(os.path.join(out_folder, f"transformation_{idx}.txt"), result_icp.transformation)
    # Salvataggio point cloud allineata
    o3d.io.write_point_cloud(os.path.join(out_folder, f"aligned_{idx}.ply"), src_pcd)

    summary_lines.append(f"Frame {idx}: RANSAC fitness={result_ransac.fitness:.4f}, ICP fitness={result_icp.fitness:.4f}, ICP RMSE={result_icp.inlier_rmse:.4f}\n")

# -------------------------
# Salvataggi finali
# -------------------------
if failed_frames:
    summary_lines.append("Frames failed RANSAC: " + ", ".join(map(str, failed_frames)) + "\n")

total_time = time.time() - start_time
summary_lines.append(f"Total execution time: {total_time:.2f} s\n")

with open(os.path.join(out_folder, "ransac_icp_refine_summary.txt"), "w") as f:
    f.writelines(summary_lines)

print("\nRANSAC + ICP refine completed.")
if failed_frames:
    print("Frames failed RANSAC:", failed_frames)
print("Results saved in:", out_folder)

# Visualizzazione finale di tutte le nuvole allineate
aligned_pcds = [target_pcd] + [o3d.io.read_point_cloud(os.path.join(out_folder, f"aligned_{i+1}.ply")) for i in range(len(source_files))]
o3d.visualization.draw_geometries(aligned_pcds)
