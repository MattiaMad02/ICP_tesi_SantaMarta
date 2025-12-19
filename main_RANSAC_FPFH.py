import os
import time
import numpy as np
import open3d as o3d
import pandas as pd
start_time = time.time()
project_folder = os.path.dirname(__file__)
pc_folder = os.path.join(project_folder, "pointclouds")
df_file = os.path.join(project_folder, "pointclouds_ordered.txt")
out_folder = os.path.join(project_folder, "ransac_results")
os.makedirs(out_folder, exist_ok=True)
voxel_size = 0.012
distance_threshold = voxel_size * 5.0
ransac_n = 3
convergence_criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
    max_iteration=2000000, confidence=0.999
)
df = pd.read_csv(df_file, sep="\t")
df["file"] = df["base_name"].apply(lambda x: os.path.join(pc_folder, x + ".ply"))
target_file = df.loc[0, "file"]
source_files = df.loc[1:, "file"].tolist()
print("Target:", target_file)
print("Source files in ordine temporale:")
for f in source_files:
    print(" -", f)
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
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=False,
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
target_pcd = o3d.io.read_point_cloud(target_file)
target_down, target_fpfh = preprocess(target_pcd, voxel_size)
summary_lines = []
failed_frames = []
aligned_files = []
for idx, src_file in enumerate(source_files, start=1):
    print(f"\n[{idx}/{len(source_files)}] Processing {src_file}")
    src_pcd = o3d.io.read_point_cloud(src_file)
    src_down, src_fpfh = preprocess(src_pcd, voxel_size)
    result = run_ransac(src_down, target_down, src_fpfh, target_fpfh)
    print(f"  Fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.4f}")
    if result.fitness == 0.0:
        print("  --> RANSAC failed. Frame skipped.")
        failed_frames.append(idx)
        continue
    src_pcd.transform(result.transformation)
    np.savetxt(os.path.join(out_folder, f"transformation_{idx}.txt"), result.transformation)
    aligned_file = os.path.join(out_folder, f"aligned_{idx}.ply")
    o3d.io.write_point_cloud(aligned_file, src_pcd)
    aligned_files.append(aligned_file)
    summary_lines.append(f"Frame {idx}: Fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.4f}\n")
merged_pcd = target_pcd
for aligned_file in aligned_files:
    src_pcd = o3d.io.read_point_cloud(aligned_file)
    merged_pcd += src_pcd
merged_pcd = merged_pcd.voxel_down_sample(voxel_size)
merged_file = os.path.join(out_folder, "merged_ransac_map.ply")
o3d.io.write_point_cloud(merged_file, merged_pcd)
print(f"Mappa cumulativa salvata in: {merged_file}")
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
o3d.visualization.draw_geometries([merged_pcd])
vis = o3d.visualization.Visualizer()
vis.create_window(visible=True)
vis.add_geometry(merged_pcd)
opt = vis.get_render_option()
opt.background_color = np.asarray([1, 1, 1])
opt.point_size = 2.0
vis.poll_events()
vis.update_renderer()
screenshot_path = os.path.join(out_folder, "merged_ransac_map.png")
vis.capture_screen_image(screenshot_path)
print(f"Screenshot salvato in: {screenshot_path}")
vis.run()
vis.destroy_window()

