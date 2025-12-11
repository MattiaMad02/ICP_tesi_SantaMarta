import open3d as o3d
import numpy as np
import os
import time
import pandas as pd
start_time = time.time()
project_folder = os.path.dirname(__file__)
folder = os.path.join(project_folder, "pointclouds")
df_file = os.path.join(project_folder, "pointclouds_ordered.txt")
df = pd.read_csv(df_file, sep="\t")
df["file"] = df["base_name"].apply(lambda x: os.path.join(folder, x + ".ply"))
target_file = df.loc[0, "file"]
source_files = df.loc[1:, "file"].tolist()  # resto dei file
print("Target:", target_file)
voxel_size = 0.02
threshold = 0.15
print("\nCaricamento mappa iniziale (target)...")
map_pcd = o3d.io.read_point_cloud(target_file)
map_pcd = map_pcd.voxel_down_sample(voxel_size)
map_pcd.estimate_normals()
point_clouds = [o3d.io.read_point_cloud(f) for f in source_files]
print("\nPreprocessing delle nuvole...")
for i in range(len(point_clouds)):
    pcd = point_clouds[i].voxel_down_sample(voxel_size)
    pcd.estimate_normals()
    point_clouds[i] = pcd
save_folder = os.path.join(project_folder, "gicp_results")
os.makedirs(save_folder, exist_ok=True)
transformations = []
summary_lines = []
print("\nAvvio GICP...\n")
for i, source in enumerate(point_clouds):
    print(f"\nAllineamento nuvola {i+1}/{len(point_clouds)}...")
    reg_gicp = o3d.pipelines.registration.registration_icp(
        source,
        map_pcd,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=80
        )
    )
    print(f"  Fitness: {reg_gicp.fitness:.4f}, RMSE: {reg_gicp.inlier_rmse:.4f}")
    source.transform(reg_gicp.transformation)
    transformations.append(reg_gicp.transformation)
    np.savetxt(os.path.join(save_folder, f"transformation_{i+1}.txt"),
               reg_gicp.transformation)
    o3d.io.write_point_cloud(os.path.join(save_folder, f"aligned_{i+1}.ply"), source)
    summary_lines.append(
        f"Frame {i+1}: Fitness={reg_gicp.fitness:.4f}, RMSE={reg_gicp.inlier_rmse:.4f}\n"
    )
    map_pcd += source
    map_pcd = map_pcd.voxel_down_sample(voxel_size)
    map_pcd.estimate_normals()
print("\nFusione finale delle mappe...")
output_file = os.path.join(save_folder, "merged_gicp_map.ply")
o3d.io.write_point_cloud(output_file, map_pcd)
end_time = time.time()
total_time = end_time - start_time
summary_lines.append(f"\nTempo totale esecuzione: {total_time:.2f} secondi\n")
summary_file = os.path.join(save_folder, "gicp_summary.txt")
with open(summary_file, "w") as f:
    f.writelines(summary_lines)
print(f"\nGICP completato in {total_time:.2f} secondi")
print(f"Risultati salvati in: {save_folder}")
o3d.visualization.draw_geometries([map_pcd])
