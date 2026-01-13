import open3d as o3d
import numpy as np
import os
import time
import pandas as pd
start_time = time.time()
project_folder = os.path.dirname(__file__)
folder = os.path.join(project_folder, "pointclouds")
df_file = os.path.join(project_folder, "pointclouds_ordered.txt")
save_folder = os.path.join(project_folder, "icpPointToPoint")
os.makedirs(save_folder, exist_ok=True)
df = pd.read_csv(df_file, sep="\t")
df["file"] = df["base_name"].apply(lambda x: os.path.join(folder, x + ".ply"))
target_file = df.loc[0, "file"]
source_files = df.loc[1:, "file"].tolist()
print("Target:", target_file)
print("Source files in ordine temporale:")
for f in source_files:
    print(" -", f)
voxel_size = 0.02
threshold = 0.1
nb_neighbors = 25
std_ratio = 2.5
def preprocess_frame(pcd, voxel, nb_neighbors, std_ratio):
    pcd_down = pcd.voxel_down_sample(voxel)
    pcd_clean, _ = pcd_down.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return pcd_clean
map_pcd = o3d.io.read_point_cloud(target_file)
map_pcd = preprocess_frame(map_pcd, voxel_size, nb_neighbors, std_ratio)
point_clouds = [
    preprocess_frame(
        o3d.io.read_point_cloud(f),
        voxel_size,
        nb_neighbors,
        std_ratio
    )
    for f in source_files
]
transformations = []
summary_lines = []

print("\nAvvio ICP Point-to-Point con pulizia outlier...\n")

for i, source in enumerate(point_clouds):
    print(f"\nAllineamento nuvola {i+1}/{len(point_clouds)}...")

    reg_icp = o3d.pipelines.registration.registration_icp(
        source,
        map_pcd,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    print(f"  Fitness: {reg_icp.fitness:.4f}, RMSE: {reg_icp.inlier_rmse:.4f}")
    source.transform(reg_icp.transformation)
    transformations.append(reg_icp.transformation)
    np.savetxt(
        os.path.join(save_folder, f"transformation_{i+1}.txt"),
        reg_icp.transformation
    )
    o3d.io.write_point_cloud(
        os.path.join(save_folder, f"aligned_{i+1}.ply"),
        source
    )
    summary_lines.append(
        f"Frame {i+1}: Fitness={reg_icp.fitness:.4f}, RMSE={reg_icp.inlier_rmse:.4f}\n"
    )
    map_pcd += source
    map_pcd = map_pcd.voxel_down_sample(voxel_size)

    map_pcd, _ = map_pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
map_pcd_cleaned, _ = map_pcd.remove_statistical_outlier(
    nb_neighbors=nb_neighbors,
    std_ratio=std_ratio
)
output_file = os.path.join(save_folder, "merged_icp_point_map.ply")
o3d.io.write_point_cloud(output_file, map_pcd_cleaned)

end_time = time.time()
total_time = end_time - start_time

summary_lines.append(
    f"\nTempo totale esecuzione: {total_time:.2f} secondi\n"
)
summary_file = os.path.join(save_folder, "icp_summary.txt")
with open(summary_file, "w") as f:
    f.writelines(summary_lines)
print(f"\nICP completato in {total_time:.2f} secondi")
print(f"Risultati salvati in: {save_folder}")
o3d.visualization.draw_geometries([map_pcd_cleaned])
screenshot_path = os.path.join(save_folder, "merged_icp_map_point.png")
vis = o3d.visualization.Visualizer()
vis.create_window(
    window_name="ICP-Point-To-Point Result",
    width=1920,
    height=1080,
    visible=True
)
vis.add_geometry(map_pcd_cleaned)
vis.poll_events()
vis.update_renderer()
time.sleep(0.5)
vis.capture_screen_image(screenshot_path)
vis.destroy_window()
print(f"Screenshot salvato in: {screenshot_path}")

