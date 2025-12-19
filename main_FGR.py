import open3d as o3d
import numpy as np
import pandas as pd
import os
import time
def universal_fgr(source_down, target_down, source_fpfh, target_fpfh, dist_threshold):
    reg_module = o3d.pipelines.registration
    possible_functions = [
        "registration_fgr_based_on_feature_matching",
        "registration_fast_based_on_feature_matching",
    ]
    for fn_name in possible_functions:
        if hasattr(reg_module, fn_name):
            print(f"\nFound FGR function: {fn_name}")
            fgr_fn = getattr(reg_module, fn_name)
            break
    else:
        raise RuntimeError(
            "\nNessuna funzione FGR trovata nella tua versione di Open3D!\n"
            "Versione: " + str(o3d.__version__)
        )
    option = reg_module.FastGlobalRegistrationOption(
        maximum_correspondence_distance=dist_threshold
    )
    result = fgr_fn(
        source_down, target_down,
        source_fpfh, target_fpfh,
        option
    )

    return result
def preprocess(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=120)
    )
    return pcd_down, fpfh
start_time = time.time()
project_folder = os.path.dirname(__file__)
folder = os.path.join(project_folder, "pointclouds")
df_file = os.path.join(project_folder, "pointclouds_ordered.txt")
df = pd.read_csv(df_file, sep="\t")
df["file"] = df["base_name"].apply(lambda x: os.path.join(folder, x + ".ply"))
target_file = df.loc[0, "file"]
source_files = df.loc[1:, "file"].tolist()
print(f"Open3D version detected: {o3d.__version__}")
print("Target file:", target_file)
voxel_size = 0.05
dist_threshold = voxel_size * 1.5
target_raw = o3d.io.read_point_cloud(target_file)
target_down, target_fpfh = preprocess(target_raw, voxel_size)
map_pcd = target_raw.voxel_down_sample(voxel_size)
save_folder = os.path.join(project_folder, "fgr_universal_results")
os.makedirs(save_folder, exist_ok=True)
summary = []
print("\n=== FGR UNIVERSALE START ===\n")
for i, file in enumerate(source_files):
    print(f"\nFrame {i+1}/{len(source_files)} --> {file}")
    source_raw = o3d.io.read_point_cloud(file)
    source_down, source_fpfh = preprocess(source_raw, voxel_size)
    reg = universal_fgr(
        source_down, target_down, source_fpfh, target_fpfh, dist_threshold
    )
    print(f"  Fitness FGR: {reg.fitness:.4f}")
    print(f"  RMSE FGR:    {reg.inlier_rmse:.4f}")
    T = reg.transformation
    source_raw.transform(T)
    map_pcd += source_raw
    map_pcd = map_pcd.voxel_down_sample(voxel_size)
    o3d.io.write_point_cloud(os.path.join(save_folder, f"aligned_{i+1}.ply"), source_raw)
    np.savetxt(os.path.join(save_folder, f"T_fgr_{i+1}.txt"), T)
    summary.append(f"Frame {i+1}: Fitness={reg.fitness:.4f}, RMSE={reg.inlier_rmse:.4f}\n")
    target_down, target_fpfh = preprocess(map_pcd, voxel_size)
o3d.io.write_point_cloud(os.path.join(save_folder, "merged_fgr_map.ply"), map_pcd)
with open(os.path.join(save_folder, "fgr_summary.txt"), "w") as f:
    f.writelines(summary)
total = time.time() - start_time
print(f"\nFGR COMPLETATO in {total:.2f} secondi")
print(f"Risultati salvati in: {save_folder}")
o3d.visualization.draw_geometries([map_pcd])
vis = o3d.visualization.Visualizer()
vis.create_window(visible=True)
vis.add_geometry(map_pcd)
opt = vis.get_render_option()
opt.background_color = np.asarray([1, 1, 1])  # sfondo bianco
opt.point_size = 2.0
vis.poll_events()
vis.update_renderer()
screenshot_path = os.path.join(save_folder, "merged_fgr_map.png")
vis.capture_screen_image(screenshot_path)
print(f"Screenshot FGR salvato in: {screenshot_path}")
vis.run()
vis.destroy_window()

