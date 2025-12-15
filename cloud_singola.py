import open3d as o3d
file_path = "multiscale_icp_robust/merged_multiscale_icp_map_robust.ply"
pcd = o3d.io.read_point_cloud(file_path)
o3d.visualization.draw_geometries([pcd])
