import open3d as o3d

# Percorso al file .ply
file_path = "ransac_results/merged_ransac_map.ply"

# Carica la point cloud
pcd = o3d.io.read_point_cloud(file_path)

# Visualizza la point cloud
o3d.visualization.draw_geometries([pcd])
