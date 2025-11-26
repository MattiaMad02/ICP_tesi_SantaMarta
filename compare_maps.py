import open3d as o3d
import numpy as np
import os
#Confronto tra la mappa grezza e quella finale dopo ICP
# Cartella contenente i risultati ICP
base_folder = os.path.dirname(__file__)
map_no_icp_path = os.path.join(base_folder,"complete_map.ply")  # mappa originale
map_icp_path = os.path.join(base_folder, "icp_results", "merged_icp_map.ply")  # mappa ICP corretta


if not os.path.exists(map_no_icp_path):
   raise FileNotFoundError(f"Mappa originale non trovata: {map_no_icp_path}")

if not os.path.exists(map_icp_path):
    raise FileNotFoundError(f"Mappa ICP non trovata: {map_icp_path}")

map_no_icp = o3d.io.read_point_cloud(map_no_icp_path)
map_icp = o3d.io.read_point_cloud(map_icp_path)

print("Mappe caricate correttamente.")
print(f" - Mappa originale: {len(map_no_icp.points)} punti")
print(f" - Mappa ICP: {len(map_icp.points)} punti")

# Usa nearest neighbor per confrontare i punti
distances = map_no_icp.compute_point_cloud_distance(map_icp)
distances = np.asarray(distances)
rmse = np.sqrt(np.mean(distances ** 2))
mean_dist = np.mean(distances)
max_dist = np.max(distances)

print(f"\n Confronto tra mappe:")
print(f" - RMSE medio: {rmse:.6f}")
print(f" - Distanza media: {mean_dist:.6f}")
print(f" - Distanza massima: {max_dist:.6f}")

# Colora la mappa originale in rosso, quella ICP in verde
map_no_icp.paint_uniform_color([1, 0, 0])  # rosso
map_icp.paint_uniform_color([0, 0, 1])

print("\n Visualizzazione: ROSSO = originale | BLU = ICP")
o3d.visualization.draw_geometries([map_no_icp, map_icp])
