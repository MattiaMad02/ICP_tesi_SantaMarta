import re
import matplotlib.pyplot as plt
import os

def read_fitness_rmse(txt_file):
    fitness = []
    rmse = []

    with open(txt_file, "r") as f:
        for line in f:
            f_match = re.search(r"Fitness=([0-9.eE+-]+)", line)
            r_match = re.search(r"RMSE=([0-9.eE+-]+)", line)

            if f_match and r_match:
                fitness.append(float(f_match.group(1)))
                rmse.append(float(r_match.group(1)))

    return fitness, rmse


# === Percorsi ai file ===
files = {
    "ICP": "icp_results/icp_summary.txt",
    "GICP": "gicp_cleaned_results/gicp_summary.txt",
    "FGR": "fgr_universal_results/fgr_summary.txt",
    "RANSAC+FPFH": "ransac_results/ransac_optimized_summary.txt",
    "ICP PTPo":"icpPointToPoint/icp_summary.txt",
    "MultiScaleICP": "multiscale_icp_robust/multiscale_icp_robust_summary.txt",
}

fitness_all = []
rmse_all = []
labels = []

for name, path in files.items():
    if os.path.exists(path):
        f, r = read_fitness_rmse(path)
        fitness_all.append(f)
        rmse_all.append(r)
        labels.append(name)
    else:
        print(f"⚠️ File non trovato: {path}")

# === BOXPLOT FITNESS ===
plt.figure()
plt.boxplot(fitness_all, tick_labels=labels, showfliers=True)
plt.ylabel("Fitness")
plt.title("Confronto Fitness tra algoritmi")
plt.grid(True)
plt.tight_layout()
plt.savefig("boxplot_fitness.png", dpi=300)
plt.show()

# === BOXPLOT RMSE ===
plt.figure()
plt.boxplot(rmse_all, tick_labels=labels, showfliers=True)
plt.ylabel("RMSE")
plt.title("Confronto RMSE tra algoritmi")
plt.grid(True)
plt.tight_layout()
plt.savefig("boxplot_rmse.png", dpi=300)
plt.show()
