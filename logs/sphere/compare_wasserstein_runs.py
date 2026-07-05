"""Wasserstein vs L_max (2–19): new method vs standard FM."""
import numpy as np
import matplotlib.pyplot as plt

L_max = np.arange(2, 20)

new_method = [
    9.73006993368114, 7.610179122328078, 8.654909315846238, 7.06516516875101,
    7.340493969744561, 9.037157644353586, 8.440344934855613, 9.027870459977443,
    7.741281712394866, 9.260131104567996, 8.023023329076711, 8.164318208477225,
    8.11960528716276, 8.835061438832065, 8.482222640587306, 8.163547313442097,
    9.68428197756341, 8.346483554323196,
]

standard_fm = [
    25.960222333711293, 25.83828349113599, 25.823245899050285, 26.146963205025756,
    25.96264771376764, 25.783768150577306, 25.476996349394497, 26.0799878770999,
    26.02226466774386, 25.815620843314168, 26.14078639790656, 25.73980373986508,
    26.55635312034102, 25.997635550458767, 25.87168745131189, 25.628318869428796,
    25.69446396927192, 25.656674752105488,
]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    L_max, new_method,
    color="#2e86ab", marker="o", markersize=6, linewidth=1.8,
    markeredgecolor="white", markeredgewidth=0.8, label="New method",
)
ax.plot(
    L_max, standard_fm,
    color="#e94f37", marker="s", markersize=5, linewidth=1.8,
    markeredgecolor="white", markeredgewidth=0.8, label="Standard FM",
)

ax.set_xlabel(r"$L_{\max}$")
ax.set_ylabel("Wasserstein distance")
ax.set_title("Wasserstein distance vs spherical harmonic cutoff")
ax.set_xticks(L_max)
ax.set_xticklabels(L_max, rotation=45, ha="right")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.35)
ax.set_xlim(1.5, 19.5)
ax.set_ylim(0, max(standard_fm) * 1.05)
plt.tight_layout()

out_path = __file__.replace(".py", ".png")
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
if __import__("matplotlib").get_backend().lower() != "agg":
    plt.show()
