import matplotlib.pyplot as plt

methods = ["NN", "GNN", "JPDA", "MHT"]
cost = [1.0, 2.2, 3.6, 4.8]
accuracy = [1.2, 2.8, 4.0, 4.8]

fig, ax = plt.subplots(figsize=(8, 6))

# Background regions
ax.axvspan(0.5, 1.7, alpha=0.08)
ax.axvspan(1.7, 3.0, alpha=0.05)
ax.axvspan(3.0, 5.3, alpha=0.03)

ax.scatter(cost, accuracy, s=240, zorder=3)

for x, y, label in zip(cost, accuracy, methods):
    ax.text(x + 0.08, y + 0.08, label, fontsize=12, weight="bold")

ax.plot(cost, accuracy, linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)

ax.set_xlim(0.5, 5.3)
ax.set_ylim(0.7, 5.3)
ax.set_xlabel("Computational cost (qualitative)", fontsize=12)
ax.set_ylabel("Association accuracy / ambiguity handling (qualitative)", fontsize=12)
ax.set_title("Qualitative trade-off of data association methods", fontsize=13)
ax.grid(True, alpha=0.25)

ax.annotate(
    "Higher cost",
    xy=(5.05, 0.85),
    xytext=(3.9, 0.85),
    arrowprops=dict(arrowstyle="->", lw=1.5),
    fontsize=11,
    va="center",
)

ax.annotate(
    "Better accuracy",
    xy=(0.75, 5.05),
    xytext=(0.75, 3.8),
    arrowprops=dict(arrowstyle="->", lw=1.5),
    fontsize=11,
    rotation=90,
    ha="center",
)

ax.text(0.55, 5.18, "Relative placement only", fontsize=10, alpha=0.8)

plt.tight_layout()

plt.savefig('poster/plots/association_methods_qual_tradeoff.svg', dpi=300, bbox_inches='tight')