import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Configuración general de estilo ──────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#2e3147",
    "axes.labelcolor":  "#c8cde0",
    "axes.grid":        True,
    "grid.color":       "#2e3147",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "xtick.color":      "#7b82a8",
    "ytick.color":      "#7b82a8",
    "text.color":       "#c8cde0",
    "font.family":      "monospace",
    "legend.framealpha": 0.15,
    "legend.edgecolor": "#2e3147",
})

COLORS = {
    "mnist":  "#00d4ff",   # cian eléctrico
    "cifar10": "#ff6b6b",  # rojo-coral
}
MARKERS = {"mnist": "o", "cifar10": "s"}

# ── Carga y limpieza de datos ─────────────────────────────────────────────────
df = pd.read_csv("training_results/experiments_summary.csv")
df["dataset"] = df["dataset"].str.lower().str.strip()

# Promedio por (dataset, epochs) para que no haya puntos duplicados superpuestos
agg = (
    df.groupby(["dataset", "epochs"])
    .agg(
        accuracy=("final_test_accuracy", "mean"),
        total_time_minutes=("total_time_minutes", "mean"),
        n=("final_test_accuracy", "count"),
    )
    .reset_index()
)

datasets = sorted(agg["dataset"].unique())


def plot_panel(ax, x_col, y_col, ylabel, title, yformat=None):
    """Dibuja una línea por dataset en el eje `ax`."""
    for ds in datasets:
        sub = agg[agg["dataset"] == ds].sort_values(x_col)
        color  = COLORS.get(ds, "#ffffff")
        marker = MARKERS.get(ds, "D")

        ax.plot(
            sub[x_col], sub[y_col],
            color=color, marker=marker,
            linewidth=2.2, markersize=8,
            markeredgewidth=1.5, markeredgecolor="#0f1117",
            label=ds.upper(), zorder=3,
        )
        # Etiquetas de valor sobre cada punto
        for _, row in sub.iterrows():
            val = row[y_col]
            label_text = f"{val:.1f}" if y_col == "total_time_minutes" else f"{val:.3f}"
            ax.annotate(
                label_text,
                xy=(row[x_col], val),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=7.5, color=color, alpha=0.85,
            )

    ax.set_title(title, fontsize=13, fontweight="bold",
                 color="#e0e4f5", pad=14)
    ax.set_xlabel("Épocas de entrenamiento", fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=8)
    if yformat:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(yformat))
    ax.legend(fontsize=10, loc="best")
    ax.spines[["top", "right"]].set_visible(False)


# ── Figura principal ──────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(15, 6),
    gridspec_kw={"wspace": 0.32},
)
fig.patch.set_facecolor("#0f1117")

# ── Gráfico 1 · Precisión vs Épocas ──────────────────────────────────────────
plot_panel(
    ax1,
    x_col="epochs",
    y_col="accuracy",
    ylabel="Precisión en test (accuracy)",
    title="Precisión final  ·  MNIST vs CIFAR-10",
    yformat=lambda v, _: f"{v:.0%}",
)
ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

# Banda de referencia — "buena" precisión
ax1.axhspan(0.85, 1.0, color="#00d4ff", alpha=0.04, zorder=0)
ax1.axhline(0.85, color="#00d4ff", linewidth=0.7, linestyle=":", alpha=0.4)
ax1.text(
    ax1.get_xlim()[0] if ax1.get_xlim()[0] > 0 else 0,
    0.855, "  85 % ref.", fontsize=7.5, color="#00d4ff", alpha=0.5,
)

# ── Gráfico 2 · Tiempo total vs Épocas ───────────────────────────────────────
plot_panel(
    ax2,
    x_col="epochs",
    y_col="total_time_minutes",
    ylabel="Tiempo total de entrenamiento (min)",
    title="Tiempo de entrenamiento  ·  MNIST vs CIFAR-10",
)

# ── Pie de figura ─────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.01,
    "Fuente: experimentos propios  ·  LR = 0.01  ·  valores promediados por (dataset, épocas)",
    ha="center", fontsize=7.5, color="#4a5070",
)

plt.suptitle(
    "Análisis comparativo de entrenamiento de red neuronal",
    fontsize=15, fontweight="bold", color="#e8ecff", y=1.02,
)

plt.savefig(
    "training_metrics.png",
    dpi=150, bbox_inches="tight",
    facecolor="#0f1117",
)
print("✓ Gráfico guardado como training_metrics.png")
plt.show()