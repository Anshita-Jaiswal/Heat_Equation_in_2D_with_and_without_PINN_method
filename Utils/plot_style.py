# Global dark theme for all figures
# - Backgrounds: black
# - Axes, labels, ticks: pure white (#ffffff)
# - No grid lines
# Applied to Matplotlib/Seaborn/Plotly (Altair if present)

# Matplotlib / Seaborn
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    # Backgrounds
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "axes.facecolor": "black",
    "savefig.facecolor": "black",

    # Axes & text colors
    "axes.edgecolor": "#ffffff",
    "axes.labelcolor": "#ffffff",
    "axes.titlecolor": "#ffffff",
    "text.color": "#ffffff",

    # Tick colors (labels and marks)
    "xtick.color": "#ffffff",
    "ytick.color": "#ffffff",

    # Remove grid entirely
    "axes.grid": False,
})

# Optional: consistent color cycle with good contrast on black
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[
    "#4cc9f0", "#f72585", "#b5179e", "#7209b7",
    "#3a0ca3", "#4895ef", "#4361ee", "#2ec4b6"
])

# Utility to enforce styling on an axes object when needed
# (useful if a library overrides defaults)
def _cf_black_axes(ax):
    for spine in ax.spines.values():
        spine.set_color("#ffffff")
    ax.tick_params(colors="#ffffff")
    ax.xaxis.label.set_color("#ffffff")
    ax.yaxis.label.set_color("#ffffff")
    ax.title.set_color("#ffffff")
    ax.grid(False)

# Seaborn (inherits Matplotlib settings)
try:
    import seaborn as sns
    # Use a non-grid style and strictly enforce white labels/ticks
    sns.set_theme(style="ticks", rc={
        "axes.facecolor": "black",
        "figure.facecolor": "black",
        "axes.edgecolor": "#ffffff",
        "axes.labelcolor": "#ffffff",
        "text.color": "#ffffff",
        "axes.grid": False,
        "xtick.color": "#ffffff",
        "ytick.color": "#ffffff",
    })
    # Some seaborn plots set tick params post-render; provide a helper
    def cf_seaborn_enforce(ax=None):
        if ax is None:
            ax = plt.gca()
        _cf_black_axes(ax)
except Exception:
    pass

# Plotly
try:
    import plotly.io as pio
    import plotly.graph_objects as go

    cf_black = go.layout.Template(
        layout=dict(
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="#ffffff"),
            xaxis=dict(
                title_font=dict(color="#ffffff"),
                tickfont=dict(color="#ffffff"),
                tickcolor="#ffffff",
                linecolor="#ffffff",
                showgrid=False,
                gridcolor="#000000",
                zeroline=False,
                zerolinecolor="#000000",
            ),
            yaxis=dict(
                title_font=dict(color="#ffffff"),
                tickfont=dict(color="#ffffff"),
                tickcolor="#ffffff",
                linecolor="#ffffff",
                showgrid=False,
                gridcolor="#000000",
                zeroline=False,
                zerolinecolor="#000000",
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffffff"),
            ),
            title=dict(font=dict(color="#ffffff")),
        )
    )
    pio.templates["cf_black"] = cf_black
    pio.templates.default = "cf_black"
except Exception:
    pass

# Altair (if used)
try:
    import altair as alt

    def cf_altair_theme():
        return {
            "config": {
                "background": "black",
                "axis": {
                    "labelColor": "#ffffff",
                    "titleColor": "#ffffff",
                    "tickColor": "#ffffff",
                    "domainColor": "#ffffff",
                    "grid": False,
                    "gridColor": "#000000",
                },
                "legend": {"labelColor": "#ffffff", "titleColor": "#ffffff"},
                "title": {"color": "#ffffff"},
            }
        }

    alt.themes.register("cf_black", cf_altair_theme)
    alt.themes.enable("cf_black")
except Exception:
    pass