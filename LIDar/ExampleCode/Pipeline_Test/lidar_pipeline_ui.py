import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import ipywidgets as widgets
from IPython.display import display, clear_output

# ─── helpers ──────────────────────────────────────────────────────────────────

def normalize_columns(df):
    """lowercase + underscore all column names"""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )
    return df


def run_pipeline(filepath, max_toggle_rate, min_toggle_rate, dead_time_ns, bin_shift):
    """Run the full lidar pipeline and return (df_sig, slope, offset, r2)."""

    # ── config (fixed params) ─────────────────────────────────────────────────
    config = {
        "bin_width_ns": 25,
        "bin_spacing_m": 3.75,
        "prf_hz": 20,
        "bg_start_m": 0,
        "bg_end_m": 3750,
        "afterpulse_provided": True,
        "sig_start": 3840,
        "sig_end": 18836.25,
        "overlap_r1_m": 200,
        "overlap_r2_m": 300,
        "overlap_function_r1_m": 50,
        "overlap_function_r2_m": 800,
        "overlap_function_k": 0.01,
        "overlap_min": 0.2,
        "energy_mj": 25,
        # user params
        "dead_time_ns": dead_time_ns,
        "min_toggle_rate": min_toggle_rate,
        "max_toggle_rate": max_toggle_rate,
        "bin_shift": bin_shift,
    }

    # ── load & normalise columns ──────────────────────────────────────────────
    df = pd.read_csv(filepath)
    df = normalize_columns(df)

    # ── bin index & range ─────────────────────────────────────────────────────
    df.insert(0, "bin_index", np.arange(len(df)))
    df.insert(1, "range_m", df["bin_index"] * config["bin_spacing_m"])

    # ── bin shift ─────────────────────────────────────────────────────────────
    df["photon_counting_shifted"] = df["photon_counting"].shift(config["bin_shift"])
    df["pc_sterr_shifted"] = df["pc_sterr"].shift(config["bin_shift"])

    # ── SNR ───────────────────────────────────────────────────────────────────
    df["snr_analog"] = df["analog"] / df["analog_sterr"]
    df["snr_photon"] = df["photon_counting_shifted"] / df["pc_sterr_shifted"]

    # ── dead-time correction ──────────────────────────────────────────────────
    dead_time_s = config["dead_time_ns"] * 1e-9
    rate_meas_hz = df["photon_counting_shifted"] * 1e6
    ratio = rate_meas_hz * dead_time_s
    df["photon_deadtime_corr_hz"] = np.where(ratio < 1, rate_meas_hz / (1 - ratio), np.nan)
    df["photon_deadtime_corr"] = df["photon_deadtime_corr_hz"] / 1e6

    # ── signal region & background ───────────────────────────────────────────
    df_sig = df[
        (df["range_m"] >= config["sig_start"]) &
        (df["range_m"] <= config["sig_end"])
    ].copy()

    bg_region = df[
        (df["range_m"] >= config["bg_start_m"]) &
        (df["range_m"] <= config["bg_end_m"])
    ]
    bg_region = bg_region[
        np.isfinite(bg_region["analog"]) &
        np.isfinite(bg_region["photon_deadtime_corr"])
    ]
    analog_bg_mean = bg_region["analog"].mean()
    photon_bg_mean = bg_region["photon_deadtime_corr"].mean()

    # ── re-zero range ─────────────────────────────────────────────────────────
    df_sig["bin_index"] = df_sig["bin_index"] - df_sig["bin_index"].iloc[0]
    df_sig["range_m"] = df_sig["bin_index"] * config["bin_spacing_m"]

    df_sig["analog_bg_corr"] = df_sig["analog"] - analog_bg_mean
    df_sig["photon_bg_corr"] = df_sig["photon_deadtime_corr"] - photon_bg_mean

    # ── afterpulse correction ─────────────────────────────────────────────────
    r = df_sig["range_m"].to_numpy()
    df_sig["afterpulse_sim"] = (
        30.0 * np.exp(-r / 60.0) +
        8.0  * np.exp(-r / 250.0)
    )
    df_sig["photon_apcorr_counts"] = df_sig["photon_deadtime_corr"] - df_sig["afterpulse_sim"]

    # ── regression (toggle window) ────────────────────────────────────────────
    mask_overlap = (
        (df_sig["photon_deadtime_corr"] >= config["min_toggle_rate"]) &
        (df_sig["photon_deadtime_corr"] <= config["max_toggle_rate"]) &
        np.isfinite(df_sig["analog"]) &
        np.isfinite(df_sig["photon_deadtime_corr"])
    )
    m_valid = df_sig.loc[mask_overlap, ["analog", "photon_deadtime_corr"]].copy()

    slope = offset = r2 = np.nan
    if len(m_valid) >= 2:
        x = m_valid["analog"].to_numpy()
        y = m_valid["photon_deadtime_corr"].to_numpy()
        slope, offset, r_val, *_ = linregress(x, y)
        r2 = r_val ** 2
        df_sig["analog_scaled_for_glue"] = slope * df_sig["analog"] + offset
    else:
        df_sig["analog_scaled_for_glue"] = np.nan

    # ── glue weights ──────────────────────────────────────────────────────────
    blend_r1 = config["overlap_r1_m"] - 100
    blend_r2 = config["overlap_r2_m"] + 100
    pc = df_sig["photon_counting"].to_numpy()
    w = np.ones(len(r), dtype=float)
    m_blend = (r >= blend_r1) & (r <= blend_r2)
    w[m_blend] = 0.5 * (1.0 - np.cos(np.pi * (r[m_blend] - blend_r1) / (blend_r2 - blend_r1)))
    w[pc > config["max_toggle_rate"]] = 0.0
    w[pc < config["min_toggle_rate"]] = 1.0

    df_sig["weight_w"] = w
    df_sig["merged_sig"] = (1.0 - w) * df_sig["analog_scaled_for_glue"] + w * df_sig["photon_deadtime_corr"]

    # ── overlap correction ────────────────────────────────────────────────────
    r0 = 0.5 * (config["overlap_function_r1_m"] + config["overlap_function_r2_m"])
    k = config["overlap_function_k"]
    r1 = config["overlap_function_r1_m"]
    r2_ov = config["overlap_function_r2_m"]
    L = 1.0 / (1.0 + np.exp(-k * (r - r0)))
    L1 = 1.0 / (1.0 + np.exp(-k * (r1 - r0)))
    L2 = 1.0 / (1.0 + np.exp(-k * (r2_ov - r0)))
    overlap = np.clip((L - L1) / (L2 - L1), 0.0, 1.0)
    overlap = np.maximum(overlap, config["overlap_min"])
    df_sig["overlap_func"] = overlap
    df_sig["merged_counts_ovcorr"] = df_sig["merged_sig"] / df_sig["overlap_func"]

    # ── range² correction ─────────────────────────────────────────────────────
    df_sig["range2_corrected_counts"] = df_sig["merged_sig"] * df_sig["range_m"] ** 2
    df_sig["range2_corrected_counts_overlap"] = df_sig["merged_counts_ovcorr"] * df_sig["range_m"] ** 2

    e_joule = config["energy_mj"] * 1e-3
    df_sig["nrb"] = df_sig["range2_corrected_counts"] / e_joule
    max_val = np.nanmax(df_sig["nrb"].to_numpy())
    df_sig["range2_norm"] = df_sig["nrb"] / max_val

    return df_sig, slope, offset, r2


def make_plot(df_sig):
    """Plot merged_sig (glued profile), photon_deadtime_corr, and analog."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df_sig["range_m"],
        df_sig["merged_sig"],
        color="#1a6faf",
        linewidth=2,
        linestyle="--",
        label="merged_sig (glued profile)",
        zorder=3,
    )
    ax.plot(
        df_sig["range_m"],
        df_sig["photon_deadtime_corr"],
        color="#e05c2d",
        linewidth=1.2,
        alpha=0.85,
        label="photon_deadtime_corr",
        zorder=2,
    )
    ax.plot(
        df_sig["range_m"],
        df_sig["analog_scaled_for_glue"],
        color="#2ca05a",
        linewidth=1.2,
        alpha=0.85,
        label="analog (scaled for glue)",
        zorder=2,
    )

    ax.set_xlabel("Range (m)", fontsize=13)
    ax.set_ylabel("Signal (MHz)", fontsize=13)
    ax.set_title("LIDAR Signal — Merged / Photon deadtime corr / Analog", fontsize=14)
    ax.set_yscale("log")
    ax.set_xlim(0, df_sig["range_m"].max())
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    return fig


# ─── widgets ──────────────────────────────────────────────────────────────────

style = {"description_width": "160px"}
layout = widgets.Layout(width="420px")

w_file = widgets.Text(
    value="",
    placeholder="/path/to/your/file.csv",
    description="File path:",
    style=style, layout=layout,
)
w_max_toggle = widgets.FloatSlider(
    value=10.0, min=1.0, max=50.0, step=0.5,
    description="Max toggle rate:", style=style, layout=layout,
    readout_format=".1f",
)
w_min_toggle = widgets.FloatSlider(
    value=0.5, min=0.01, max=5.0, step=0.05,
    description="Min toggle rate:", style=style, layout=layout,
    readout_format=".2f",
)
w_dead_time = widgets.FloatSlider(
    value=3.06, min=0.5, max=20.0, step=0.01,
    description="Dead time (ns):", style=style, layout=layout,
    readout_format=".2f",
)
w_bin_shift = widgets.IntSlider(
    value=0, min=-10, max=10, step=1,
    description="Bin shift:", style=style, layout=layout,
)

btn_run = widgets.Button(
    description="Run pipeline",
    button_style="primary",
    layout=widgets.Layout(width="160px", height="36px"),
)

out_metrics = widgets.Output()
out_plot    = widgets.Output()

def on_run(_):
    with out_metrics:
        clear_output(wait=True)
    with out_plot:
        clear_output(wait=True)

    fp = w_file.value.strip()
    if not fp:
        with out_metrics:
            print("⚠  Please enter a file path.")
        return

    try:
        df_sig, slope, offset, r2 = run_pipeline(
            filepath=fp,
            max_toggle_rate=w_max_toggle.value,
            min_toggle_rate=w_min_toggle.value,
            dead_time_ns=w_dead_time.value,
            bin_shift=w_bin_shift.value,
        )
    except Exception as e:
        with out_metrics:
            print(f"❌  Error: {e}")
        return

    with out_metrics:
        print(f"{'Slope':<14}: {slope:.6f}")
        print(f"{'Offset':<14}: {offset:.6f}")
        print(f"{'Fit quality R²':<14}: {r2:.6f}")

    with out_plot:
        fig = make_plot(df_sig)
        plt.show()
        plt.close(fig)

btn_run.on_click(on_run)

# ─── layout ───────────────────────────────────────────────────────────────────
panel = widgets.VBox([
    widgets.HTML("<h3 style='margin:0 0 8px'>LIDAR Pipeline</h3>"),
    w_file,
    w_max_toggle,
    w_min_toggle,
    w_dead_time,
    w_bin_shift,
    btn_run,
    widgets.HTML("<b>Results</b>"),
    out_metrics,
    out_plot,
])

display(panel)
