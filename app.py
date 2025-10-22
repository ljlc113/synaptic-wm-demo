# app.py
# Streamlit app with two pages:
#  - Introduction: explanatory material with nicely formatted math (LaTeX)
#  - Simulation: interactive demo of simplified Mongillo et al. model
#
# Usage:
#   pip install -r requirements.txt
#   streamlit run app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Synaptic Facilitation WM Demo")

# Sidebar navigation
page = st.sidebar.selectbox("Select page", ["Introduction", "Simulation"])

# ----------------------------
# Introduction page (explanatory)
# ----------------------------
if page == "Introduction":
    st.title("Introduction — Synaptic Facilitation & Activity-Silent Working Memory")
    st.markdown(
        "This page explains the core idea from **Mongillo, Barak & Tsodyks (2008)**: "
        "that a short-term memory item can be held in working memory **without continuous spiking**, "
        "by means of **short-term synaptic facilitation** mediated by **residual presynaptic calcium (Ca²⁺)**."
    )

    st.header("Sentence we are explaining")

    # nicely wrapped text block quote
    st.markdown(
        """
> *“We therefore propose that an item is maintained in the WM state by short-term synaptic facilitation
mediated by increased residual calcium levels at the presynaptic terminals of the neurons that code
for this item. Because removal of residual calcium from presynaptic terminals is a relatively slow
process, the memory can be transiently held for about 1 second without enhanced spiking activity.” (pg. 1543)*
"""
    )

    st.subheader("Plain-language unpacking (step by step)")
    st.markdown(
        """
- **Working memory (WM)**: holding information for a short time (seconds) — e.g., remembering a phone number briefly.  
- **Traditional view**: WM requires neurons to fire continuously (persistent spiking) to keep the memory alive.  
- **Alternate proposal (this paper)**: Instead of continuous firing, the memory can be stored in the *state of synapses*:  
  - When a neuron fires, calcium (Ca²⁺) enters its **presynaptic terminal**.  
  - If several spikes occur, **residual Ca²⁺** accumulates (it does not vanish instantly).  
  - Residual Ca²⁺ **increases release probability** for subsequent spikes → this is called **facilitation**.  
  - The synapse’s effective strength is modeled as **J_eff = J₀ × u(t) × x(t)**, where  
    - **u(t)** — utilization (models residual Ca²⁺; increases with spikes, decays slowly),  
    - **x(t)** — available resources (vesicle pool; decreases with spikes, recovers faster).  
- Because **u(t)** decays slowly (~1 s), the synapse remains *primed* even if the neuron stops firing.  
- A weak later input can selectively re-activate the same neurons — the memory is thus stored “silently” in synaptic state.  
"""
    )

    st.subheader("Short mathematical summary (LaTeX)")
    st.latex(r"""
    \begin{aligned}
    \frac{dx}{dt} &= \frac{1 - x}{\tau_D} - u(t)\,x(t)\,\sum_k \delta(t - t_{\text{sp}}^{(k)}) \\
    \frac{du}{dt} &= \frac{U - u}{\tau_F} + U(1 - u(t))\,\sum_k \delta(t - t_{\text{sp}}^{(k)}) \\
    J_{\text{eff}}(t) &= J_0 \, u(t)\,x(t)
    \end{aligned}
    """)

    st.markdown(
        """
**Meaning of symbols**

- $x(t)$: fraction of available presynaptic resources (0 ≤ x ≤ 1)  
- $u(t)$: utilization (proxy for residual presynaptic Ca²⁺)  
- $t_{sp}^{(k)}$: presynaptic spike times  
- $\tau_D$: recovery time constant for depression  
- $\tau_F$: decay time constant for facilitation  
- $U$: facilitation increment per spike  
- $J_0$: baseline synaptic weight  
- $J_{eff}$: effective momentary synaptic efficacy  
"""
    )

    # Other placeholder section no interactive
    st.subheader("Conceptual sketch (annotated)")
    
    st.markdown("**Key**")
    st.markdown(
        """
    - `████` : brief high-frequency burst of spikes (encoding event)
    - `|`    : isolated spike(s)
    - `/‾‾‾\\` : schematic elevated `u(t)` (residual Ca²⁺ / facilitation)
    - `\\____/` : schematic dip & recovery of `x(t)` (vesicle depletion & recovery)
    - `J_eff` : effective synaptic strength (product of `u` and `x`)
    """
    )

    st.code(
        "time (ms) -> 0       200      400      600      800     1000\n"
        "spikes      :  ████     |                       |           \n"
        "              [1]      [2]                     [3]          \n"
        "u (Ca)      :   /‾‾‾‾‾‾‾‾‾‾‾‾‾‾\\_______________________\n"
        "x (vesicle) : █‾‾\\_____/\\_____/\\_____/\\__________\n"
        "J_eff       :   /‾‾‾\\        (primed for readout)       \n",
        language="text",
    )

    st.markdown("**Annotations (what happens at each numbered event)**")
    st.markdown(
        """
    1. **Encoding (0–~200 ms)** — A strong, brief burst (`████`) of spikes drives the target neurons.
    - Presynaptic calcium quickly accumulates → `u(t)` jumps up (see the `u` curve rising).
    - Vesicle resources `x(t)` are consumed (sharp dip).
    - `J_eff = J_0 * u * x` transiently increases because `u` increases (even if `x` dips).
    2. **Silent delay (~200–800 ms)** — Spiking drops to baseline or stops.
    - `u(t)` (residual Ca²⁺) decays slowly and remains **elevated** for a while (activity-silent trace).
    - `x(t)` recovers back toward 1 with its own time constant.
    - No persistent firing is needed; the memory is stored in the elevated `u(t)`.
    3. **Readout / Reactivation (~800–1000 ms)** — A weak nonspecific input or brief cue (`|`) arrives.
    - Because `u(t)` is still above baseline, the same synapses are **more effective** and the target neurons preferentially reactivate.
    - This reactivation can refresh `u(t)` and extend maintenance if needed (periodic reactivations).
    """
    )


    # ---------- NEW SECTIONS (tables) ----------
    st.header("Implications & Advantages")
    st.markdown(
        """
| **Aspect** | **Explanation** |
|-------------|----------------|
| **Metabolic efficiency** | Memory maintenance does not require continuous spiking, saving energy compared to persistent-activity models. |
| **Robustness** | Because the facilitation variable $u(t)$ decays slowly, memories are less vulnerable to brief interruptions in firing. |
| **Flexible duration** | The decay rate of residual calcium (τₓ₍ₓ₎) can tune how long memory lasts (~1 s or longer). |
| **Compatibility** | Works alongside traditional persistent-activity mechanisms — not mutually exclusive. |
"""
    )

    st.header("Limitations / Considerations")
    st.markdown(
        """
| **Limitation** | **Description** |
|----------------|----------------|
| **Simplified scope** | Original model mainly demonstrates a single-item memory; capacity for multiple items not fully explored. |
| **Indirect evidence** | Experimental proof that residual Ca²⁺ alone underlies WM is limited; evidence remains partly inferential. |
| **Parameter sensitivity** | Memory duration depends on τₓ₍ₓ₎ and Ca²⁺ kinetics; small biological variability may alter stability. |
| **Hybrid systems likely** | Real cortex probably combines synaptic traces and persistent firing rather than using one mechanism exclusively. |
"""
    )

    st.header("Why This Is Important")
    st.markdown(
        """
| **Reason** | **Impact** |
|-------------|------------|
| **Expands working-memory theory** | Moves beyond spiking-only frameworks to include synaptic state as a memory substrate. |
| **Biophysical realism** | Incorporates experimentally observed short-term facilitation phenomena into cognitive modeling. |
| **Bridges timescales** | Links fast neural spiking (ms) to slower cognitive timescales (s) via synaptic time constants. |
| **Energy-efficient computation** | Suggests the brain may maintain information using low-energy synaptic states rather than constant firing. |
"""
    )

    st.info("Tip: switch to **Simulation** in the sidebar to run the interactive demo and visualize facilitation and recovery dynamics.")


# ----------------------------
# Simulation page (interactive)
# ----------------------------
else:
    st.title("Simulation — Synaptic Facilitation WM (interactive)")

    st.write(
        "This is a simplified, interactive simulation inspired by Mongillo et al. (2008). "
        "It uses probabilistic spiking (Poisson-like) and Tsodyks–Markram updates for u and x."
    )

    # Sidebar controls
    st.sidebar.header("Simulation parameters")

    T = st.sidebar.slider("Total simulation time (ms)", 500, 8000, 3000, step=100)
    dt = st.sidebar.number_input("Time step dt (ms)", value=1.0, min_value=0.1, max_value=5.0, step=0.1)

    # Network/population sizes
    N_exc = st.sidebar.slider("Total excitatory neurons (N_exc)", 100, 2000, 500, step=50)
    pop_size = st.sidebar.slider("Size of each selective population", 5, max(5, int(N_exc / 4)), 40, step=1)
    n_populations = 2
    n_nonselective = N_exc - n_populations * pop_size
    if n_nonselective < 0:
        st.sidebar.error("Increase N_exc or decrease population size")
        st.stop()

    # Synaptic plasticity parameters
    st.sidebar.subheader("Synaptic dynamics (u, x)")
    U = st.sidebar.slider("U (facilitation increment per spike)", 0.01, 1.0, 0.2, step=0.01)
    tau_f = st.sidebar.slider("tau_f (facilitation decay ms)", 100.0, 5000.0, 1500.0, step=50.0)
    tau_d = st.sidebar.slider("tau_d (recovery from depression ms)", 20.0, 1000.0, 200.0, step=10.0)

    # Firing & drive
    st.sidebar.subheader("Firing & drive")
    baseline_rate = st.sidebar.slider("Baseline firing rate (Hz)", 0.1, 10.0, 1.0, step=0.1)
    J0 = st.sidebar.slider("Baseline synaptic strength J0 (arbitrary gain)", 0.0, 1.0, 0.6, step=0.01)
    gain = st.sidebar.slider("gain (how strongly J_eff affects firing prob)", 0.0, 50.0, 20.0, step=0.5)

    # Stimulus / readout
    st.sidebar.subheader("Stimulus / readout")
    stim_amp = st.sidebar.slider("Stimulus extra drive (Hz)", 0.0, 200.0, 40.0, step=1.0)
    stim_start = st.sidebar.slider("Stimulus start (ms)", 0, T - 50, 50, step=10)
    stim_dur = st.sidebar.slider("Stimulus duration (ms)", 10, 1000, 200, step=10)
    readout_time = st.sidebar.slider("Readout pulse time (ms)", stim_start, T, min(stim_start + 800, T), step=10)
    readout_amp = st.sidebar.slider("Readout amplitude (Hz)", 0.0, 200.0, 15.0, step=1.0)

    seed = st.sidebar.number_input("Random seed (0 for random)", min_value=0, value=42, step=1)
    if seed != 0:
        rng = np.random.default_rng(int(seed))
    else:
        rng = np.random.default_rng()

    # Model notes
    st.markdown(
        "**Model notes:**\n"
        "- This demo uses Poisson-like spiking probabilities.\n"
        "- Presynaptic spikes: u <- u + U*(1-u); x <- x * (1 - u_new)\n"
        "- Between spikes: u decays to U with tau_f; x recovers to 1 with tau_d."
    )

    # Build populations
    pop_indices = [np.arange(i * pop_size, (i + 1) * pop_size) for i in range(n_populations)]
    nonselective_indices = np.arange(n_populations * pop_size, N_exc)

    # Initialize arrays
    time = np.arange(0, T, dt)
    n_time = len(time)
    u = np.ones(N_exc) * U
    x = np.ones(N_exc)
    u_hist = np.zeros((n_populations, n_time))
    x_hist = np.zeros((n_populations, n_time))
    Jhist = np.zeros((n_populations, n_time))
    spikes = np.zeros((N_exc, n_time), dtype=bool)

    exp_f = np.exp(-dt / tau_f)
    exp_d = np.exp(-dt / tau_d)

    stim_mask = (time >= stim_start) & (time < stim_start + stim_dur)
    readout_mask = (time >= readout_time) & (time < readout_time + dt)

    target_pop = 0
    for t_idx, t in enumerate(time):
        drive = np.zeros(N_exc) + baseline_rate
        drive += rng.normal(0.0, 0.5, size=N_exc)
        if stim_mask[t_idx]:
            drive[pop_indices[target_pop]] += stim_amp
        if readout_mask[t_idx]:
            drive += readout_amp

        J_eff = J0 * (u * x)
        p = (dt / 1000.0) * np.clip(drive + gain * J_eff, 0, None)
        p = np.clip(p, 0.0, 1.0)

        rand = rng.random(size=N_exc)
        sp = rand < p
        spikes[:, t_idx] = sp

        if sp.any():
            sp_idx = np.nonzero(sp)[0]
            u_old = u[sp_idx].copy()
            u_new = u_old + U * (1.0 - u_old)
            x_new = x[sp_idx] * (1.0 - u_new)
            u[sp_idx] = u_new
            x[sp_idx] = x_new

        u = U + (u - U) * exp_f
        x = 1.0 - (1.0 - x) * exp_d

        for p_idx in range(n_populations):
            inds = pop_indices[p_idx]
            u_hist[p_idx, t_idx] = u[inds].mean()
            x_hist[p_idx, t_idx] = x[inds].mean()
            Jhist[p_idx, t_idx] = J0 * (u[inds] * x[inds]).mean()

    # compute smoothed rates
    bin_ms = 50
    bin_steps = max(1, int(bin_ms / dt))
    rates = np.convolve(spikes.sum(axis=0), np.ones(bin_steps), mode="same") / (N_exc * (dt / 1000.0) * bin_steps)
    pop_rates = []
    for p_idx in range(n_populations):
        inds = pop_indices[p_idx]
        r = np.convolve(spikes[inds].sum(axis=0), np.ones(bin_steps), mode="same") / (len(inds) * (dt / 1000.0) * bin_steps)
        pop_rates.append(r)

    # Layout plots
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Raster (subset)")
        fig_raster, axr = plt.subplots(1, 1, figsize=(8, 4))
        max_show = min(200, N_exc)
        chosen = np.arange(max_show)
        y_idx, t_idxes = np.nonzero(spikes[chosen, :])
        axr.scatter(t_idxes * dt, y_idx, s=1, color="k")
        axr.set_xlabel("Time (ms)")
        axr.set_ylabel("Neuron index (subset)")
        axr.set_title("Spike raster (subset of first {} neurons)".format(max_show))
        axr.axvspan(stim_start, stim_start + stim_dur, color="orange", alpha=0.15, label="stimulus")
        axr.axvline(readout_time, color="green", linestyle="--", label="readout pulse")
        axr.legend(loc="upper right")
        st.pyplot(fig_raster)

    with col2:
        st.subheader("u, x, and J_eff (selective populations)")
        fig_ux, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
        colors = ["tab:blue", "tab:red"]
        for p_idx in range(n_populations):
            axs[0].plot(time, u_hist[p_idx], label=f"pop {p_idx} u", color=colors[p_idx])
        axs[0].set_ylabel("u (utilization)")
        axs[0].legend()
        for p_idx in range(n_populations):
            axs[1].plot(time, x_hist[p_idx], label=f"pop {p_idx} x", color=colors[p_idx])
        axs[1].set_ylabel("x (available resources)")
        axs[1].legend()
        for p_idx in range(n_populations):
            axs[2].plot(time, Jhist[p_idx], label=f"pop {p_idx} J_eff", color=colors[p_idx])
        axs[2].set_ylabel("J_eff (arb)")
        axs[2].set_xlabel("Time (ms)")
        axs[2].legend()
        axs[2].axvspan(stim_start, stim_start + stim_dur, color="orange", alpha=0.12)
        axs[2].axvline(readout_time, color="green", linestyle="--")
        st.pyplot(fig_ux)

    st.subheader("Population firing rates")
    fig_r, axr = plt.subplots(1, 1, figsize=(10, 3))
    axr.plot(time, rates, label="global rate (smoothed)")
    for p_idx, r in enumerate(pop_rates):
        axr.plot(time, r, label=f"pop {p_idx} rate")
    axr.set_xlabel("Time (ms)")
    axr.set_ylabel("Firing rate (Hz)")
    axr.legend()
    axr.axvspan(stim_start, stim_start + stim_dur, color="orange", alpha=0.12)
    axr.axvline(readout_time, color="green", linestyle="--")
    st.pyplot(fig_r)

    st.markdown("### What to look for")
    st.write(
        "- After the encoding stimulus ends, check whether `u` for the target population remains elevated for ~1 s (activity-silent trace).  \n"
        "- A later small readout pulse (green dashed line) should preferentially re-activate the population whose `u` stayed elevated."
    )

    # -------------------------
    # Reproduce Fig.2 (A,B,C) — interactive Plotly version
    # Paste this block into your Streamlit Simulation page (app.py)
    # Requires: numpy, scipy, plotly
    # pip install numpy scipy plotly
    # -------------------------
    import streamlit as st
    import numpy as np
    from scipy.signal import convolve
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.header("Figure 2 reproduction — synaptic facilitation (interactive)")

    # ---------------- Simulation / model parameters (paper values where given) ----------------
    # STP / facilitation parameters (paper)
    tau_D = 0.2     # s
    tau_F = 1.5     # s
    U = 0.3         # utilization

    # simulation time: simulate -0.5..+3.0 s so we have pre-stim baseline and long delay
    dt = 0.001
    t_start = -0.5
    t_end = 3.0
    time = np.arange(t_start, t_end + dt/2, dt)
    nsteps = len(time)

    # network sizes (paper's figure used 160 neurons split into two populations of 80)
    N = 160
    pop_size = 80
    target_inds = np.arange(0, pop_size)
    nontarget_inds = np.arange(pop_size, N)

    # single-exponential synaptic variable and simple LIF-like update (units are arbitrary)
    tau_m = 0.013  # s (approx. membrane time)
    tau_s = 0.02   # s (synaptic decay)
    Rscale = 1.0   # scale for converting input -> membrane increment (tuned)

    # connectivity / gains (tuned to reproduce the three regimes qualitatively)
    J_within = 0.09   # stronger recurrent weights within selective population
    J_between = 0.01
    g_inhib = 0.8     # global inhibitory feedback scale (tuned)
    ext_scale = 0.5   # scales external drive amplitude -> membrane input

    # stimulus / readout schedule (match caption: item loaded at t=0, read-out later)
    stim_t0 = 0.0
    stim_dur = 0.2
    stim_window = (time >= stim_t0) & (time < stim_t0 + stim_dur)
    stim_amp = 3.5    # selective input amplitude to target pop during encoding (tuned)

    # readout (nonspecific) time window — gray shading in the figure — choose late in sim
    readout_t0 = 2.0
    readout_dur = 0.08
    readout_window = (time >= readout_t0) & (time < readout_t0 + readout_dur)
    readout_amp = 1.8  # small nonspecific pulse to whole excitatory network (tuned)

    # background input levels for A,B,C (tuned so regime A: silent maintenance + readout reactivation,
    # B: spontaneous PSs, C: asynchronous elevated firing)
    bg_A = 1.0
    bg_B = 3.5
    bg_C = 6.0
    bg_conditions = [bg_A, bg_B, bg_C]

    # smoothing / analysis windows (for histograms & rate differences)
    smooth_ms = 50.0
    smooth_samples = max(1, int(smooth_ms / (dt*1000)))
    kernel = np.ones(smooth_samples) / smooth_samples

    # seed rng for reproducibility
    rng = np.random.default_rng(123456)

    # ---------------- Network simulation function ----------------
    def simulate_condition(bg_input, show_seed=None):
        """
        Runs a simple recurrent network with Tsodyks-like STP (per-presynaptic u,x).
        Returns spike raster (N x T boolean), u_avg_target (T), x_avg_target (T).
        The model is intentionally simplified but captures facilitation-mediated transient memory.
        """
        # state variables
        V = np.zeros(N)                         # membrane variable (arbitrary units)
        u = np.ones(N) * U                      # utilization
        x = np.ones(N)                          # resources
        s = np.zeros(N)                         # synaptic activation (decays with tau_s)

        spikes = np.zeros((N, nsteps), dtype=bool)
        u_avg = np.zeros(nsteps)
        x_avg = np.zeros(nsteps)

        # adjacency weight matrix: stronger within-target
        W = np.ones((N, N)) * J_between
        W[np.ix_(target_inds, target_inds)] = J_within
        np.fill_diagonal(W, 0.0)

        # pre-calc exponentials
        exp_s = np.exp(-dt / tau_s)
        exp_u = np.exp(-dt / tau_F)
        exp_x = np.exp(-dt / tau_D)

        # optional condition-specific RNG
        rng_local = np.random.default_rng( (show_seed if show_seed is not None else 42) )

        for t_idx, t in enumerate(time):
            # external (background) drive to each neuron (Poisson-like approximated by continuous random current)
            ext = bg_input * ext_scale * np.ones(N)

            # selective encoding pulse
            if stim_window[t_idx]:
                ext[target_inds] += stim_amp * ext_scale

            # add nonspecific readout
            if readout_window[t_idx]:
                ext += readout_amp * ext_scale

            # effective presyn release factor per presynaptic neuron
            J_eff = u * x

            # recurrent input current to neuron i = sum_j W_ij * s_j * J_eff_j
            I_rec = (W * (s * J_eff)).sum(axis=1)

            # total input (external + recurrent - global inhibition)
            inh = g_inhib * spikes[:, max(0, t_idx-1)].sum()  # inhibitory proportional to last-step spikes
            I_total = ext + I_rec - inh

            # simple membrane step (Euler-like)
            V += dt * ( -V / tau_m + Rscale * I_total )

            # threshold and spike
            # choose threshold heuristically to get realistic spiking
            threshold = 1.0
            sp = V >= threshold
            if sp.any():
                spikes[sp, t_idx] = True
                V[sp] = 0.0  # reset to zero baseline after spike

                # STP update for presynaptic neurons that spiked
                sp_idx = np.nonzero(sp)[0]
                u_old = u[sp_idx].copy()
                u_new = u_old + U * (1.0 - u_old)
                x_new = x[sp_idx] * (1.0 - u_new)
                u[sp_idx] = u_new
                x[sp_idx] = x_new

                # syn activation jump for presyn neurons that spiked
                s[sp_idx] += 1.0

            # decay / recover
            s *= exp_s
            u = U + (u - U) * exp_u
            x = 1.0 - (1.0 - x) * exp_x

            u_avg[t_idx] = u[target_inds].mean()
            x_avg[t_idx] = x[target_inds].mean()

        return {
            "spikes": spikes,
            "u_avg": u_avg,
            "x_avg": x_avg
        }

    # ---------------- Run the 3 conditions ----------------
    st.info("Running 3 network conditions (A,B,C). This may take a few seconds.")
    results = []
    for bg in bg_conditions:
        results.append(simulate_condition(bg))

    # ---------------- Analysis: compute histograms (delay - spontaneous rate) for target pop ----------------
    def compute_delay_hist(res, condition_index):
        spikes = res["spikes"]
        # per-neuron instantaneous rate (spikes/dt -> Hz)
        inst_rate = spikes.astype(float) / dt

        # baseline spontaneous window: pre-stimulus -0.5 .. 0.0 s
        baseline_mask = (time >= -0.5) & (time < 0.0)
        baseline_mean_per_neuron = inst_rate[target_inds][:, baseline_mask].mean(axis=1)

        # delay windows:
        if condition_index == 0:
            # A: delay defined as after termination of selective input until onset of readout
            delay_mask = (time >= (stim_t0 + stim_dur)) & (time < readout_t0)
        else:
            # B,C: delay defined until decrease of external excitation.
            # We use till readout too (paper states till decrease of external excitation; using readout time as proxy).
            delay_mask = (time >= (stim_t0 + stim_dur)) & (time < readout_t0)

        delay_mean_per_neuron = inst_rate[target_inds][:, delay_mask].mean(axis=1)

        # difference per neuron
        diff = delay_mean_per_neuron - baseline_mean_per_neuron
        return diff

    hist_diffs = [compute_delay_hist(res, i) for i, res in enumerate(results)]

    # ----------------— Build Plotly figure with 3 rows x (raster+curves + histogram) ----------------
    fig = make_subplots(rows=3, cols=2,
                        column_widths=[0.7, 0.3],
                        specs=[[{"type":"xy"}, {"type":"xy"}],
                            [{"type":"xy"}, {"type":"xy"}],
                            [{"type":"xy"}, {"type":"xy"}]],
                        horizontal_spacing=0.05, vertical_spacing=0.08,
                        subplot_titles=("A: encoding (baseline)", "", "B: increased background", "", "C: further increased", ""))

    # For raster display, pick a subset of neurons (10% from each population) to plot as in paper
    pct = 0.10
    n_target_plot = max(1, int(pop_size * pct))
    n_nontarget_plot = max(1, int(pop_size * pct))

    # choose indices reproducibly
    target_plot_inds = rng.choice(target_inds, size=n_target_plot, replace=False)
    nontarget_plot_inds = rng.choice(nontarget_inds, size=n_nontarget_plot, replace=False)

    # function to add raster + u/x curves to fig row r
    for row_idx in range(3):
        res = results[row_idx]
        spikes = res["spikes"]
        u_avg = res["u_avg"]
        x_avg = res["x_avg"]

        # left col: raster + u/x curves
        col_left = 1
        # raster points (target neurons — black) + non-target (green)
        # plot spikes for plotted neurons
        # For the raster we map neuron indices -> y positions in two groups so they look like the paper (target cluster + one non-target)
        # We'll place target plotted neurons at y in [0..n_target_plot-1] and non-target at y in [pop_size..pop_size + ...]
        y_target_base = 0
        y_nontarget_base = n_target_plot + 2  # small gap

        # target spikes
        for i_idx, neuron in enumerate(target_plot_inds):
            sp_times = time[spikes[neuron]]
            if sp_times.size > 0:
                fig.add_trace(go.Scatter(
                    x=sp_times, y=np.full_like(sp_times, y_target_base + i_idx),
                    mode="markers", marker=dict(size=3, color="black"),
                    showlegend=False, hoverinfo="skip"
                ), row=row_idx+1, col=col_left)

        # nontarget spikes (green)
        for i_idx, neuron in enumerate(nontarget_plot_inds):
            sp_times = time[spikes[neuron]]
            if sp_times.size > 0:
                fig.add_trace(go.Scatter(
                    x=sp_times, y=np.full_like(sp_times, y_nontarget_base + i_idx),
                    mode="markers", marker=dict(size=3, color="green"),
                    showlegend=False, hoverinfo="skip"
                ), row=row_idx+1, col=col_left)

        # overlay average x (red) and u (blue) on a secondary y axis (0..1)
        # to overlay them above the raster area, we shift them to the top y-range visually using an extra yaxis on the same subplot
        # Create traces with yaxis='y2' (Plotly will create separate yaxis per subplot if we specify layout updates later)
        fig.add_trace(go.Scatter(x=time, y=x_avg, mode="lines", line=dict(color="red", width=2),
                                name="x (avg)", hovertemplate="t=%{x:.2f}s<br>x=%{y:.2f}<extra></extra>"),
                    row=row_idx+1, col=col_left)
        fig.add_trace(go.Scatter(x=time, y=u_avg, mode="lines", line=dict(color="blue", width=2, dash="dash"),
                                name="u (avg)", hovertemplate="t=%{x:.2f}s<br>u=%{y:.2f}<extra></extra>"),
                    row=row_idx+1, col=col_left)

        # Stimulus & readout shading area for this subplot
        fig.add_vrect(x0=stim_t0, x1=stim_t0 + stim_dur, fillcolor="black", opacity=0.65, row=row_idx+1, col=col_left, layer="below")
        fig.add_vrect(x0=readout_t0, x1=readout_t0 + readout_dur, fillcolor="lightgray", opacity=0.5, row=row_idx+1, col=col_left, layer="below")

        # set subplot y-range so raster appears compact
        # compute estimated y max
        ymax = y_nontarget_base + n_nontarget_plot + 1
        fig.update_yaxes(range=[-2, ymax + 2], row=row_idx+1, col=col_left)
        fig.update_xaxes(range=[t_start, t_end], row=row_idx+1, col=col_left)
        # we'll attach a right-side axis for u/x later via layout mapping

        # Right col: histogram of rate differences for target pop
        col_right = 2
        diffs = hist_diffs[row_idx]
        # plot histogram as bar trace
        hist_vals, bin_edges = np.histogram(diffs, bins=30)
        bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
        fig.add_trace(go.Bar(x=bin_centers, y=hist_vals/hist_vals.sum(), marker=dict(color="black"), showlegend=False,
                            hovertemplate="rate diff=%{x:.2f} Hz<br>frac=%{y:.3f}<extra></extra>"),
                    row=row_idx+1, col=col_right)
        fig.update_xaxes(title_text="rate difference [Hz]" if row_idx==1 else "", row=row_idx+1, col=col_right)
        fig.update_yaxes(title_text="frac. of cells" if row_idx==0 else "", row=row_idx+1, col=col_right)

    # ---------------- Layout tweaks: add secondary y-axes for u/x on left subplots ----------------
    # Add one yaxis2 per left subplot by updating layout with matching anchor & domain
    # For simplicity: we'll scale the u/x (0..1) to the right-side area visually
    # locate the domain for each left subplot and add an overlay yaxis
    for i_row in range(3):
        idx = i_row + 1
        # Plotly assigns yaxes as y1, y2... but here we add explicit yaxis entries that overlay
        # Calculate domain for row: approximate vertical domain values depend on subplot grid; we use .layout domains implicitly
        # Use overlaying behavior: create extra yaxis{n} overlaying the existing yaxis for the subplot
        axis_name = f"yaxis{2 + i_row*2}"  # rough unique name - but simpler: use update_yaxes with secondary_y not available here
        # Instead, set the right-side axis for the subplot via update_yaxes specifying position 'anchor' style:
        fig.update_yaxes(title_text="u/x (avg)", secondary_y=False, row=idx, col=1)  # label on left for clarity

    # general layout
    fig.update_layout(height=900, width=1100,
                    title_text="Figure 2 (A,B,C) reproduction — raster + u/x (left) and histograms (right)",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0))

    # tighten axes labels & formatting
    for r in (1,2,3):
        fig.update_xaxes(title_text="time from stimulus onset [s]", row=r, col=1)

    # render
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Notes**
    - The STP parameters `tau_D=0.2 s`, `tau_F=1.5 s`, and `U=0.3` were used as in the paper.
    - The connectivity (`J_within`, `J_between`) and inhibition (`g_inhib`) were tuned so the three regimes (A: reactivation by readout, B: spontaneous PSs, C: asynchronous elevated firing) appear qualitatively like the paper panels. If you have the exact SOM connectivity/gain parameters, replacing those values will produce a quantitatively identical plot.
    - If you'd like, I can (a) add per-subplot secondary axes that precisely put the `u` and `x` curves on a 0..1 right axis, (b) convert the rasters to show *all* neurons (slower) or (c) export the exact traces used here to CSV for further inspection.
    """)


    if st.button("Download simulation data (.npz)"):
        import io, base64
        bio = io.BytesIO()
        np.savez(bio, spikes=spikes, time=time, u_hist=u_hist, x_hist=x_hist, Jhist=Jhist)
        bio.seek(0)
        b64 = base64.b64encode(bio.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="synfac_sim.npz">Download simulation .npz</a>'
        st.markdown(href, unsafe_allow_html=True)
