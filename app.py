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

# ----------------------------
# Introduction page (explanatory)
# ----------------------------
def introduction_page():
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
def simulation_page():
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

        # --- Add interpretive restatement of u, x, and J_eff ---
    st.header("Restatement of synaptic variables")

    st.markdown(
        """
### **u(t)** — Utilization of synaptic efficacy  
- Fraction of available resources used per presynaptic spike.  
- Reflects **residual calcium** in the presynaptic terminal:  
  - Each spike increases `u` (facilitation).  
  - Between spikes, `u` decays slowly toward baseline `U` with time constant `τ_F ≈ 1.5 s`.  
- A larger `u` → higher release probability → stronger synaptic effect.  
**Think:** the synapse’s *gain knob* that turns up with recent activity.

---

### **x(t)** — Fraction of available synaptic resources  
- Represents how much neurotransmitter supply (vesicles) remains available.  
  - Each spike consumes some resources → `x` decreases.  
  - Between spikes, vesicles are replenished → `x` recovers toward 1 with `τ_D ≈ 0.2 s`.  
- A smaller `x` → short-term **depression** (temporary weakening).  
**Think:** the synapse’s *fuel tank* that empties with use and refills over time.

---

### **Jₑₓₑₒₜₑ₋ₓₑ₋ₓₑₓₑ(t) = J₀ · u(t) · x(t)** — Effective synaptic strength  
- The instantaneous efficacy of the synapse.  
- Combines the opposite influences of facilitation and depression:  
  - `u(t)` ↑ strengthens transmission.  
  - `x(t)` ↓ weakens transmission.  
- `J₀` sets the baseline coupling strength.  

**Think:** the synapse’s *actual power output* — the balance between being **primed** (high u) and **depleted** (low x).
        """
    )

    # Summary table
    st.markdown(
        """
| Variable | Biophysical meaning | Dynamics | Functional role |
|-----------|---------------------|-----------|-----------------|
| **u(t)** | Probability of vesicle release (residual Ca²⁺) | Increases at spikes; decays slowly (τ<sub>F</sub> ≈ 1.5 s) | Short-term **facilitation** |
| **x(t)** | Fraction of available vesicles | Decreases at spikes; recovers quickly (τ<sub>D</sub> ≈ 0.2 s) | Short-term **depression** |
| **J<sub>eff</sub>(t)** | Effective synaptic efficacy | J₀ · u · x | Net synaptic strength (balance of facilitation & depression) |
        """,
        unsafe_allow_html=True,
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

    if st.button("Download simulation data (.npz)"):
        import io, base64
        bio = io.BytesIO()
        np.savez(bio, spikes=spikes, time=time, u_hist=u_hist, x_hist=x_hist, Jhist=Jhist)
        bio.seek(0)
        b64 = base64.b64encode(bio.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="synfac_sim.npz">Download simulation .npz</a>'
        st.markdown(href, unsafe_allow_html=True)

# ----------------- Begin THEME / THEORY Page Block -----------------
# Paste this block into app.py (after your imports at the top)
# If your app already has a page selector in the sidebar, replace it with the "page" selectbox below.
# Otherwise this will add a new sidebar selector that includes the existing pages "Introduction" and "Simulation".
#
# Requires: streamlit, numpy, plotly
# -----------------------------------------------------------------

import streamlit as st
import numpy as np
import plotly.graph_objects as go

def simulate_u_x(time, spike_times, U=0.3, tau_D=0.2, tau_F=1.5, dt=0.001):
    """
    Simulate u(t) and x(t) using the Mongillo et al. (2008) model.
    time : 1D array of times (s)
    spike_times : iterable of spike times (s)
    Returns u, x, J_eff arrays aligned with time
    """
    n = time.size
    u = np.zeros(n)
    x = np.zeros(n)
    u_val = U
    x_val = 1.0
    exp_F = np.exp(-dt / tau_F)
    exp_D = np.exp(-dt / tau_D)

    spike_idx = set(int(np.round(t / dt)) for t in spike_times if t >= time[0] and t <= time[-1])
    for i, t in enumerate(time):
        if i in spike_idx:
            # spike occurs
            u_val = u_val + U * (1.0 - u_val)
            x_val = x_val * (1.0 - u_val)
        # store
        u[i] = u_val
        x[i] = x_val
        # relax between spikes
        u_val = U + (u_val - U) * exp_F
        x_val = 1.0 - (1.0 - x_val) * exp_D

    J_eff = u * x
    return u, x, J_eff

def theory_page():
    st.title("Theory: Synaptic Facilitation & Network Architecture")

        # --- Display Fig.1 directly from Science.org ---
    fig1_url = "https://www.science.org/cms/10.1126/science.1150769/asset/457cb14a-ce60-47ca-b1fd-93181fd697e8/assets/graphic/319_1543_f1.jpeg"

    st.image(
        fig1_url,
        caption=(
            "Fig. 1 from Mongillo, Barak & Tsodyks (2008):  "
            "Short-term synaptic plasticity model (left) and network architecture (right)."
        ),
        use_container_width=True,
    )

    # Explanatory text
    st.header("Overview")
    st.markdown(
        """
This page explains the short-term synaptic facilitation model used by Mongillo et al. (2008) and the network architecture they simulated.
- Panel **A** (left) shows the kinetic scheme and equations for the synaptic variables `x(t)` (available resources) and `u(t)` (utilization / residual Ca).
- Panel **A** (right) shows how `u` increases (facilitation) and `x` decreases (depression) during a presynaptic spike train; synaptic efficacy is proportional to `u·x`.
- Panel **B** shows the recurrent network architecture: colored triangles are selective excitatory neurons (each color codes one memory item), open black triangles are nonselective excitatory neurons, and black circles are inhibitory neurons providing global feedback.
"""
    )

    st.header("Equations (Short-term facilitation / depression)")
    st.latex(r"\frac{dx}{dt} = \frac{1-x}{\tau_D} - u\,x\,\delta(t-t_{sp})")
    st.latex(r"\frac{du}{dt} = \frac{U - u}{\tau_F} + U(1-u)\,\delta(t-t_{sp})")
    st.markdown(
        """
- `x(t)` = fraction of available vesicles (decreases at spikes, recovers with time constant `τ_D`).
- `u(t)` = utilization / release probability (jumps at spikes, decays slowly with `τ_F`).
- `δ(t-t_{sp})` indicates instantaneous jumps at presynaptic spike times.
- Synaptic efficacy is proportional to `u(t)·x(t)`.
"""
    )

    st.header("Explanation (step-by-step)")
    st.markdown(
        """
**Encoding (presynaptic spike train):** each spike causes  
- an increase in `u` (facilitation) because of residual Ca²⁺, and  
- a decrease in `x` (vesicle use).  

**During the train:** if `u` grows faster than `x` falls, the synapse becomes *facilitated* (postsynaptic responses grow). If `x` becomes heavily depleted, you see net depression.

**Between spikes:** `x` recovers with τ_D (~0.2 s in the paper) while `u` decays slowly with τ_F (~1.5 s). Because `u` decays slowly, information can be stored in elevated `u` for ~1 s without persistent spiking (activity-silent WM).
"""
    )

    st.header("Interactive demo — u(t), x(t) and J_eff = u·x")
    st.markdown("Adjust the parameters and spike pattern to see how facilitation and depression interact.")

    # interactive controls
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        U_val = st.slider("U (per-spike increment)", min_value=0.05, max_value=0.6, value=0.30, step=0.01)
        tau_D = st.slider("τ_D (s)", min_value=0.05, max_value=1.0, value=0.2, step=0.05)
    with col2:
        tau_F = st.slider("τ_F (s)", min_value=0.2, max_value=3.0, value=1.5, step=0.1)
        dt = 0.001
    with col3:
        # pre-defined spike patterns and custom box
        pattern = st.selectbox("Spike pattern", options=["Burst (5 spikes @ 50 Hz)","Single spike","Poisson (rate 20 Hz)","Manual times"])
        if pattern == "Manual times":
            manual = st.text_input("Enter spike times (s), comma-separated", value="0.05,0.06,0.07")
            try:
                spike_times = [float(s.strip()) for s in manual.split(",") if s.strip()!='']
            except Exception:
                spike_times = [0.05,0.06,0.07]
        else:
            spike_times = None

    # construct time vector and spike times
    T = 1.0  # 1 second demo
    time = np.arange(0.0, T + dt/2, dt)

    if pattern == "Burst (5 spikes @ 50 Hz)":
        base = 0.05
        spike_times = [base + k*(1/50.0) for k in range(5)]
    elif pattern == "Single spike":
        spike_times = [0.05]
    elif pattern == "Poisson (rate 20 Hz)":
        rng = np.random.default_rng(42)
        rate = 20.0
        # generate Poisson spike train converted to event times
        p = rate * dt
        spikes = rng.random(time.size) < p
        spike_times = list(time[spikes])
    # otherwise manual handled above

    # simulate
    u, x, J = simulate_u_x(time, spike_times, U=U_val, tau_D=tau_D, tau_F=tau_F, dt=dt)

    # Build Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=u, mode="lines", name="u(t)", line=dict(color="royalblue")))
    fig.add_trace(go.Scatter(x=time, y=x, mode="lines", name="x(t)", line=dict(color="crimson")))
    fig.add_trace(go.Scatter(x=time, y=J, mode="lines", name="J_eff = u·x", line=dict(color="black")))

    # overlay spike markers
    fig.add_trace(go.Scatter(x=spike_times, y=[1.02]*len(spike_times), mode="markers", marker=dict(size=8, color="black"), name="spikes", hoverinfo="x"))

    fig.update_layout(
        height=420,
        xaxis_title="Time (s)",
        yaxis_title="u, x, J_eff (arb. units)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=40, r=10, t=30, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # show numeric summary and interpretation
    st.markdown("**Interpretation**")
    st.markdown(
        f"""
- Peak `u` after the burst: **{u.max():.3f}**  
- Minimum `x` during burst: **{x.min():.3f}**  
- Peak `J_eff` during the trial: **{J.max():.3f}**  

If `u` increases sufficiently and `x` doesn't deplete too much, `J_eff` can increase during a brief stimulus — that's facilitation. If `x` is strongly depleted, `J_eff` will fall (depression dominates).
"""
    )

    st.markdown("---")
    st.markdown("If you want this same explanation added as a static second page (e.g., a printable/markdown page) or integrated into your existing multi-page selector, tell me where your current page selector lives in `app.py` and I will provide a one-line patch you can paste in.")

# ---------- Page selector (place this AFTER your page function definitions) ----------
pages = ["Introduction", "Theory", "Simulation"]
page = st.sidebar.selectbox("Select page", pages)

if page == "Introduction":
    try:
        introduction_page()   # call existing function if defined
    except NameError:
        st.error("Introduction page is not defined. Please implement `introduction_page()` or update the selector.")
elif page == "Theory":
    try:
        theory_page()
    except NameError:
        st.error("Theory page is not defined. Please implement `theory_page()` or update the selector.")
elif page == "Simulation":
    try:
        simulation_page()
    except NameError:
        st.error("Simulation page is not defined. Please implement `simulation_page()` or update the selector.")
# -------------------------------------------------------------------------------
