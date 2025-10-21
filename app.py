# app.py
# Streamlit app demonstrating a simplified synaptic-facilitation WM model
# Based on Mongillo, Barak & Tsodyks (2008) ideas (u,x dynamics, readout/reactivation).
#
# Requirements:
#   pip install -r requirements.txt
#
# Run:
#   streamlit run app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Synaptic Facilitation WM (interactive)")

st.title("Interactive demo: Synaptic facilitation as activity-silent working memory")
st.write(
    "Simplified interactive simulation inspired by Mongillo et al. (2008). "
    "Synapses have short-term facilitation (u) and depression (x). "
    "Spikes increase u and consume x; u decays slowly (residual Ca²⁺) enabling activity-silent storage."
)

# Sidebar controls
st.sidebar.header("Simulation parameters")

T = st.sidebar.slider("Total simulation time (ms)", 500, 8000, 3000, step=100)
dt = st.sidebar.number_input("Time step dt (ms)", value=1.0, min_value=0.1, max_value=5.0, step=0.1)

# Network/population sizes (kept modest for interactivity)
N_exc = st.sidebar.slider("Total excitatory neurons (N_exc)", 100, 2000, 500, step=50)
pop_size = st.sidebar.slider("Size of each selective population", 5, max(5, int(N_exc/4)), 40, step=1)
n_populations = 2  # fixed: demonstrate multiple items
n_nonselective = N_exc - n_populations * pop_size
if n_nonselective < 0:
    st.sidebar.error("Increase N_exc or decrease population size")
    st.stop()

# Synaptic plasticity parameters
st.sidebar.subheader("Synaptic dynamics (u, x)")
U = st.sidebar.slider("U (facilitation increment per spike)", 0.01, 1.0, 0.2, step=0.01)
tau_f = st.sidebar.slider("tau_f (facilitation decay ms)", 100.0, 5000.0, 1500.0, step=50.0)
tau_d = st.sidebar.slider("tau_d (recovery from depression ms)", 20.0, 1000.0, 200.0, step=10.0)

# Baseline rates and gains
st.sidebar.subheader("Firing & drive")
baseline_rate = st.sidebar.slider("Baseline firing rate (Hz)", 0.1, 10.0, 1.0, step=0.1)
J0 = st.sidebar.slider("Baseline synaptic strength J0 (arbitrary gain)", 0.0, 1.0, 0.6, step=0.01)
gain = st.sidebar.slider("gain (how strongly J_eff affects firing prob)", 0.0, 50.0, 20.0, step=0.5)

# Stimulus / readout controls
st.sidebar.subheader("Stimulus / readout")
stim_amp = st.sidebar.slider("Stimulus extra drive (Hz)", 0.0, 200.0, 40.0, step=1.0)
stim_start = st.sidebar.slider("Stimulus start (ms)", 0, T-50, 50, step=10)
stim_dur = st.sidebar.slider("Stimulus duration (ms)", 10, 1000, 200, step=10)
readout_time = st.sidebar.slider("Readout pulse time (ms)", stim_start, T, min(stim_start + 800, T), step=10)
readout_amp = st.sidebar.slider("Readout amplitude (Hz)", 0.0, 200.0, 15.0, step=1.0)

# Random seed
seed = st.sidebar.number_input("Random seed (0 for random)", min_value=0, value=42, step=1)
if seed != 0:
    rng = np.random.default_rng(int(seed))
else:
    rng = np.random.default_rng()

# Quick help / assumptions
st.markdown(
    "**Model notes:**\n"
    "- Simplified Poisson-like spiking probability: p_spike(dt) ≈ dt*(baseline_rate + gain * J_eff)\n"
    "- u(t) ~ utilization (proxy for residual calcium), increases by U*(1-u) on a presynaptic spike and decays with tau_f\n"
    "- x(t) ~ available resources (vesicle depletion), decreased by u*x on spike and recovers with tau_d\n"
    "- We simulate two selective populations (two items) and the rest nonselective background neurons."
)

# Build population indices
pop_indices = [np.arange(i * pop_size, (i + 1) * pop_size) for i in range(n_populations)]
nonselective_indices = np.arange(n_populations * pop_size, N_exc)

# Initialize variables
time = np.arange(0, T, dt)
n_time = len(time)

# Initialize u and x for synapses of each neuron (we track synapses as per presyn neuron)
u = np.ones(N_exc) * U  # start at baseline U
x = np.ones(N_exc)      # start fully recovered
# store history of mean quantities for selective populations
u_hist = np.zeros((n_populations, n_time))
x_hist = np.zeros((n_populations, n_time))
Jhist = np.zeros((n_populations, n_time))

# spike raster: boolean matrix (N_exc x n_time)
spikes = np.zeros((N_exc, n_time), dtype=bool)

# precompute decay factors per dt
exp_f = np.exp(-dt / tau_f)
exp_d = np.exp(-dt / tau_d)

# Stimulus mask
stim_mask = (time >= stim_start) & (time < stim_start + stim_dur)
readout_mask = (time >= readout_time) & (time < readout_time + dt)  # single dt pulse

# Simulation loop
# We'll drive only the first selective population with the stimulus (load item)
target_pop = 0
for t_idx, t in enumerate(time):
    # compute instantaneous drive (Hz)
    drive = np.zeros(N_exc) + baseline_rate

    # background nonspecific drive shaping: small noise-driven variations
    drive += rng.normal(0.0, 0.5, size=N_exc)

    # stimulus: boost the target population rate during stim_mask
    if stim_mask[t_idx]:
        drive[pop_indices[target_pop]] += stim_amp

    # readout pulse: nonspecific weak excitatory input to all excitatory neurons
    if readout_mask[t_idx]:
        drive += readout_amp

    # effective synaptic strength for each presynaptic neuron -> contributes to postsyn firing prob (simplification)
    # J_eff = J0 * u * x
    J_eff = J0 * (u * x)

    # instantaneous spike probability in dt for each neuron:
    p = (dt / 1000.0) * np.clip(drive + gain * J_eff, 0, None)
    p = np.clip(p, 0.0, 1.0)

    # draw spikes
    rand = rng.random(size=N_exc)
    sp = rand < p
    spikes[:, t_idx] = sp

    # For every presynaptic spike, update u and x (Tsodyks-Markram like)
    if sp.any():
        sp_idx = np.nonzero(sp)[0]
        u_old = u[sp_idx].copy()
        u_new = u_old + U * (1.0 - u_old)
        x_new = x[sp_idx] * (1.0 - u_new)
        u[sp_idx] = u_new
        x[sp_idx] = x_new

    # Between spikes, continuous recovery (exponential Euler)
    u = U + (u - U) * exp_f
    x = 1.0 - (1.0 - x) * exp_d

    # record population averages
    for p_idx in range(n_populations):
        inds = pop_indices[p_idx]
        u_hist[p_idx, t_idx] = u[inds].mean()
        x_hist[p_idx, t_idx] = x[inds].mean()
        Jhist[p_idx, t_idx] = J0 * (u[inds] * x[inds]).mean()

# compute population firing rates (smoothed)
bin_ms = 50
bin_steps = max(1, int(bin_ms / dt))
rates = np.convolve(spikes.sum(axis=0), np.ones(bin_steps), mode="same") / (N_exc * (dt/1000.0) * bin_steps)  # Hz
pop_rates = []
for p_idx in range(n_populations):
    inds = pop_indices[p_idx]
    r = np.convolve(spikes[inds].sum(axis=0), np.ones(bin_steps), mode="same") / (len(inds) * (dt/1000.0) * bin_steps)
    pop_rates.append(r)

# Layout plots
col1, col2 = st.columns([1.2, 1])
with col1:
    st.subheader("Raster (subset of neurons)")
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

# Summary statistics & explanation
st.markdown("### What to look for")
st.write(
    "- After the encoding stimulus ends, check whether `u` for the target population remains elevated for ~1 s "
    "(activity-silent trace).  \n"
    "- A later small readout pulse (green dashed line) should preferentially re-activate the population whose `u` stayed elevated.  \n"
    "- Increase `tau_f` to make `u` decay more slowly; increase `tau_d` to speed recovery from depression.  \n"
    "- Increasing background drive or `gain` can move the network into regimes with spontaneous reactivations or sustained firing (see Mongillo et al.)."
)

st.markdown("### Export data")
if st.button("Download spike raster as .npz"):
    import io, base64, numpy as _np
    bio = io.BytesIO()
    _np.savez(bio, spikes=spikes, time=time, u_hist=u_hist, x_hist=x_hist, Jhist=Jhist)
    bio.seek(0)
    b64 = base64.b64encode(bio.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="synfac_sim.npz">Download simulation .npz</a>'
    st.markdown(href, unsafe_allow_html=True)