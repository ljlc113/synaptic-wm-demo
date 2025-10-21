# Synaptic Facilitation Working Memory — Interactive Streamlit Demo

This repository contains a lightweight **Streamlit** app that demonstrates a simplified version of the *Mongillo, Barak & Tsodyks (2008)* synaptic facilitation model of working memory.

It shows how short-term synaptic facilitation (mediated by residual presynaptic calcium) can maintain an **activity-silent** memory trace — information held in synaptic state rather than continuous spiking.

---

## Background

In the Mongillo et al. (2008) model:

- Each synapse tracks two variables:
  - **u(t)** — utilization (facilitation, proportional to residual Ca²⁺)
  - **x(t)** — available neurotransmitter resources (depression)
- The effective synaptic strength is **J_eff = J₀ × u × x**
- After a burst of spikes, **u** decays slowly (~1 s), enabling short-term memory even when spiking stops.

This demo simulates those dynamics interactively.

---

## Quick Start

### 1. Clone the repository
```
git clone https://github.com/<your-username>/synaptic-wm-demo.git
cd synaptic-wm-demo
```

### 2. Create and activate a virtual environment

**macOS / Linux**
```
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**
```
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the app
```
streamlit run app.py
```

Then open the local URL shown in your terminal (usually http://localhost:8501).

---

## One-Step Setup (Optional)

You can also use the included helper scripts.

**macOS / Linux**
```
bash setup_env.sh
```

**Windows**
```
setup_env.bat
```

These scripts create `.venv`, activate it, and install all dependencies automatically.

---

## Project Structure
```
synaptic-wm-demo/
├── app.py
├── requirements.txt
├── setup_env.sh
├── setup_env.bat
├── README.md
├── LICENSE
├── .gitignore
└── .github/workflows/ci.yml
```

---

## Features

- Interactive sliders for synaptic parameters (**U**, **tau_f**, **tau_d**)
- Adjustable stimulus and readout pulses
- Raster plots and population averages (**u**, **x**, **J_eff**)
- Demonstrates:
  - Activity-silent memory (no firing, elevated u)
  - Periodic reactivation (population spikes)
  - Persistent-activity regimes

---

## How It Works

- Simplified probabilistic spiking neurons  
- Each presynaptic spike:
  - increases **u** (Ca²⁺ buildup)
  - decreases **x** (vesicle depletion)
- Between spikes:
  - **u** decays with time constant **tau_f**
  - **x** recovers with **tau_d**
- Memory reactivates when elevated **u** increases response probability to weak inputs

---

## Reference

Mongillo G., Barak O., & Tsodyks M. (2008).  
*Synaptic Theory of Working Memory.* Science 319 (5869): 1543–1546.  
https://doi.org/10.1126/science.1150769

---

## License

MIT License — see [LICENSE](LICENSE) for details.
