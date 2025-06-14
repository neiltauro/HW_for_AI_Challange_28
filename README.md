# HW_for_AI_Challange_28# Memristor Model and Simulation (Biolek Model)

## 🔹 Summary 🔹

This project involves **modeling and simulating a memristor** — a two-terminal electronic component whose resistance evolves based on its history of current flow.  
Memristors are a key building block for **neuromorphic computing and adaptable synaptic weights**.  
This makes them especially valuable for developing **self-learning circuits and chips**.

---

## 🔹 Challenge 🔹

This project was implemented to solve Challenge #28:

> Model and simulate a memristor (using the **Biolek Model**).
> Visualize its pinched I–V curve under sinusoidal voltage stimulus.

---

## 🔹 How I Got Here (Vibe Coding Procedure) 🔹

When I first started this challenge, I wanted to appreciate **memristors’ unique properties**.  
I began by researching their mathematical models, choosing **Biolek’s Model** due to its simplicity and physical significance.

I then implemented this from scratch in **Python**.  
Instead of adding a large number of files or a heavy framework, I kept it lightweight — just **Python, NumPy, and Matplotlib**.  
This lets me directly visualize their **current–voltage (I–V) characteristics** and see the **pinched hysteresis loop** form under sinusoidal stimulus.

I implemented:
- The **memristor’s state variable (w)** evolves based on **current flow and a window function**, which prevents it from growing outside its physical bounds.
- The **current** is governed by Ohm’s Law with a state-dependent resistance.

---

## 🔹 Findings and Results 🔹

Running the script produces **the well-known pinched I–V curve**, which crosses origin (V = 0, I = 0) in a symmetric trajectory:

- The slope evolves depending on its state — reflecting its memory of previous signals.
- This shows how **memristors can retain information even after the stimulus is gone**, which forms the basis for their applications in neuromorphic computing, memory storage, and adaptive circuits.

![Memristor I–V Curve](https://raw.githubusercontent.com/neiltauro/HW_for_AI_Challange_28/main/Figure_1.png)

---

## 🔹 How to Run 🔹

✅ **Requirements:**

- **Python 3.x**
- **Matplotlib**
- **NumPy**

✅ **Installation (if needed):**

```bash
pip install matplotlib numpy
```

✅ **Running the script:**  

```bash
python memristor_sim.py
```

This script will:
- Perform the numerical simulation
- Plot the **current vs. voltage (I–V) curve with pinched hysteresis**

---

## 🔹 File List 🔹

- `memristor_sim.py`: The main script for simulation
- `memristor.spice`: An example LTSpice netlist (concept; for future expansion)
- `README.md`: This document
- (Optionally) PNG plot of I–V curve (generated by script)

---

## 🔹 Conclusion 🔹

This project shows **how a memristor evolves its resistance** under stimulus — reflecting its memory.  
Such components enable **adaptive and neuromorphic computing**, adding a powerful dimension to classical circuits.

