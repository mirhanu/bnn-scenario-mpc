# BNN for System Dynamics Learning & Scenario MPC

This repository implements a **Bayesian Neural Network (BNN)** for learning system dynamics and a **Scenario-Based Model Predictive Control (MPC)** framework that uses BNN-based dynamics for robust control under uncertainty.

The dynamics are learned with a BNN, and a **Scenario-MPC** is built on top of this BNN model. Scenario MPC **uses posterior samples from the BNN** and ensures that constraints are satisfied for all sampled dynamics. The cost function is averaged over all posterior dynamics. 

As an example, the **CartPole** system is used. However, you can apply this approach to **any dynamical system** by defining a new class that inherits from the `DynamicSystem` base class. The only requirement is to implement the `dynamics()` function.

---


---

## 📂 Project Structure
```
/bnn-scenario-mpc
│── /src                 # Source code
│   │── __init__.py      # Package initialization
│   │── bnn.py           # BNN class for learning system dynamics
│   │── scenario_mpc.py  # Scenario MPC implementation
│   │── dynamic_system.py # Base class for dynamic systems
│   │── cartpole.py      # CartPole dynamics class
│── /scripts             # Scripts for running experiments
│   │── train_bnn.py     # Script to train the BNN
│   │── run_mpc.py       # Script to run MPC using the trained BNN
│── /models              # Saved models & results
│   │── posterior_samples.pth  # Saved BNN posterior samples
│── LICENSE              # MIT License
│── README.md            # Project documentation
│── requirements.txt     # Dependencies
```

---

## 📖 Usage
### **1️⃣ Train the BNN on a dynamic system**
```sh
python scripts/train_bnn.py
```
🔹 This will generate training data, run MCMC, and save posterior samples.

### **2️⃣ Run Scenario MPC using the trained BNN**
```sh
python scripts/run_mpc.py
```
🔹 Simulates the **CartPole** system using **Scenario MPC**, where the BNN's posterior samples are used as the dynamics.

---

## ⚙️ Components
### **🔹 Bayesian Neural Network (BNN)**
- Uses **Pyro** for probabilistic modeling.
- Learns system dynamics from collected data.
- Trained via **MCMC (NUTS)** to estimate posterior distributions.

### **🔹 Scenario MPC**
- Uses **multiple sampled dynamics from the BNN** for robust control.
- Optimized using **CasADi** for trajectory optimization.
- Supports bounds on **state and control inputs**.

### **🔹 Dynamic System**
- Base class for defining dynamical systems.
- Any custom dynamical system can inherit from this class by implementing `dynamics()`.
- Can simulate dynamic systems using **Runge-Kutta integration**.
- Can generate training data for the BNN.

---


## 📜 License
This project is open-source and available under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---


