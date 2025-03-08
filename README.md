# BNN for System Dynamics Learning & Scenario MPC

This repository implements a **Bayesian Neural Network (BNN)** for learning system dynamics and a **Scenario-Based Model Predictive Control (MPC)** framework that uses BNN-based dynamics for robust control under uncertainty.

The dynamics are learned with a BNN, and a **Scenario-MPC** is built on top of this BNN model. Scenario MPC **uses posterior samples from the BNN** and ensures that constraints are satisfied for all sampled dynamics. The cost function is averaged over all posterior dynamics. 

As an example, the **CartPole** system is used. However, you can apply this approach to **any dynamical system** by defining a new class that inherits from the `DynamicSystem` base class. The only requirement is to implement the `dynamics()` function.

---


## 📂 Project Structure
```
/bnn-scenario-mpc
│── Bnn.py               # Bnn class for learning system dynamics
│── ScenarioMPC.py       # Scenario MPC implementation
│── DynamicSystem.py     # Dynamic system base class 
│── CartPole.py          # Cart Pole dynamics class 
│── train_bnn.py         # Script for training the BNN
│── run_mpc.py           # Script for running MPC using trained BNN
│── requirements.txt     # Dependencies
│── README.md            # Project documentation
```

---

## 📖 Usage
### **1️⃣ Train the BNN on a dynamic system**
```sh
python train_bnn.py
```
🔹 This will generate training data, run MCMC, and save posterior samples.

### **2️⃣ Run Scenario MPC using the trained BNN**
```sh
python run_mpc.py
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


