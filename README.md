# 🧮**OpenIsing**

This repository aims at exploring different flavors of Ising model solvers with the overarching goal of developing
on-chip Ising machines. The codebase serves as a platform for testing, benchmarking, and evaluating various algorithms
and strategies in software and hardware.

## **Getting Started**

### **Requirements**
- **Python Version**: 3.12
- **Python-deps**: Automatically installed via `pip` using the provided setup script.

### **Setup**
The git repository can be cloned using

```bash
git clone https://github.com/sofiedeweer/openising.git
```

To use OpenIsing, we recommend setting up a Python virtual environment and using an IDE like VS Code.


## **How to get results**
### **Pre-run script**

Before running any script, some environment variables need to be set. Therefore, don't forget to run the following steps before running any code.
 
```bash
cd openising
source .setup
```
### **Examples**
There are two examples. The first example shows how to run a solver with specific parameter settings. The configuration file (YAML) [here](./ising/inputs/config/example.yaml) shows what the exact parameter settings are for this example. If you want to change some of these settings, be sure to consult the [readme](./ising/inputs/config/README.md). The example can be runned through:

```bash
python main.py
```

The second example shows how OpenIsing can be used to test different parameter settings. You can run:
```bash
python main_loop.py
```
The configuration file for this exact example can be found [here](./ising/inputs/config/example_loop.yaml). This simulation will run for the given problems and parameter values. For each correpsonding problem a histogram and boxplot are generated and stored under `./ising/outputs/<problem>/plots`.
### **Gurobi**
It is allowed to use [Gurobi](https://www.gurobi.com/), indicated by the argument `-use_gurobi`. However, it can only be used when you have an active [Gurobi license](https://www.gurobi.com/solutions/licensing/).
