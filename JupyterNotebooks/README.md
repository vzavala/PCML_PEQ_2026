# Physics-Constrained Machine Learning: Concepts, Tools, and Algorithms — Notebooks

All notebooks run on a laptop (a few minutes each) or on Google Colab (use the **Open in Colab** badge at the top of each notebook). They follow the notation of the slides.

Setup (local):

    conda create -n pcml -c conda-forge python=3.11 numpy scipy matplotlib scikit-learn jupyterlab ipykernel
    conda activate pcml
    python -m pip install torch
    jupyter lab

The hands-on tutorials additionally need `pyomo` and `ipopt` (Tutorial 1) and `jax` (Tutorial 2); see the first cell of each.

## Workshop notebooks

| Module | Notebook | Concepts | Tools |
|---|---|---|---|
| 1 | `PCML_Module1_NN_NODE_CSTR` | Feedforward NN, neural ODE, training with Adam | PyTorch, SciPy |
| 1 | `PCML_Module1_GP_CSTR` | Gaussian process, kernels, MLE of hyperparameters | scikit-learn |
| 2 | `PCML_Module2_Membrane` | Mass balances: pure NN, PINN, PCNN; additive GP; learning curves | PyTorch, scikit-learn |
| 2 | `PCML_Module2_UDE_CSTR` | Universal differential equation vs. neural ODE; physics-informed filtering | PyTorch, SciPy, scikit-learn |
| 3 | `PCML_Module3_Spurious_HougenWatson` | Spurious solutions and constraints on parameters (adapted from *Statistics for Chemical Engineers*) | SciPy |
| 3 | `PCML_Module3_Bayes_LinearModel` | Prior, likelihood, posterior for a linear model (adapted from *Statistics for Chemical Engineers*) | NumPy |
| 3 | `PCML_Module3_UQ_Membrane` | MLE vs. MAP, Laplace, variational inference, GP uncertainty propagation | PyTorch, scikit-learn |

## Hands-on tutorials (end of workshop)

| Notebook | Concepts | Tools |
|---|---|---|
| `PEQ_Colloquium_Tutorial_1` | Membrane: NN, PINN, PCNN with hard constraints (Ipopt), constrained bootstrap | PyTorch, Pyomo, Ipopt |
| `PEQ_Colloquium_Tutorial_2` | CO oxidation: neural ODE, ensembles, VI for a microkinetic model | JAX |
