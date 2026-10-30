# Physics-Constrained Machine Learning: Concepts, Tools, and Algorithms

**Victor M. Zavala and Angan Mukherjee** · Department of Chemical and Biological Engineering, University of Wisconsin-Madison

Materials of a 3-hour workshop on physics-constrained machine learning (PCML): lecture slides (PDF), Jupyter notebooks that
implement every example of the slides, and hands-on tutorials. Every notebook runs in Google Colab with one click
(no installation) or locally.

## Workshop outline

| Module | Topics | Duration | Slides | Handout (2x2) |
|---|---|---|---|---|
| 1. ML Architectures | Neural nets, neural ODEs, Gaussian processes | 50 min | [PDF](Slides/pcml_module1_architectures.pdf) | [PDF](Slides/handouts/pcml_module1_architectures_4up.pdf) |
| 2. Physics-Informed ML Architectures | Features, additive, PINN, PCNN, UDE, filtering | 45 min | [PDF](Slides/pcml_module2_physics.pdf) | [PDF](Slides/handouts/pcml_module2_physics_4up.pdf) |
| 3. Estimation and UQ | MLE, MAP, Laplace, variational inference, GP propagation | 55 min | [PDF](Slides/pcml_module3_estimation_uq.pdf) | [PDF](Slides/handouts/pcml_module3_estimation_uq_4up.pdf) |
| 4. Implementation and Case Studies | Tools, LLMs, ionic liquids, microbial communities, Bayesian optimization, solvent design | 30 min | [PDF](Slides/pcml_module4_case_studies.pdf) | [PDF](Slides/handouts/pcml_module4_case_studies_4up.pdf) |

## Notebooks

Click **Open in Colab** to run a notebook in your browser.

| Module | Notebook | Topic | Run |
|---|---|---|---|
| 1 | [`PCML_Module1_NN_NODE_CSTR`](JupyterNotebooks/PCML_Module1_NN_NODE_CSTR.ipynb) | Feedforward NN and neural ODE for a CSTR | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vzavala/PIML_PEQ_2026/blob/main/JupyterNotebooks/PCML_Module1_NN_NODE_CSTR.ipynb) |
| 1 | [`PCML_Module1_GP_CSTR`](JupyterNotebooks/PCML_Module1_GP_CSTR.ipynb) | Gaussian process for a CSTR | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vzavala/PIML_PEQ_2026/blob/main/JupyterNotebooks/PCML_Module1_GP_CSTR.ipynb) |
| 2 | [`PCML_Module2_Membrane`](JupyterNotebooks/PCML_Module2_Membrane.ipynb) | Mass balances as physics: pure NN, PINN, PCNN, additive GP | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vzavala/PIML_PEQ_2026/blob/main/JupyterNotebooks/PCML_Module2_Membrane.ipynb) |
| 2 | [`PCML_Module2_UDE_CSTR`](JupyterNotebooks/PCML_Module2_UDE_CSTR.ipynb) | Universal differential equation and physics-informed filtering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vzavala/PIML_PEQ_2026/blob/main/JupyterNotebooks/PCML_Module2_UDE_CSTR.ipynb) |
| 3 | [`PCML_Module3_Spurious_HougenWatson`](JupyterNotebooks/PCML_Module3_Spurious_HougenWatson.ipynb) | Spurious solutions and constraints on parameters (Hougen-Watson) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vzavala/PIML_PEQ_2026/blob/main/JupyterNotebooks/PCML_Module3_Spurious_HougenWatson.ipynb) |
| 3 | [`PCML_Module3_Bayes_LinearModel`](JupyterNotebooks/PCML_Module3_Bayes_LinearModel.ipynb) | Bayesian estimation: prior, likelihood, posterior | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vzavala/PIML_PEQ_2026/blob/main/JupyterNotebooks/PCML_Module3_Bayes_LinearModel.ipynb) |
| 3 | [`PCML_Module3_UQ_Membrane`](JupyterNotebooks/PCML_Module3_UQ_Membrane.ipynb) | MLE, MAP, Laplace, variational inference, GP uncertainty propagation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vzavala/PIML_PEQ_2026/blob/main/JupyterNotebooks/PCML_Module3_UQ_Membrane.ipynb) |

### Hands-on tutorials (end of workshop)

| Notebook | Topic | Run |
|---|---|---|
| [`PEQ_Colloquium_Tutorial_1`](JupyterNotebooks/PEQ_Colloquium_Tutorial_1.ipynb) | Constrained NNs for a membrane separator (PINN, PCNN with Ipopt, bootstrap) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vzavala/PIML_PEQ_2026/blob/main/JupyterNotebooks/PEQ_Colloquium_Tutorial_1.ipynb) |
| [`PEQ_Colloquium_Tutorial_2`](JupyterNotebooks/PEQ_Colloquium_Tutorial_2.ipynb) | Neural ODEs and variational inference for microkinetic modeling (JAX) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vzavala/PIML_PEQ_2026/blob/main/JupyterNotebooks/PEQ_Colloquium_Tutorial_2.ipynb) |

Tutorial 1 installs Pyomo and Ipopt automatically when run in Colab (first code cell).

## Running locally

```bash
conda create -n pcml -c conda-forge python=3.11 numpy scipy matplotlib scikit-learn pyomo ipopt jupyterlab ipykernel
conda activate pcml
python -m pip install torch "jax[cpu]"
jupyter lab
```

## Repository layout

- `Slides/`: slides of the four modules (PDF); `Slides/handouts/` has the same slides, four per landscape page, for note taking.
- `JupyterNotebooks/`: workshop notebooks and hands-on tutorials (see `JupyterNotebooks/README.md`).

## Main reference

Mukherjee, A., and Zavala, V. M. (2026). Physics-constrained machine learning for chemical engineering.
*Current Opinion in Chemical Engineering*, 51, 101228. https://doi.org/10.1016/j.coche.2026.101228
