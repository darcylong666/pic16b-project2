# EpiScope

## Overview
EpiScope is a interactive project that models COVID-19 dynamics using SEIS-based ODEs combined with real-world policy and vaccination data. The goal is to show how different interventions, such as school closures, workplace restrictions, travel limits, and vaccination rollout, affect transmission rates over time.  

## Repository Structure
- **UnifiedSEISPolicyModel.py** — Implements the SEIS model with policy index & vaccination.  
- **GlobalParameterOptimizer.py** — Fits global parameters (`beta_base`, `sigma`, `gamma`, `alpha`) using L-BFGS-B across all countries.  
- **SEIS model.ipynb** — Main notebook that cleans data, runs the model, performs optimization, and produces the figures in the report.  
- **vaccinations_global.csv.zip** — Raw vaccination dataset (aggregated later in the notebook).  
- **interactive_plot.html / refined.html** — Saved Plotly visualizations from earlier stages.  
- **README.md** — Usage instructions.  

## Data Used
The notebook loads four datasets from Our World in Data:
- COVID cases & deaths  
- Population data (used as total population `N`)  
- Policy data (six NPIs: school closure, workplace closure, gathering limits, etc.)  
- Vaccination data (aggregated monthly)  

All country names are standardized to **ISO-3166 codes**.

## Core Components

### 1. Unified SEIS–Policy–Vaccination Model

This model simulates each country with shared parameters.

- **SEIS compartments:** `S → E → I → S`
- **Policy effect:** `beta(t) = beta_base * exp(-alpha * PI(t))`
- **Vaccination effect:** `- epsilon * v(t) * S(t)`

**Main interface:**
```python
from UnifiedSEISPolicyModel import UnifiedSEISPolicyModel

model = UnifiedSEISPolicyModel(beta_base, sigma, gamma, alpha)
I_pred = model.simulate_country(country_df, policy_df, vacc_df)
```

### 2. Global Parameter Optimization

The optimizer fits global parameters using monthly incidence from ~214 countries.
```python
from GlobalParameterOptimizer import GlobalParameterOptimizer

optimizer = GlobalParameterOptimizer(covid_data, policy_data, vacc_data)
result = optimizer.optimize_parameters()
optimal_params = result.x
```

This produces the parameter set used in all case studies (USA, China, etc.).
## How to Run the Project

1. Clone the repository and open `SEIS model.ipynb`.
2. Run the notebook top-to-bottom.
   - Data will be cleaned and merged automatically
   - The global optimizer will run
   - Figures for the final report will be generated

No special setup is required beyond standard scientific Python packages (`numpy`, `pandas`, `scipy`, `plotly`, `ipywidgets`).

## Reproducing Report Figures

All figures in the final write-up — such as:

- SEIS model curves
- `beta(t)` and `R_eff(t)`
- Policy index over time
- Case studies for the U.S. and China

…are generated directly by running `SEIS model.ipynb`.


## Group Contributions Statement
**Weimo Zhu:**
- Processed, merged, and standardized population and vaccination datasets
- Derived and implemented the SEIS ordinary differential equations based on relevant literature
- Incorporated vaccination effects into the SEIS model formulation
- Developed visualizations
- Conducted model evaluation using multiple error metrics
- Performed detailed case studies for the United States and China to analyze model performance and limitations
  
**Darcy Long:**
 - Cleaned and processed the cases & deaths dataset and the policy dataset, and introduced the use of ISO country codes
 - Built the SIR model and the SEIS model framework
 - Modeled the policy effects on transmission rate
 - Solved the ODEs and optimized the parameter with L-BFGS-B
 - Created visualizations of COVID dynamics and built the framework of the interactive plots for SIR and SEIS simulations
 - Built the comprehensive plots to analyze final results

