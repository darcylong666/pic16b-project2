import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize, least_squares
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from ipywidgets import widgets, interactive_output
from IPython.display import display, clear_output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from UnifiedSEISPolicyModel import UnifiedSEISPolicyModel

class GlobalParameterOptimizer:
    """
    Optimize global parameters using data from all countries
    """
    def __init__(self, covid_data, policy_data, vacc_data=None):
        self.covid_data = covid_data
        self.policy_data = policy_data
        self.vacc_data = vacc_data
        self.countries = covid_data["iso_code"].unique()
        self.country_data_cache = {}
        self.prepare_country_data()
    
    def prepare_country_data(self):
        for country in self.countries:
            #covid data
            country_covid = self.covid_data[self.covid_data["iso_code"] == country].copy()
            country_covid = country_covid.sort_values("month")
            country_covid["infected"] = country_covid["total_cases"].ffill().fillna(0)

            #policy data
            country_policy = self.policy_data[self.policy_data["iso_code"] == country].copy()
            country_policy = country_policy.sort_values("month")

            #vac data
            if self.vacc_data is not None:
                country_vacc = self.vacc_data[self.vacc_data["iso_code"] == country].copy()
                country_vacc = country_vacc.sort_values("month")
            else:
                country_vacc = None
            
            if len(country_covid) > 0 and len(country_policy) > 0:
                self.country_data_cache[country] = {'covid': country_covid, 'policy': country_policy, 'vacc': country_vacc}
            
    
    def objective_function(self, params, method='rmse'):
        """
        Minimize the error of all countries
        
        params: [beta_base, sigma, gamma, alpha]
        """
        beta_base, sigma, gamma, alpha = params
        
        model = UnifiedSEISPolicyModel(beta_base, sigma, gamma, alpha)
        
        total_error = 0
        country_count = 0
        
        for country, data in self.country_data_cache.items():
            try:
                I_pred = model.simulate_country(data['covid'], data['policy'], data['vacc'])
                I_obs = data['covid']["infected"].values
                
                skip_days = min(14, len(I_pred)//4)
                if len(I_pred) > skip_days:
                    I_pred = I_pred[skip_days:]
                    I_obs = I_obs[skip_days:]
                
                if method == 'rmse':
                    error = np.sqrt(np.mean((I_pred - I_obs) ** 2))
                elif method == 'relative_rmse':
                    epsilon = 1
                    error = np.sqrt(np.mean(((I_pred - I_obs) / (I_obs + epsilon)) ** 2))
                elif method == 'log_rmse':
                    epsilon = 1
                    error = np.sqrt(np.mean((np.log(I_pred + epsilon) - np.log(I_obs + epsilon)) ** 2))
                else:
                    error = np.mean(np.abs(I_pred - I_obs))
                
                total_error += error
                country_count += 1
                
            except Exception as e:
                print(f" {country} FAILS {str(e)}")
                continue
        
        if country_count > 0:
            return total_error / country_count
        else:
            return 1e10  
    
    def optimize_parameters(self, initial_params=None, bounds=None):
        """
        Optimize all parameters
        """
        print(f"{len(self.country_data_cache)} countries")
        
        if initial_params is None:
            initial_params = [0.3, 0.1, 0.07, 2.0]  # [beta, sigma, gamma, alpha]
        
        if bounds is None:
            bounds = [(0.01, 1.0), (0.01, 0.5), (0.01, 0.5), (0.0, 5.0)] # beta_base, sigma, gamma, alpha
                
        result = minimize(fun=lambda p: self.objective_function(p, 'relative_rmse'), x0=initial_params, 
                          bounds=bounds, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8, 'disp': True})
        
        if result.success:
            beta_base, sigma, gamma, alpha = result.x
            print(f"Optimized parameters:")
            print(f"  β_base = {beta_base:.4f}")
            print(f"  σ = {sigma:.4f}  - incubation = {1/sigma:.1f} (day)")
            print(f"  γ = {gamma:.4f}  - infection = {1/gamma:.1f} (day)")
            print(f"  α = {alpha:.4f}  - (policy effect)")
            print(f"  R0 = {beta_base/gamma:.2f}")
            print(f"  error = {result.fun:.4f}")
        else:
            print(f"FAILS!! {result.message}")
        
        return result