from scipy.optimize import minimize, least_squares
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from IPython.display import display, clear_output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.integrate import odeint

class UnifiedSEISPolicyModel:
    """
    This class aims to create a unified SEIS model with policy index
    """
    
    def __init__(self, beta_base=0.3, sigma=0.1, gamma=0.07, alpha=2.0, epsilon=1.0):
        # fixed parameters (same for all countries)
        self.beta_base = beta_base  
        self.sigma = sigma          # incubation rate 
        self.gamma = gamma          # recovery rate
        self.alpha = alpha          # policy effect
        
        # vaccination
        self.use_vaccination = True
        self.epsilon = epsilon      # vaccine efficacy (0-1), 1 by default (100% immune)

        # policy weight
        self.policy_config = {
            'weights': {
                'c1m_school_closing': float(1/6),
                'c2m_workplace_closing': float(1/6),
                'c3m_cancel_public_events': float(1/6),
                'c4m_restrictions_on_gatherings': float(1/6),
                'c5m_close_public_transport': float(1/6),
                'c8ev_international_travel_controls': float(1/6)
            },
            'max_values': {
                'c1m_school_closing': 3,
                'c2m_workplace_closing': 3,
                'c3m_cancel_public_events': 2,
                'c4m_restrictions_on_gatherings': 4,
                'c5m_close_public_transport': 2,
                'c8ev_international_travel_controls': 4
            }
        }
    
    def calculate_policy_index(self, policy_data):
        """Computed the policy index"""
        policy_index = np.zeros(len(policy_data))
        total_weight = 0
        
        for policy, weight in self.policy_config['weights'].items():
            if policy in policy_data.columns:
                policy_values = policy_data[policy].fillna(0).values
                max_val = self.policy_config['max_values'][policy]
                
                # normalization
                normalized = policy_values / max_val
                policy_index += normalized * weight
                total_weight += weight
        
        if total_weight > 0:
            policy_index = policy_index / total_weight
        
        return np.clip(policy_index, 0, 1)
    
    def beta_with_policy(self, policy_index):
        """write beta as a function of policy index"""
        return self.beta_base * np.exp(-self.alpha * policy_index)
    
    def simulate_country(self, country_data, policy_data, vacc_data=None, return_all=False):
        """
        Simulate each country using uniform parameters
        """
        df = country_data.copy()
        df = df.sort_values("month")
        
        policy_index = self.calculate_policy_index(policy_data)
        
        N_array = df["population"].to_numpy()
        N0 = N_array[0]
        
        I0 = max(df["infected"].iloc[0], 1)
        S_approx = N0 - I0
        E0 = (self.beta_base / self.sigma) * S_approx * I0 / N0
        E0 = max(min(E0, N0 - I0), 0)
        S0 = N0 - E0 - I0
        
        t = np.arange(len(df))
        
        policy_interp = interp1d(t, policy_index, kind='linear', fill_value=(policy_index[0], policy_index[-1]), bounds_error=False)
        
        # vaccination interpolation 
        if vacc_data is not None and self.use_vaccination:
            # align vaccination to covid months
            merged_vacc = pd.merge(df[["month"]], vacc_data[["month", "vaccinations"]], on="month", how="left")
            # missing months -> 0 vaccination
            vacc = merged_vacc["vaccinations"].fillna(0).to_numpy()
        else:
            # no vacc data -> all zeros
            vacc = np.zeros(len(df))
        
        # now vacc has same length as N_array
        vacc_rate = vacc / N_array   # per-capita vaccination rate
        
        vacc_interp = interp1d(t, vacc_rate, kind='linear', fill_value=(vacc_rate[0], vacc_rate[-1]), bounds_error=False)          
        
        sol = odeint(self._seis_equation, (S0, E0, I0), t, args=(N_array, policy_interp, vacc_interp))
        S, E, I = sol.T
        
        beta_t = np.array([self.beta_with_policy(policy_interp(t_i)) for t_i in t])
        
        R_eff = (beta_t * S) / (self.gamma * N_array)
        
        if return_all:
            return S, E, I, beta_t, R_eff, policy_index
        else:
            return I
    
    def _seis_equation(self, y, t, N_array, policy_interp, vacc_interp):
        """SEIS ODE"""
        S, E, I = y
        
        t_idx = int(min(t, len(N_array)-1))
        if t_idx < len(N_array)-1:
            frac = t - t_idx
            N = N_array[t_idx] + frac * (N_array[t_idx+1] - N_array[t_idx])
        else:
            N = N_array[t_idx]
        
        P = policy_interp(t)
        beta_t = self.beta_base * np.exp(-self.alpha * P)
        
        # vaccination rate
        v_t = vacc_interp(t)              # raw vacc rate
        v_eff = self.epsilon * v_t        # effective vacc rate (only epsilon part works)

        # SEIS with vaccination efficacy
        dSdt = -beta_t * S * I / N + self.gamma * I - v_eff * S
        dEdt = beta_t * S * I / N - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        
        return dSdt, dEdt, dIdt
