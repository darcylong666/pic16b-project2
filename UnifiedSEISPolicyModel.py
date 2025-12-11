from scipy.optimize import minimize, least_squares
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from ipywidgets import widgets, interactive_output
from IPython.display import display, clear_output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import requests
import os
import plotly.express as px
from scipy.integrate import odeint

class UnifiedSEISPolicyModel:
    """
    统一的SEIS政策模型
    
    核心假设：
    1. 所有国家的生物学参数相同（σ, γ）
    2. 基础传播率β_base相同
    3. 政策效应强度α相同
    4. 只有政策强度P(t)随国家和时间变化
    """
    
    def __init__(self, beta_base=0.3, sigma=0.1, gamma=0.07, alpha=2.0):
        # fixed parameters(same for all countries)
        self.beta_base = beta_base  
        self.sigma = sigma          # incucation rate 
        self.gamma = gamma          # recovery rate
        self.alpha = alpha          # policy effect
        
        # 政策权重配置
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
        """计算综合政策指数 (0-1范围)"""
        policy_index = np.zeros(len(policy_data))
        total_weight = 0
        
        for policy, weight in self.policy_config['weights'].items():
            if policy in policy_data.columns:
                # 获取政策值
                policy_values = policy_data[policy].fillna(0).values
                max_val = self.policy_config['max_values'][policy]
                
                # 归一化
                normalized = policy_values / max_val
                policy_index += normalized * weight
                total_weight += weight
        
        if total_weight > 0:
            policy_index = policy_index / total_weight
        
        return np.clip(policy_index, 0, 1)
    
    def beta_with_policy(self, policy_index):
        """计算政策调整后的β"""
        return self.beta_base * np.exp(-self.alpha * policy_index)
    
    def simulate_country(self, country_data, policy_data, return_all=False):
        """
        使用统一参数模拟单个国家
        
        参数在所有国家间相同，只有政策数据不同
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
        
        policy_interp = interp1d(t, policy_index,
                               kind='linear',
                               fill_value=(policy_index[0], policy_index[-1]),
                               bounds_error=False)
        
        sol = odeint(self._seis_equation, (S0, E0, I0), t,
                    args=(N_array, policy_interp))
        S, E, I = sol.T
        
        beta_t = np.array([self.beta_with_policy(policy_interp(t_i)) 
                          for t_i in t])
        
        R_eff = (beta_t * S) / (self.gamma * N_array)
        
        if return_all:
            return S, E, I, beta_t, R_eff, policy_index
        else:
            return I
    
    def _seis_equation(self, y, t, N_array, policy_interp):
        """SEIS模型方程"""
        S, E, I = y
        
        t_idx = int(min(t, len(N_array)-1))
        if t_idx < len(N_array)-1:
            frac = t - t_idx
            N = N_array[t_idx] + frac * (N_array[t_idx+1] - N_array[t_idx])
        else:
            N = N_array[t_idx]
        
        P = policy_interp(t)
        beta_t = self.beta_base * np.exp(-self.alpha * P)
        
        dSdt = -beta_t * S * I / N + self.gamma * I
        dEdt = beta_t * S * I / N - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        
        return dSdt, dEdt, dIdt