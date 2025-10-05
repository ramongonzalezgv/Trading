import numpy as np
import pandas as pd
import openpyxl
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as si
from scipy.optimize import minimize, brentq
from scipy.stats import norm
import time
from scipy.fft import ifft
from scipy.interpolate import interp1d

def black_price(F, K, r, sigma, T, option_type="call"):
    """
    Analytic Black formula for European options on forwards/futures.
    
    Parameters:
        F : Forward price
        K : Strike price
        r : Risk-free rate
        sigma : Volatility
        T : Time to maturity
        option_type : "call" or "put"

    Returns:
        Option price
    """
    if T == 0:
        intrinsic = max((F - K if option_type == "call" else K - F), 0.0)
        return np.exp(-r * T) * intrinsic

    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = np.exp(-r * T) * (F * si.norm.cdf(d1) - K * si.norm.cdf(d2))
    elif option_type == "put":
        price = np.exp(-r * T) * (K * si.norm.cdf(-d2) - F * si.norm.cdf(-d1))
    else:
        raise ValueError("Invalid option_type. Use 'call' or 'put'.")
    
    return price

def charfun_black(u, F, sigma, T):
    """
    Characteristic function for log-forward under Black model.
    
    Parameters:
        u : Complex argument
        F : Forward price
        sigma : Volatility
        T : Time to maturity
    """
    return np.exp(1j * u * np.log(F) - 0.5 * u ** 2 * sigma ** 2 * T)

from numpy.fft import ifft
from scipy.interpolate import interp1d

def fft_lewis_black(F, K, r, sigma, T, option_type="call", interp="cubic"):
    """
    FFT-based pricing using the Lewis method for the Black model.
    
    Parameters:
        F : Forward price
        K : Strike price
        r : Risk-free rate
        sigma : Volatility
        T : Time to maturity
        option_type : "call" or "put"
        interp : Interpolation method ("linear" or "cubic")
    
    Returns:
        Option price
    """
    N = 2**12
    B = 200
    dx = B / N
    x = np.arange(N) * dx

    weight = np.arange(N)
    weight = 3 + (-1) ** (weight + 1)
    weight[0] = 1
    weight[-1] = 1

    dk = 2 * np.pi / B
    b = N * dk / 2
    ks = -b + dk * np.arange(N)

    integrand = (
        np.exp(-1j * b * np.arange(N) * dx)
        * charfun_black(x - 0.5j, F, sigma, T)
        / (x ** 2 + 0.25)
        * weight
        * dx / 3
    )
    integral_value = np.real(ifft(integrand) * N)

    if interp == "linear":
        spline = interp1d(ks, integral_value, kind="linear", fill_value="extrapolate")
    elif interp == "cubic":
        spline = interp1d(ks, integral_value, kind="cubic", fill_value="extrapolate")
    else:
        raise ValueError("Invalid interpolation method.")

    log_moneyness = np.log(F / K)
    call_price = np.exp(-r * T) * (F - np.sqrt(F * K) / np.pi * spline(log_moneyness))

    if option_type == "call":
        return call_price
    elif option_type == "put":
        return call_price + np.exp(-r * T) * (K - F)
    else:
        raise ValueError("Invalid option_type. Use 'call' or 'put'.")
    

def charfun_heston(u: complex, kappa_h: float, theta_h: float, sigma_h: float, rho_h: float, v0: float, ttm: float, r: float, q: float) -> complex:
    """ Heston risk-neutral characteristic function. """
    d = np.sqrt((rho_h * sigma_h * u * 1j - kappa_h)**2 + (sigma_h**2) * (u * 1j + u**2))
    g = (kappa_h - rho_h * sigma_h * u * 1j - d) / (kappa_h - rho_h * sigma_h * u * 1j + d)
    
    exp_dt = np.exp(-d * ttm)
    G = (1 - g * exp_dt) / (1 - g)

    C = (r - q) * u * 1j * ttm + (kappa_h * theta_h / sigma_h**2) * ((kappa_h - rho_h * sigma_h * u * 1j - d) * ttm - 2 * np.log(G))
    D = ((kappa_h - rho_h * sigma_h * u * 1j - d) / sigma_h**2) * ((1 - exp_dt) / (1 - g * exp_dt))
    
    return np.exp(C + D * v0)


def fft_lewis_price_heston(kappa_h: float, theta_h: float, sigma_h: float, rho_h: float, v0: float, ttm: float, r: float, q: float, S: float, K: float, option_type: str, interp: str = "cubic") -> float:
    """
    Prices European call and put options using the Lewis method with FFT. Includes continuous dividend yield q.
    """
    N = 2**12  # FFT more efficient for N power of 2
    B = 200    # Integration limit
    dx = B / N
    x = np.arange(N) * dx  # Integration domain

    weight = np.arange(N)  # Simpson weights
    weight = 3 + (-1) ** (weight + 1)
    weight[0] = 1
    weight[N - 1] = 1

    dk = 2 * np.pi / B
    b = N * dk / 2
    ks = -b + dk * np.arange(N)

    # Compute the characteristic function integrand
    integrand = np.exp(-1j * b * np.arange(N) * dx) * charfun_heston(x - 0.5j, kappa_h, theta_h, sigma_h, rho_h, v0, ttm, r, q) * 1 / (x**2 + 0.25) * weight * dx / 3
    integral_value = np.real(ifft(integrand) * N)

    # Interpolation
    if interp == "linear":
        spline = interp1d(ks, integral_value, kind="linear")
    elif interp == "cubic":
        spline = interp1d(ks, integral_value, kind="cubic")

    log_moneyness = np.log(S / K)
    call_price = S * np.exp(-q * ttm) - np.sqrt(S * K) * np.exp(-r * ttm) / np.pi * spline(log_moneyness)

    if option_type == "call":
        return call_price
    elif option_type == "put":
        return call_price + K * np.exp(-r * ttm) - S * np.exp(-q * ttm)
    else:
        raise ValueError("Invalid option_type. Choose 'call' or 'put'.")


def black_model_d1_d2(F, K, T, sigma):
    """
    Calcula d1 y d2 según el modelo de Black (para opciones sobre futuros).

    Parámetros:
    F (float): Precio actual del futuro.
    K (float): Precio de ejercicio.
    T (float): Tiempo hasta el vencimiento en años.
    sigma (float): Volatilidad anualizada del futuro.

    Retorna:
    tuple: Una tupla que contiene (d1, d2).
    """
    # Evitar divisiones por cero si T es muy pequeño o sigma es cero
    if T <= 0 or sigma <= 0:
        return np.nan, np.nan # Devolver NaN si los parámetros no son válidos

    d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_call_put_price(F, K, r, sigma, T):
    """
    Analytic Black formula for European options on forwards/futures.

    Parameters:
        F : Forward price
        K : Strike price
        r : Risk-free rate
        sigma : Volatility
        T : Time to maturity

    Returns:
        tuple: (call_price, put_price)
    """
    if T == 0:
        call_intrinsic = max(F - K, 0.0)
        put_intrinsic = max(K - F, 0.0)
        discount_factor = np.exp(-r * T)
        return discount_factor * call_intrinsic, discount_factor * put_intrinsic

    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = np.exp(-r * T) * (F * si.norm.cdf(d1) - K * si.norm.cdf(d2))
    put_price = np.exp(-r * T) * (K * si.norm.cdf(-d2) - F * si.norm.cdf(-d1))
    
    return call_price, put_price

# 2. Black-76 pricing
def black76_price_OLD(F, K, T, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return max(0.0, (F - K) if option_type == "call" else (K - F))
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return F * norm.cdf(d1) - K * norm.cdf(d2)
    else:
        return K * norm.cdf(-d2) - F * norm.cdf(-d1)

def implied_vol_brent(price, F, K, T, option_type):
    if price <= 0 or T <= 0:
        return np.nan
    def objective(sigma):
        return black76_price_OLD(F, K, T, sigma, option_type) - price
    try:
        return brentq(objective, 1e-6, 5.0, maxiter=100, xtol=1e-6)
    except:
        return np.nan
    
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# Black-76 pricing formula
def black76_price(F, K, T, sigma, r, option_type):
    """ Black-76 option pricing formula. """
    if sigma <= 0 or T <= 0:
        return 0.0
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == 'put':
        price = np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    else:
        raise ValueError("Invalid option_type. Must be 'call' or 'put'.")
    return price

def implied_volatility(F, K, T, price, r, option_type):
    """
    Computes the implied volatility for an option using the Black-76 model.
    
    Parameters:
        F (float): Forward price.
        K (float): Strike price.
        T (float): Time to maturity (in years).
        price (float): Observed market price of the option.
        r (float): Risk-free interest rate.
        option_type (str): 'call' for call option, 'put' for put option.
    
    Returns:
        float: Implied volatility.
    """
    # Objective function: The difference between observed price and model price
    def objective(sigma):
        return black76_price(F, K, T, sigma, r, option_type) - price
    
    # Define bounds for volatility
    min_vol = 1e-4  # Minimum volatility (0.01%)
    max_vol = 4.0   # Maximum volatility (400%)
    
    try:
        # Use Brent's method to find the root of the objective function
        implied_vol = brentq(objective, min_vol, max_vol, xtol=1e-6, maxiter=100)
        return implied_vol
    except ValueError:
        # If no solution is found, return NaN
        return np.nan

# Implied volatility function
def implied_volatility_row(row, r, option_type):
    """ Computes implied volatility for a row of a DataFrame. """
    F = row['ATM_Forward']
    K = row['STRIKE_PRC']
    T = row['TTM']
    price = row['Heston_call_price']
    
    # Define the objective function
    def objective(sigma):
        return black76_price(F, K, T, sigma, r, option_type) - price

    # Bounds for volatility
    min_vol = 1e-4  # Minimum volatility
    max_vol = 4.0   # Maximum volatility

    try:
        # Use Brent's method to find the root
        implied_vol = brentq(objective, min_vol, max_vol, xtol=1e-6, maxiter=100)
        return implied_vol
    except ValueError:
        # If no solution is found, return NaN
        return np.nan
    
def heston_rmse_loss(params, data, r):
    kappa, theta, sigma, rho, v0 = params
    squared_errors = []

    for row in data.itertuples():
        T = row.TTM
        K = row.STRIKE_PRC
        F0 = row.ATM_Forward  # ATM Forward
        market_price = row.call_price  # call price (or row.call_price)

        try:
            model_price = fft_lewis_price_heston(
                kappa, theta, sigma, rho, v0,
                ttm=T, r=r, q=r,  # q = r because F = S * exp((r - q) * T)
                S=F0, K=K, option_type="call"
            )
            squared_errors.append((model_price - market_price)**2)
        except Exception as e:
            continue  # Skip options where pricing fails

    if not squared_errors:
        return np.inf  # Avoid invalid calibration

    rmse = np.sqrt(np.mean(squared_errors))
    return rmse

def heston_rmse_loss_smart(params, data, r):
    kappa, theta, sigma, rho, v0 = params
    squared_errors = []

    for row in data.itertuples():
        T = row.TTM
        K = row.STRIKE_PRC
        F0 = row.ATM_Forward # ATM Forward
        market_call = row.call_price  # call price
        market_put = row.put_price  # put price

        if np.isnan(T) or T <= 0 or np.isnan(K) or np.isnan(F0):
            continue

        # Choose call for OTM calls (K < F0), put for OTM puts (K > F0)
        if K < F0:
            option_type = "call"
            market_price = market_call
        elif K > F0:
            option_type = "put"
            market_price = market_put
        else:
            # At-the-money: you can choose one or average both
            option_type = "call"
            market_price = 0.5 * (market_call + market_put)

        try:
            model_price = fft_lewis_price_heston(
                kappa, theta, sigma, rho, v0,
                ttm=T, r=r, q=r,  # q = r for futures
                S=F0, K=K, option_type=option_type
            )
            squared_errors.append((model_price - market_price) ** 2)
        except Exception:
            continue

    if not squared_errors:
        return np.inf

    return np.sqrt(np.mean(squared_errors))


# Combine all maturities for a global RMSE loss
def heston_global_rmse_loss(params, df_all, r):
    kappa, theta, sigma, rho, v0 = params
    errors = []
    for row in df_all.itertuples(index=False):
        F0, K, T = row.ATM_Forward, row.STRIKE_PRC, row.TTM
        c, p = row.call_price, row.put_price
        if K < F0:
            opt, mkt = "call", c
        elif K > F0:
            opt, mkt = "put",  p
        else:
            opt, mkt = "call", 0.5*(c + p)

        try:
            mdl = fft_lewis_price_heston(kappa, theta, sigma, rho, v0,
                                         T, r, r, F0, K, opt)
            errors.append((mdl - mkt)**2)
        except:
            continue
    return np.sqrt(np.mean(errors)) if errors else np.inf
    
def compute_heston_prices(row, heston_params, r, option_type):

        kappa, theta, sigma, rho, v0 = heston_params
        return fft_lewis_price_heston(
                kappa, theta, sigma, rho, v0,
                ttm=row['TTM'], r=r, q=r, S=row['ATM_Forward'], K=row['STRIKE_PRC'], option_type=option_type)

def compute_Heston_implied_vol(row, option_type):
    
    F = row['ATM_Forward']
    K = row['STRIKE_PRC']
    T = row['TTM']
    price = row['Heston_call_price']

    if price <= 0 or T <= 0:
        return np.nan
    def objective(sigma):
        return black76_price(F, K, T, sigma, option_type) - price
    try:
        return brentq(objective, 1e-6, 6.0, maxiter=300, xtol=1e-6)
    except:
        return np.nan

def plot_mkt_vs_model_iv(df,maturity,mkt_iv_col = 'LAST',model_iv_col = 'Heston_call_IV',x_axis = 'Moneyness'):

    date_object = datetime.strptime(maturity, '%Y-%m-%d')
    
    # Format the datetime object into the desired string format
    # %d: Day of the month as a zero-padded decimal number.
    # %B: Month as locale’s full name.
    # %Y: Year with century as a decimal number.
    formatted_date = date_object.strftime('%d %B %Y')

    # Plot
    plt.figure(figsize=(9, 5))
    plt.plot(df[x_axis], df[mkt_iv_col], 'o-', label="Market IV")
    plt.plot(df[x_axis], df[model_iv_col]*100, 'o-', label="Heston Model IV")
    plt.xlabel(f"{x_axis}")
    #plt.ylim(40, 80)
    plt.ylabel("Implied Volatility")
    plt.title(f"Market vs. Heston Model Implied Volatility Smile - {formatted_date}")
    plt.legend()
    plt.grid(True)
    plt.show()



