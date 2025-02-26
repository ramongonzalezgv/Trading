import numpy as np
import scipy.stats as si
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import scipy.stats as ss
from functools import partial
from scipy.integrate import quad
from scipy.fftpack import fft, ifft
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from typing import Union

class BlackScholes:
    def __init__(self,
                S: float, 
                K: float, 
                T: Union[float, str], 
                sig: float, 
                r: float, 
                q: float = 0, 
                option_type: str = 'call', 
                qty: int = 1, 
                premium: Union[float, None] = None):
        
        """
        S = Spot price \n
        K = Strike or vector of strikes \n
        T = Maturity (either expressed as a date (%d-%m-%Y) or as days to expiration). Time to maturity (ttm) will be automatically calculated independent of the input \n
        sig = Annualized volatility (expressed as decimal) \n
        r = Annualized risk free interest rate (expressed as decimal) \n
        q = Annualized ividend yield \n
        option_type = call or put \n
        qty = Number of options purchased (positive for buys, negative for sales) \n
        premium = Premium paid for the option (If no premium is specified, the price given by the Black-Scholes model will be taken) \n

        """

        if np.any(S <= 0):
            raise ValueError("Spot price (S) must be positive.")
        if np.any(K <= 0):
            raise ValueError("Strike price (K) must be positive.")
        if q < 0:
            raise ValueError("Dividend yield (q) cannot be negative.")
        if not isinstance(qty, int):
            raise ValueError("Quantity (qty) must be an integer (positive for buys, negative for sales).")
        if option_type.lower() not in ["call", "put"]:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")
        
        self.S = S
        self.K = K
        self.T = T
        self.sig = sig
        self.r = r
        self.q = q
        self.qty = qty
        self.option_type = option_type.lower()

        
        if premium is None:
            self.premium = self.analytical_price
        else:
            self.premium = premium * qty
        

    @property
    def ttm(self) -> float:
        """Time to maturity of the option."""

        if isinstance(self.T, str):
            today = date.today()
            maturity_date = datetime.strptime(self.T, '%d-%m-%Y').date()
            ttm = (maturity_date - today).days / 365.0
        else:
            ttm = self.T / 365
        return ttm
    
    @property
    def maturity_date(self) -> float:
        """Maturity date of the option."""

        if isinstance(self.T, str):
            maturity_date = self.T
        else:
            maturity_date = date.today() + timedelta(days=self.T)
        return maturity_date
    
    @property
    def d1(self) -> float:
        return (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sig ** 2) * self.ttm) / (self.sig * np.sqrt(self.ttm))

    @property
    def d2(self) -> float:
        return self.d1 - self.sig * np.sqrt(self.ttm)

    @property
    def analytical_price(self) -> float:
        d1, d2 = self.d1, self.d2
        if self.option_type == "call":
            price = (self.S * np.exp(-self.q * self.ttm) * si.norm.cdf(d1) - self.K * np.exp(-self.r * self.ttm) * si.norm.cdf(d2))
        else:
            price = (self.K * np.exp(-self.r * self.ttm) * si.norm.cdf(-d2) - self.S * np.exp(-self.q * self.ttm) * si.norm.cdf(-d1))
        return price * self.qty
    
    @property
    def delta(self) -> float:
        """Delta of the option."""

        d1 = self.d1
        ttm = self.ttm
        
        if self.option_type == "call":
            delta = np.exp(-self.q * ttm) * si.norm.cdf(d1)
        else:
            delta = np.exp(-self.q * ttm) * (si.norm.cdf(d1) - 1)
        return delta*self.qty
    
    @property
    def gamma(self) -> float:
        """Gamma of the option."""

        d1 = self.d1
        ttm = self.ttm
        gamma = (np.exp(-self.q * ttm) * si.norm.pdf(d1)) / (self.S * self.sig * np.sqrt(ttm))
        return gamma*self.qty
    
    @property
    def vega(self) -> float:
        """Vega of the option."""

        d1 = self.d1
        ttm = self.ttm
        vega = self.S * np.exp(-self.q * ttm) * si.norm.pdf(d1) * np.sqrt(ttm) / 100
        return vega*self.qty
    
    @property
    def theta(self) -> float:
        """Theta of the option."""

        d1, d2 = self.d1, self.d2
        ttm = self.ttm
        
        first_term = (-self.S * si.norm.pdf(d1) * self.sig * np.exp(-self.q * ttm)) / (2 * np.sqrt(ttm))
        second_term = self.r * self.K * np.exp(-self.r * ttm) * si.norm.cdf(d2 if self.option_type == "call" else -d2)
        third_term = self.q * self.S * np.exp(-self.q * ttm) * si.norm.cdf(d1 if self.option_type == "call" else -d1)
        
        if self.option_type == "call":
            theta = (first_term - second_term - third_term) / 365
        else:
            theta = (first_term + second_term - third_term) / 365
        return theta*self.qty
    
    @property
    def rho(self) -> float:
        """Rho of the option."""

        d2 = self.d2
        ttm = self.ttm
        
        if self.option_type == "call":
            rho = (self.K * ttm * np.exp(-self.r * ttm) * si.norm.cdf(d2)) / 100
        else:
            rho = (-self.K * ttm * np.exp(-self.r * ttm) * si.norm.cdf(-d2)) / 100
        return rho*self.qty
    
    def greeks(self) -> dict:
        """Returns a dictionary of the form {Greek_name : greek_value} with all of the greeks of the option."""

        greeks = {}
        greeks['Delta'] = self.delta
        greeks['Gamma'] = self.gamma
        greeks['Theta'] = self.theta
        greeks['Vega'] = self.vega
        greeks['Rho'] = self.rho

        return greeks

    def greeks_info(self) -> None:
        """Calculates and prints the Greeks in a table format."""

        greeks = self.greeks()

        print(f"{'Greek':<10} {'Value':<10}")
        print("-" * 20)
        for greek, value in greeks.items():
            print(f"{greek:<10} {value:<10.4f}")

    
    def charfun(self,u: complex) -> complex:
        """Characteristic function of the Black-Scholes model."""

        return np.exp(1j * u * (self.r - 0.5 * self.sig**2) * self.ttm - 0.5 * u**2 * (self.sig * np.sqrt(self.ttm))**2)

    def fft_lewis_price(self, interp: str = "cubic") -> float:
        """
        Computes the price of the option using Lewis approach. To solve the integrals the Simpson's rule and the Inverse Fast Fourier Transform (IFFT) are used. Args:
        interp: Type of interpolation. It can be linear or cubic.
        
        """
        
        
        N = 2**12  # FFT more efficient for N power of 2
        B = 200  # integration limit
        dx = B / N
        x = np.arange(N) * dx  # the final value B is excluded

        weight = np.arange(N)  # Simpson weights
        weight = 3 + (-1) ** (weight + 1)
        weight[0] = 1
        weight[N - 1] = 1

        dk = 2 * np.pi / B
        b = N * dk / 2
        ks = -b + dk * np.arange(N)

        integrand = np.exp(-1j * b * np.arange(N) * dx) * self.charfun(x - 0.5j) * 1 / (x**2 + 0.25) * weight * dx / 3
        integral_value = np.real(ifft(integrand) * N)

        if interp == "linear":
            spline_lin = interp1d(ks, integral_value, kind="linear")
            prices = self.S - np.sqrt(self.S * self.K) * np.exp(-self.r * self.ttm) / np.pi * spline_lin(np.log(self.S / self.K))
        elif interp == "cubic":
            spline_cub = interp1d(ks, integral_value, kind="cubic")
            prices = self.S - np.sqrt(self.S * self.K) * np.exp(-self.r * self.ttm) / np.pi * spline_cub(np.log(self.S / self.K))
        return prices
    
    def fft_lewis_price_putcall(self, interp: str = "cubic") -> float:
        """
        Prices European call and put options using the Lewis method with FFT. Uses Simpson's Rule for integration. Args:
        interp: Type of interpolation. It can be linear or cubic.
        
        """
        N = 2**12  # FFT more efficient for N power of 2
        B = 200  # integration limit
        dx = B / N
        x = np.arange(N) * dx  # the final value B is excluded

        weight = np.arange(N)  # Simpson weights
        weight = 3 + (-1) ** (weight + 1)
        weight[0] = 1
        weight[N - 1] = 1

        dk = 2 * np.pi / B
        b = N * dk / 2
        ks = -b + dk * np.arange(N)

        integrand = np.exp(-1j * b * np.arange(N) * dx) * self.charfun(x - 0.5j) * 1 / (x**2 + 0.25) * weight * dx / 3
        integral_value = np.real(ifft(integrand) * N)

        if interp == "linear":
            spline = interp1d(ks, integral_value, kind="linear")
        elif interp == "cubic":
            spline = interp1d(ks, integral_value, kind="cubic")

        call_price = self.S - np.sqrt(self.S * self.K) * np.exp(-self.r * self.ttm) / np.pi * spline(np.log(self.S / self.K))

        if self.option_type == "call":
            return call_price
        elif self.option_type == "put":
            return call_price + self.K * np.exp(-self.r * self.ttm) - self.S
        else:
            raise ValueError("Invalid option_type. Choose 'call' or 'put'.")
        
    
    def implied_volatility(self, premium: Union[float, None] = None, tol: float =1e-6, max_iterations: int = 100000) -> float:
        """
        Computes the implied volatility given a premium.

        """
        if premium is None:
            premium = self.premium / self.qty  # Ensure we are using per-option premium
        else:
            premium = premium / self.qty

        # Define the objective function for root finding:
        def objective_function(sigma: float) -> float:
            self.sig = sigma
            return self.analytical_price / self.qty - premium

        # Use brentq root finding within reasonable volatility bounds
        try:
            implied_vol = brentq(
                objective_function,
                a=1e-6,  # Lower bound close to 0
                b=5.0,    # Upper bound (500% volatility)
                xtol=tol,
                maxiter=max_iterations
            )
            return implied_vol
        except ValueError:
            print("Implied volatility not found within bounds")
            return np.nan
        
    
    def charfun_q(self, u: complex) -> complex:
        return np.exp(1j * u * (self.r - self.q - 0.5 * self.sig**2) * self.ttm - 0.5 * u**2 * (self.sig * np.sqrt(self.ttm))**2)
    

    def fft_lewis_price_putcall_q(self, interp: str = "cubic") -> float:
        """
        Prices European call and put options using the Lewis method with FFT. Includes continuous dividend yield q. Args:
        interp: Type of interpolation. It can be linear or cubic.

        """
        N = 2**12  # FFT more efficient for N power of 2
        B = 200  # integration limit
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
        integrand = np.exp(-1j * b * np.arange(N) * dx) * self.charfun_q(x - 0.5j) * 1 / (x**2 + 0.25) * weight * dx / 3
        integral_value = np.real(ifft(integrand) * N)

        # Interpolation
        if interp == "linear":
            spline = interp1d(ks, integral_value, kind="linear")
        elif interp == "cubic":
            spline = interp1d(ks, integral_value, kind="cubic")

        log_moneyness = np.log(self.S / self.K)
        call_price = self.S * np.exp(-self.q * self.ttm) - np.sqrt(self.S * self.K) * np.exp(-self.r * self.ttm) / np.pi * spline(log_moneyness)

        if self.option_type == "call":
            return call_price
        elif self.option_type == "put":
            return call_price + self.K * np.exp(-self.r * self.ttm) - self.S * np.exp(-self.q * self.ttm)
        else:
            raise ValueError("Invalid option_type. Choose 'call' or 'put'.")



    '''
    def charfun_q(self,u):
        return np.exp(1j * u * (self.r - self.q - 0.5 * self.sig**2) * self.ttm - 0.5 * u**2 * (self.sig * np.sqrt(self.ttm))**2)

    def fft_lewis_price_q(self, interp = "cubic"):
        """
        SIMPSON'S RULE IMPLEMENTATION
        
        """
        N = 2**16  # FFT more efficient for N power of 2
        B = 6000  # integration limit
        dx = B / N
        x = np.arange(N) * dx  # the final value B is excluded

        weight = np.arange(N)  # Simpson weights
        weight = 3 + (-1) ** (weight + 1)
        weight[0] = 1
        weight[N - 1] = 1

        weight = np.ones(N)
        weight[1:N-1:2] = 4  # Odd indices
        weight[2:N-2:2] = 2  # Even indices

        dk = 2 * np.pi / B
        b = N * dk / 2
        ks = -b + dk * np.arange(N)

        integrand = np.exp(-1j * b * np.arange(N) * dx) * self.charfun_q(x - 0.5j) * 1 / (x**2 + 0.25) * weight * dx / 3
        integral_value = np.real(ifft(integrand) * N)

        S_adj = self.S*np.exp(-self.q * self.ttm)

        if interp == "linear":
            spline_lin = interp1d(ks, integral_value, kind="linear")
            prices = S_adj - np.sqrt(self.S * self.K) * np.exp(-(self.r +self.q)* self.ttm) / np.pi * spline_lin(np.log(S_adj / self.K))
        elif interp == "cubic":
            spline_cub = interp1d(ks, integral_value, kind="cubic")
            prices = S_adj - np.sqrt(self.S * self.K) * np.exp(-(self.r +self.q)* self.ttm) / np.pi * spline_cub(np.log(S_adj / self.K))
        return prices
    
    def plot_charfun(self):
        xs = np.linspace(-10, 10, 1000)
        plt.plot(xs, np.real(self.charfun_q(xs - 0.5j) / (xs**2 + 0.25)))
        plt.show()
    '''

    
    
    def payoff(self, stock_price_range: bool = True, custom_prices: Union[list, None] = None) -> float:
        """Calculates the payoff of the option considering multiple stock prices at maturity. Stock prices are linearly spaced and centered on the strike """

        if custom_prices is not None:
            stock_prices = np.array(custom_prices)
        elif stock_price_range:
            stock_prices = np.linspace(self.K * 0.5, self.K * 1.5, 1000)
        else:
            stock_prices = np.array([self.S])  # Convertimos a array para mantener consistencia

        if self.option_type == 'call':
            payoff = np.maximum(stock_prices - self.K, 0)
        elif self.option_type == 'put':
            payoff = np.maximum(self.K - stock_prices, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        return self.qty * payoff
    

    def info(self) -> None:
        """Prints out the characteristics of the option in a compact format."""

        print(f"Option Type: {self.option_type.capitalize()}")
        print(f"Spot Price (S): {self.S}")
        print(f"Strike Price (K): {self.K}")
        print(f"Maturity Date: {self.T}")
        print(f"Time to Maturity (T): {self.ttm * 365:.2f} days") 
        print(f"Risk-Free Rate (r): {self.r:.2%}")
        print(f"Dividend Yield (q): {self.q:.2%}")
        print(f"Quantity (qty): {self.qty}")

        # Theoretical price and volatility
        theoretical_price = self.analytical_price
        print("\nTheoretical (Black-Scholes):")
        print(f"  - Input volatility (σ): {self.sig:.2%}")
        print(f"  - Price: {theoretical_price:.4f}")

        # Premium paid and implied volatility
        print("\nMarket (Based on Premium):")
        print(f"  - Premium Paid: {self.premium:.4f}")
        print(f"  - Implied Volatility: {self.implied_volatility():.2%}")