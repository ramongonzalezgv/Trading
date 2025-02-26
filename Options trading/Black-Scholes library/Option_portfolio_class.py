import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta

class OptionPortfolio:

    def __init__(self, options=None):
        if options is None:
            self.options = []
        else:
            self.options = options

    def add_option(self, option: object) -> None:
        """ Adds an option to the portfolio  """

        self.options.append(option)

    def stock_prices(self) -> np.ndarray:
        """ Returns an array of 1000 stock price values. Stock prices are linearly spaced and centered on the strike  """

        strike_list = []
        if len(self.options) > 0:
            for opt in self.options:
                strike_list.append(opt.K)
        else:
            strike_list = [0,100]

        min_strike = min(strike_list)
        max_strike = max(strike_list)

        min_strike = np.median(strike_list)
        max_strike = np.median(strike_list)
    
        return np.linspace(min_strike*0.75, max_strike*1.25, 1000)
    
    @property
    def total_delta(self) -> float:
        """ Total Delta of the portfolio """
        return sum(option.delta for option in self.options)

    @property
    def total_gamma(self) -> float:
        """ Total Gamma of the portfolio """
        return sum(option.gamma for option in self.options)

    @property
    def total_theta(self) -> float:
        """ Total Theta of the portfolio """
        return sum(option.theta for option in self.options)

    @property
    def total_vega(self) -> float:
        """ Total Vega of the portfolio """
        return sum(option.vega for option in self.options)

    @property
    def total_rho(self) -> float:
        """ Total Rho of the portfolio """
        return sum(option.rho for option in self.options)

    def total_pnl(self) -> float:
        """ Computes the total PnL of the portfolio """

        if len(self.options) > 0:

            total_pnl = np.zeros_like(self.options[0].payoff())

            for option in self.options:
                total_pnl += option.payoff(stock_price_range = True, custom_prices = self.stock_prices()) - option.premium
            return total_pnl
        else:
            raise ValueError("Option portfolio is empty, Can't calculate P&L") 
        
    def info(self) -> None:
        """Prints out information for each option in the portfolio, ordered by maturity, strike, and type."""
        # Convert maturity date to datetime for sorting
        for option in self.options:
            option.maturity_datetime = datetime.strptime(option.T, '%d-%m-%Y')

        # Sort options by maturity, strike, and option type
        sorted_options = sorted(
            self.options,
            key=lambda x: (x.maturity_datetime, x.K, x.option_type)
        )

        # Print info for each option
        for option in sorted_options:
            print("-" * 40)
            option.info()
    
    def plot_pnl(self, colored_areas: bool = True) -> None:
        """ Plots the PnL of the portfolio """


        plt.figure(figsize=(10, 6))
        
        # Plot the P&L line
        plt.plot(self.stock_prices(), self.total_pnl(), label='Total P&L', color='blue')
        
        if colored_areas:

            # Fill areas of profit and loss
            plt.fill_between(self.stock_prices(), self.total_pnl(), where=(self.total_pnl() > 0), color='lightgreen', alpha=0.5, label='Profit')
            plt.fill_between(self.stock_prices(), self.total_pnl(), where=(self.total_pnl() < 0), color='lightcoral', alpha=0.5, label='Loss')
        
        # Mark the zero line
        plt.axhline(0, color='black', lw=1)
        
        # Labeling
        plt.xlabel('Stock Price at Expiration')
        plt.ylabel('Profit and Loss')
        plt.title('Options Portfolio P&L')
        plt.legend(loc='upper right')
        plt.grid(True)


        # Show the plot
        plt.show()
