import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

class RegressionPolynomial:
    def __init__(self, 
                 league_name: str, 
                 stats: str, 
                 match_rating: Dict, 
                 range: Tuple[int, int]):
        """
            Transforms the match ratings collected by the MatchRating class using Linear Regression to fit a function. 
            Generates an interval range of values for each match rating based on the fitted model.
        """
        self.league_name = league_name
        self.stats = stats
        self.match_rating = match_rating
        self.range = range
        
    def _transform_match2percentages(self) -> None:
        """
        Converts match ratings into percentage values for outcomes and goal distributions.

        Results are stored in the following attributes as numpy arrays:
        - `self.H_perc`: Home win percentages.
        - `self.D_perc`: Draw percentages.
        - `self.A_perc`: Away win percentages.
        - `self.more_gols_perc`: More Goals percentages.
        - `self.less_gols_perc`: Less Goals percentages.

        If the total for any category is zero, percentages are set to 0.
        """
        self.keys = np.array(list(self.match_rating.keys())) 
        H_perc = []
        D_perc = []
        A_perc = []
        more_gols_perc = []
        less_gols_perc = []

        # Calculate percentages for Home (H), Draw (D), and Away (A)
        for key in self.keys:
            values = self.match_rating[key]
            total_ftr = sum(list(values.values())[:3])
            total_gols = sum(list(values.values())[3:])

            if total_ftr > 0:
                H_perc.append((values['H'] / total_ftr) * 100)
                D_perc.append((values['D'] / total_ftr) * 100)
                A_perc.append((values['A'] / total_ftr) * 100)

            if total_gols > 0:
                more_gols_perc.append((values['+gols'] / total_gols) * 100)
                less_gols_perc.append((values['-gols'] / total_gols) * 100)

            else:
                H_perc.append(0)
                D_perc.append(0)
                A_perc.append(0)
                more_gols_perc.append(0)
                less_gols_perc.append(0)

        self.H_perc = np.array(H_perc)
        self.D_perc = np.array(D_perc)
        self.A_perc = np.array(A_perc)
        
        self.more_gols_perc = np.array(more_gols_perc)
        self.less_gols_perc = np.array(less_gols_perc)
        
    def _remove_outliers(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Removes outliers from the data based on the interquartile range (IQR).
        
        Args:
            X (np.ndarray): Feature data.
            y (np.ndarray): Target data.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Filtered X and y without outliers.
        """
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        irq = q3 - q1
        
        inferior_limit = q1 - 1.5 * irq
        superior_limit = q3 + 1.5 * irq
        
        filter = (y >= inferior_limit) & (y <= superior_limit)
        
        X_without_outliers = X[filter]
        y_without_outliers = y[filter]
        
        return X_without_outliers, y_without_outliers
                
    def _find_best_polynomial(self, 
                              X: np.ndarray, 
                              y: np.ndarray, 
                              max_degree: int = 4, 
                              threshold: float = 0.05) -> Tuple[Optional[LinearRegression], int, float]:
        """
        Finds the best polynomial fit for the given data X and y by testing polynomials 
        up to a specified degree.

        Args:
            X (np.ndarray): Feature data.
            y (np.ndarray): Target data.
            max_degree (int): Maximum polynomial degree to test. Default is 4.
            threshold (float): Minimum improvement in R² score to consider a better fit. Default is 0.05.
        
        Returns:
            Tuple[Optional[LinearRegression], int, float]: Best model, degree, and R² score.
        """
        best_degree = 1
        best_r2 = -np.inf
        best_model: Optional[LinearRegression] = None
        degrees = range(1, max_degree + 1)
        
        # Loop through different degrees of polynomials
        for degree in degrees:
            poly = PolynomialFeatures(degree)
            X_poly = poly.fit_transform(X.reshape(-1, 1))
            model = LinearRegression().fit(X_poly, y)
            y_pred = model.predict(X_poly)
            
            # Calculate score
            r2 = r2_score(y, y_pred)

            # Check if current model has the best score
            if r2 > best_r2 + threshold:
                best_r2 = r2
                best_degree = degree
                best_model = model
            
        return best_model, best_degree, best_r2 

    def _plot_regression(self, 
                         keys: np.ndarray, 
                         match_percentages: np.ndarray, 
                         color: str, 
                         label: str) -> Tuple[Optional[LinearRegression], int, float]:
        """
        Plots the data points and fits a polynomial regression curve to them.
        
        Args:
            keys (np.ndarray): Independent variable values.
            match_percentages (np.ndarray): Dependent variable values (percentages).
            color (str): Color for the plot.
            label (str): Label for the data series.
        
        Returns:
            Tuple[Optional[LinearRegression], int, float]: Best polynomial model, degree, and R² score.
        """
        plt.plot(keys, match_percentages, f'{color}o', label=label)  # Plot original data points
    
        # Find the best polynomial fit
        best_model, best_degree, best_r2 = self._find_best_polynomial(X=keys, y=match_percentages)
        
        # Apply polynomial transformation and predict the fit
        poly = PolynomialFeatures(degree=best_degree)
        keys_poly = poly.fit_transform(keys.reshape(-1, 1))  # Transform the keys for polynomial fit
        
        # Make predictions
        predictions = best_model.predict(keys_poly)
        
        # Plot the polynomial regression curve
        plt.plot(keys, 
                 predictions, 
                 f'{color}-', 
                 label=f'Polynomial Regression {label} (Degree {best_degree}): $R^2 = {best_r2:.2f}$')
        
        plt.title(f'Percentage of {label} - Best Fit $R^2$ = {best_r2:.2f}')
        plt.ylabel('Percentage (%)')
        plt.legend()
        plt.grid(True)

        return best_model, best_degree, best_r2
                
    def graphs(self, show_graphs: bool = False) -> Tuple[Optional[LinearRegression], Optional[LinearRegression], 
                                                         Optional[LinearRegression], Optional[LinearRegression], Optional[LinearRegression]]:        
        """
        Generates graphs for Home, Draw, and Away percentages, as well as More and Less goals.
        Saves the graphs as PNG files and optionally displays them.
        
        Args:
            show_graphs (bool): Whether to display the graphs. Default is False.
        
        Returns:
            Tuple[Optional[LinearRegression]]: Best models for Home, Draw, Away, More Gols, and Less Gols.
        """
        #Ftr
        plt.subplot(3, 1, 1)
        keys, H_perc = self._remove_outliers(X=self.keys, y=self.H_perc)
        best_model_H, best_degree_H, best_r2_H = self._plot_regression(keys=keys, 
                                                                       match_percentages=H_perc, 
                                                                       color='b', 
                                                                       label='H')
        
        plt.subplot(3, 1, 2)
        keys, D_perc = self._remove_outliers(X=self.keys, y=self.D_perc)
        best_model_D, best_degree_D, best_r2_D = self._plot_regression(keys=keys, 
                                                                       match_percentages=D_perc, 
                                                                       color='g', 
                                                                       label='D')

        plt.subplot(3, 1, 3)
        keys, A_perc = self._remove_outliers(X=self.keys, y=self.A_perc)
        best_model_A, best_degree_A, best_r2_A = self._plot_regression(keys=keys, 
                                                                       match_percentages=A_perc, 
                                                                       color='r', 
                                                                       label='A')

        # Save the graph 
        root = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(root)
        plt.savefig(fr'{parent_path}/database/leagues/{ self.league_name }/Graphs/{ self.stats }_ftr.png')
        
        plt.close()
        
        # Gols
        plt.subplot(2, 1, 1)
        keys, More_gols = self._remove_outliers(X=self.keys, y=self.more_gols_perc)
        best_model_Mgols, best_degree_Mgols, best_r2_Mgols = self._plot_regression(keys=keys, 
                                                                                   match_percentages=More_gols, 
                                                                                   color='b', 
                                                                                   label='More Gols')
        
        plt.subplot(2, 1, 2)
        keys, Less_gols = self._remove_outliers(X=self.keys, y=self.less_gols_perc)
        best_model_Lgols, best_degree_Lgols, best_r2_Lgols = self._plot_regression(keys=keys, 
                                                                                   match_percentages=Less_gols, 
                                                                                   color='g', 
                                                                                   label='Less Gols')
        
        plt.savefig(f'{ parent_path}/database/leagues/{ self.league_name }/Graphs/{ self.stats }_gols.png')
        
        if show_graphs:
            plt.tight_layout()
            plt.show()  
        
        plt.close()  
        
        return best_model_H, best_model_D, best_model_A, best_model_Mgols, best_model_Lgols
    
    def __fix_percentages(self, 
                          H: float, 
                          D: float, 
                          A: float, 
                          MG: float, 
                          LG: float) -> Tuple[float, float, float, float, float]:
        """    
        This method ensures that all input values are non-negative and that percentages for match outcomes 
        (Home win, Draw, Away win) sum to 100%. Additionally, it normalizes the goal percentages 
        (More Goals, Less Goals) to sum to 100% if their total is greater than zero.

        Args:
            H (float): Predicted value for the Home team win percentage.
            D (float): Predicted value for the Draw percentage.
            A (float): Predicted value for the Away team win percentage.
            MG (float): Predicted value for the percentage of matches with More Goals.
            LG (float): Predicted value for the percentage of matches with Less Goals.

        Returns:
            H_perc (float): Normalized percentage for Home team wins.
            D_perc (float): Normalized percentage for Draws.
            A_perc (float): Normalized percentage for Away team wins.
            MG_perc (float): Normalized percentage for More Goals.
            LG_perc (float): Normalized percentage for Less Goals.
        """
        H = max(H, 0)
        D = max(D, 0)
        A = max(A, 0)
        
        MG = max(MG, 0)
        LG = max(LG, 0)

        total_ftr = H + D + A
        
        if total_ftr > 0:
            # Normalize percentages
            H_perc = (H / total_ftr) * 100
            D_perc = (D / total_ftr) * 100
            A_perc = (A / total_ftr) * 100

        else:
            # If total is zero, set all to 0%
            H_perc = D_perc = A_perc = 0
            
        total_gols = MG + LG
        if total_gols > 0:
            MG_perc = (MG / total_gols) * 100
            LG_perc = (LG / total_gols) * 100

        else:
            MG_perc = LG_perc = 0
        
        return H_perc, D_perc, A_perc, MG_perc, LG_perc

    def _adjust_match_predictions(self, 
                                  best_model_H: Optional[LinearRegression], 
                                  best_model_D: Optional[LinearRegression], 
                                  best_model_A: Optional[LinearRegression], 
                                  best_model_Mgols: Optional[LinearRegression], 
                                  best_model_Lgols: Optional[LinearRegression], 
                                  x_min: int, 
                                  x_max: int) -> Dict:
        """
        Adjust match prediction (H, D, A, +gols, -gols) for the given range of x values using polynomial regression models.

        Args:
            best_model_H (LinearRegression): Trained model for Home predictions.
            best_model_D (LinearRegression): Trained model for Draw predictions.
            best_model_A (LinearRegression): Trained model for Away predictions.
            best_model_Mgols (LinearRegression): Trained model for More Goals predictions.
            best_model_Lgols (LinearRegression): Trained model for Less Goals predictions.
            x_min (int): Start of the range for predictions.
            x_max (int): End of the range for predictions.

        Returns:
            Dict[int, Dict[str, float]]: A dictionary where each key is an x value and the value is another dictionary containing 
                                        percentages for H, D, A, +gols, and -gols.
        """
        resultados = {}
    
        # Coefficients and intercepts
        coefs_h = best_model_H.coef_
        interc_h = best_model_H.intercept_
        
        coefs_d = best_model_D.coef_
        interc_d = best_model_D.intercept_
        
        coefs_a = best_model_A.coef_
        interc_a = best_model_A.intercept_

        coefs_mg = best_model_Mgols.coef_
        interc_mg = best_model_Mgols.intercept_
        
        coefs_lg = best_model_Lgols.coef_
        interc_lg = best_model_Lgols.intercept_

        # Prediction function
        def f(x, intercepto, coeficientes):
            return intercepto + sum(coeficientes[i] * (x ** i) for i in range(1, len(coeficientes)))

        # For each x value in the specified range
        for x in range(x_min, x_max + 1):
            H_pred = f(x, interc_h, coefs_h)
            D_pred = f(x, interc_d, coefs_d)
            A_pred = f(x, interc_a, coefs_a)
            
            MG_pred = f(x, interc_mg, coefs_mg)
            LG_pred = f(x, interc_lg, coefs_lg)
                
            # Fix percentages
            H_perc, D_perc, A_perc, MG_perc, LG_perc = self.__fix_percentages(H_pred, D_pred, A_pred, MG_pred, LG_pred)
            
            # Save to results dictionary
            resultados[x] = {'H': round(H_perc, 2), 
                             'D': round(D_perc, 2), 
                             'A': round(A_perc, 2), 
                             '+gols': round(MG_perc, 2), 
                             '-gols': round(LG_perc, 2)}
        
        return resultados
    
    def fit(self, show_graphs: bool = False) -> Dict:
        """
            Main method to execute the full process:
            1. Transform match ratings into percentages.
            2. Generate graphs and save them.
            3. Re-adjust match ratings using the best polynomial models.
        """
        self._transform_match2percentages()  
        best_model_H, best_model_D, best_model_A, best_model_Mgols, best_model_Lgols = self.graphs(show_graphs) 
        return self._adjust_match_predictions(best_model_H, 
                                              best_model_D, 
                                              best_model_A, 
                                              best_model_Mgols, 
                                              best_model_Lgols, 
                                              self.range[0], 
                                              self.range[1]) 
