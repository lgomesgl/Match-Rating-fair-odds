import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

class RegressionPolynomial:
    def __init__(self, league_name, stats, match_rating, range):
        """
            Initialize the class with league name, statistics type,
            match rating dictionary, and range for match ratings.
        """
        self.league_name = league_name
        self.stats = stats
        self.match_rating = match_rating
        self.range = range
        
    def _transform_match2percentages(self):
        """
            Converts the match rating data (home, draw, away) into percentages and stores them
            in self.H_perc, self.D_perc, and self.A_perc arrays.
        """
        self.keys = np.array(list(self.match_rating.keys())) 
        H_perc = []
        D_perc = []
        A_perc = []
        More_gols_perc = []
        Less_gols_perc = []

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
                More_gols_perc.append((values['+gols'] / total_gols) * 100)
                Less_gols_perc.append((values['-gols'] / total_gols) * 100)

            else:
                H_perc.append(0)
                D_perc.append(0)
                A_perc.append(0)
                More_gols_perc.append(0)
                Less_gols_perc.append(0)

        self.H_perc = np.array(H_perc)
        self.D_perc = np.array(D_perc)
        self.A_perc = np.array(A_perc)
        
        self.More_gols_perc = np.array(More_gols_perc)
        self.Less_gols_perc = np.array(Less_gols_perc)
        
    def _remove_outliers(self, X, y):
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        irq = q3 - q1
        
        inferior_limit = q1 - 1.5 * irq
        superior_limit = q3 + 1.5 * irq
        
        filter = (y >= inferior_limit) & (y <= superior_limit)
        
        X_without_outliers = X[filter]
        y_without_outliers = y[filter]
        
        return X_without_outliers, y_without_outliers
                
    def _find_best_polynomial_fit(self, X, y, max_degree=4, threshold=0.05):
        """
            Finds the best polynomial fit for the given data X and y by testing polynomials 
            up to a specified degree. Returns the best model, the degree, and the R² score.
        """
        best_degree = 1
        best_r2 = -np.inf
        best_model = None
        degrees = range(1, max_degree + 1)
        r2_values = []
        
        # Loop through different degrees of polynomials
        for degree in degrees:
            poly = PolynomialFeatures(degree)
            X_poly = poly.fit_transform(X.reshape(-1, 1))
            model = LinearRegression().fit(X_poly, y)
            y_pred = model.predict(X_poly)
            
            # Calculate score
            r2 = r2_score(y, y_pred)
            r2_values.append(r2)

            # Check if current model has the best score
            if r2 > best_r2 + threshold:
                best_r2 = r2
                best_degree = degree
                best_model = model
            
        return best_model, best_degree, best_r2 

    def _plot_regression(self, keys, match_percentages, color, label):
        """
            Plots the data points and fits a polynomial regression curve to them.
            Displays the best polynomial fit and its R² score.
        """
        plt.plot(keys, match_percentages, f'{color}o', label=label)  # Plot original data points
    
        # Find the best polynomial fit
        best_model, best_degree, best_r2 = self._find_best_polynomial_fit(X=keys, y=match_percentages)
        
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
                
    def graphs(self, show_graphs):        
        """
            Generates three graphs for Home, Draw, and Away percentages and saves them as a PNG file.
            Optionally displays the graphs if show_graphs is set to True.
        """
        #Ftr
        plt.subplot(3, 1, 1)
        keys, H_perc = self._remove_outliers(X=self.keys, y=self.H_perc)
        best_model_H, best_degree_H, best_r2_H = self._plot_regression(keys=keys, match_percentages=H_perc, color='b', label='H')
        
        plt.subplot(3, 1, 2)
        keys, D_perc = self._remove_outliers(X=self.keys, y=self.D_perc)
        best_model_D, best_degree_D, best_r2_D = self._plot_regression(keys=keys, match_percentages=D_perc, color='g', label='D')

        plt.subplot(3, 1, 3)
        keys, A_perc = self._remove_outliers(X=self.keys, y=self.A_perc)
        best_model_A, best_degree_A, best_r2_A = self._plot_regression(keys=keys, match_percentages=A_perc, color='r', label='A')
        
        # Save the graph 
        root = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(root)
        plt.savefig(fr'{parent_path}/Database/{ self.league_name }/Graphs/{ self.stats }_ftr.png')
        
        plt.close()
        
        # Gols
        plt.subplot(2, 1, 1)
        keys, More_gols = self._remove_outliers(X=self.keys, y=self.More_gols_perc)
        best_model_Mgols, best_degree_Mgols, best_r2_Mgols = self._plot_regression(keys=keys, match_percentages=More_gols, color='b', label='More Gols')
        
        plt.subplot(2, 1, 2)
        keys, Less_gols = self._remove_outliers(X=self.keys, y=self.Less_gols_perc)
        best_model_Lgols, best_degree_Lgols, best_r2_Lgols = self._plot_regression(keys=keys, match_percentages=Less_gols, color='g', label='Less Gols')
        
        plt.savefig(f'{ parent_path}/Database/{ self.league_name }/Graphs/{ self.stats }_gols.png')
        
        if show_graphs:
            plt.tight_layout()
            plt.show()  
        
        plt.close()  
        
        return best_model_H, best_model_D, best_model_A, best_model_Mgols, best_model_Lgols
    
    def _fix_percentages(self, H, D, A, MG, LG):
        """
            Helper method to adjust the percentages of H, D, and A to be non-negative and normalized to 100%.
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

    def reajust_match_ratings(self, 
                              best_model_H, 
                              best_model_D, 
                              best_model_A, 
                              best_model_Mgols, 
                              best_model_Lgols, 
                              x_min, 
                              x_max):
        """
            Re-adjusts match ratings (H, D, A percentages) for the given range using the best polynomial
            models obtained. Returns a dictionary of results for each x range values [x_min, x_max].
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
                
            # Adjust percentages
            H_perc, D_perc, A_perc, MG_perc, LG_perc = self._fix_percentages(H_pred, D_pred, A_pred, MG_pred, LG_pred)
            
            # Save to results dictionary
            resultados[x] = {'H': round(H_perc, 2), 'D': round(D_perc, 2), 'A': round(A_perc, 2), '+gols': round(MG_perc, 2), '-gols': round(LG_perc, 2)}
        
        return resultados
    
    def fit(self, show_graphs=False):
        """
            Main method to execute the full process:
            1. Transform match ratings into percentages.
            2. Generate graphs and save them.
            3. Re-adjust match ratings using the best polynomial models.
        """
        self._transform_match2percentages()  # Convert match ratings to percentages
        best_model_H, best_model_D, best_model_A, best_model_Mgols, best_model_Lgols = self.graphs(show_graphs)  # Generate graphs and return best models
        return self.reajust_match_ratings(best_model_H, best_model_D, best_model_A, best_model_Mgols, best_model_Lgols, self.range[0], self.range[1])  # Re-adjust ratings
