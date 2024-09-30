import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

class RegressionPolynomial:
    def __init__(self, league_name, stats, match_rating, range):
        """
        Constructor method to initialize the class with league name, statistics type,
        match rating dictionary, and range for match ratings.
        """
        self.league_name = league_name
        self.stats = stats
        self.match_rating = match_rating
        self.range = range
        
    def transform_match2percentages(self):
        """
        Converts the match rating data (home, draw, away) into percentages and stores them
        in self.H_perc, self.D_perc, and self.A_perc arrays.
        """
        self.keys = np.array(list(self.match_rating.keys()))  # Convert keys to NumPy array for regression
        H_perc = []
        D_perc = []
        A_perc = []

        # Calculate percentages for Home (H), Draw (D), and Away (A)
        for key in self.keys:
            values = self.match_rating[key]
            total = sum(values.values())  # Sum of H, D, A
            if total > 0:
                H_perc.append((values['H'] / total) * 100)
                D_perc.append((values['D'] / total) * 100)
                A_perc.append((values['A'] / total) * 100)
            else:
                H_perc.append(0)
                D_perc.append(0)
                A_perc.append(0)

        # Convert lists to NumPy arrays
        self.H_perc = np.array(H_perc)
        self.D_perc = np.array(D_perc)
        self.A_perc = np.array(A_perc)
        
    def find_best_polynomial_fit(self, X, y, max_degree=4, threshold=0.05):
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
            
            # Calculate R² score
            r2 = r2_score(y, y_pred)
            r2_values.append(r2)

            # Check if current model has the best R² score
            if r2 > best_r2 + threshold:
                best_r2 = r2
                best_degree = degree
                best_model = model
            
        return best_model, best_degree, best_r2 

    def plot_regression(self, keys, match_percentages, color, label):
        """
        Plots the data points and fits a polynomial regression curve to them.
        Displays the best polynomial fit and its R² score.
        """
        plt.plot(keys, match_percentages, f'{color}o', label=label)  # Plot original data points
    
        # Find the best polynomial fit
        best_model, best_degree, best_r2 = self.find_best_polynomial_fit(X=keys, y=match_percentages)
        
        # Apply polynomial transformation and predict the fit
        poly = PolynomialFeatures(degree=best_degree)
        keys_poly = poly.fit_transform(keys.reshape(-1, 1))  # Transform the keys for polynomial fit
        
        # Make predictions
        predictions = best_model.predict(keys_poly)
        
        # Plot the polynomial regression curve
        plt.plot(keys, predictions, f'{color}-', label=f'Polynomial Regression {label} (Degree {best_degree}): $R^2 = {best_r2:.2f}$')
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
        plt.subplot(3, 1, 1)
        best_model_H, best_degree_H, best_r2_H = self.plot_regression(keys=self.keys, match_percentages=self.H_perc, color='b', label='H')
        
        plt.subplot(3, 1, 2)
        best_model_D, best_degree_D, best_r2_D = self.plot_regression(keys=self.keys, match_percentages=self.D_perc, color='g', label='D')

        plt.subplot(3, 1, 3)
        best_model_A, best_degree_A, best_r2_A = self.plot_regression(keys=self.keys, match_percentages=self.A_perc, color='r', label='A')
        
        # Save the graph as a PNG file
        plt.savefig(fr'D:\LUCAS\Match Rating\Database\{ self.league_name }\Graphs\{ self.stats }.png')
        
        if show_graphs:
            plt.tight_layout()
            plt.show()  # Show the graphs if requested
        
        plt.close()  # Close the figure to free up memory
        
        return best_model_H, best_model_D, best_model_A
    
    def __fix_percentages(self, H, D, A):
        """
        Helper method to adjust the percentages of H, D, and A to be non-negative and normalized to 100%.
        """
        H = max(H, 0)
        D = max(D, 0)
        A = max(A, 0)
        
        # Calculate the total sum
        total = H + D + A
        
        if total > 0:
            # Normalize percentages
            H_perc = (H / total) * 100
            D_perc = (D / total) * 100
            A_perc = (A / total) * 100
        else:
            # If total is zero, set all to 0%
            H_perc = D_perc = A_perc = 0
        
        return H_perc, D_perc, A_perc

    def reajust_match_ratings(self, best_model_H, best_model_D, best_model_A, x_min, x_max):
        """
        Re-adjusts match ratings (H, D, A percentages) for the given range using the best polynomial
        models obtained. Returns a dictionary of results for each x value.
        """
        resultados = {}
    
        # Coefficients and intercepts
        coefs_h = best_model_H.coef_
        interc_h = best_model_H.intercept_
        
        coefs_d = best_model_D.coef_
        interc_d = best_model_D.intercept_
        
        coefs_a = best_model_A.coef_
        interc_a = best_model_A.intercept_

        # Prediction function
        def f(x, intercepto, coeficientes):
            return intercepto + sum(coeficientes[i] * (x ** i) for i in range(1, len(coeficientes)))

        # For each x value in the specified range
        for x in range(x_min, x_max + 1):
            H_pred = f(x, interc_h, coefs_h)
            D_pred = f(x, interc_d, coefs_d)
            A_pred = f(x, interc_a, coefs_a)
            
            # Adjust percentages
            H_perc, D_perc, A_perc = self.__fix_percentages(H_pred, D_pred, A_pred)
            
            # Save to results dictionary
            resultados[x] = {'H': round(H_perc, 2), 'D': round(D_perc, 2), 'A': round(A_perc, 2)}
        
        return resultados
    
    def fit(self, show_graphs=False):
        """
        Main method to execute the full process:
        1. Transform match ratings into percentages.
        2. Generate graphs and save them.
        3. Re-adjust match ratings using the best polynomial models.
        """
        self.transform_match2percentages()  # Convert match ratings to percentages
        best_model_H, best_model_D, best_model_A = self.graphs(show_graphs)  # Generate graphs and return best models
        return self.reajust_match_ratings(best_model_H, best_model_D, best_model_A, self.range[0], self.range[1])  # Re-adjust ratings
