import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def clean_data_for_regression(data, dependent_var, independent_var):
    data_clean = data.dropna(subset=[dependent_var, independent_var])
    data_clean = data_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=[dependent_var, independent_var])
    return data_clean

def visualize_regression(data, independent_var, dependent_var):
    data_clean = data[data[dependent_var] < 500]

    plt.figure(figsize=(10, 6))
    
    plt.scatter(data_clean[independent_var], data_clean[dependent_var], color='blue', label='Data Points', alpha=0.5)
    
    X = sm.add_constant(data_clean[[independent_var]])
    model = sm.OLS(data_clean[dependent_var], X).fit()
    plt.plot(data_clean[independent_var], model.predict(X), color='red', label='Regression Line')
    
    plt.title(f'Regression: {dependent_var} vs {independent_var}', fontsize=14)
    plt.xlabel(independent_var, fontsize=12)
    plt.ylabel(dependent_var, fontsize=12)
    
    plt.legend(loc='best', fontsize=10)
    plt.show()

def perform_regression(data, independent_var, dependent_var):
    print(f"\nPerforming regression analysis ({independent_var} vs {dependent_var})...")
    data_clean = clean_data_for_regression(data, dependent_var, independent_var)

    if not data_clean.empty:
        try:
            X = data_clean[[independent_var]]
            Y = data_clean[dependent_var]
            X = sm.add_constant(X)
            model = sm.OLS(Y, X).fit()
            print("\nRegression analysis results:")
            print(model.summary())

            r_squared = model.rsquared
            p_value = model.f_pvalue
            if p_value < 0.05:
                print(f"\nInterpretation: {independent_var} has a significant impact on {dependent_var}, R-squared = {r_squared:.4f}, p-value = {p_value:.4f} (significant).")
            else:
                print(f"\nInterpretation: {independent_var} does not have a significant impact on {dependent_var}, R-squared = {r_squared:.4f}, p-value = {p_value:.4f} (not significant).")
            
            visualize_regression(data_clean, independent_var, dependent_var)

        except Exception as e:
            print(f"Regression analysis failed: {e}")
    else:
        print(f"The dataset does not contain valid {independent_var} or {dependent_var} columns.")

def perform_actors_duration_regression(data):
    print("\nPerforming regression analysis for actors vs movie duration...")
    data_clean = clean_data_for_regression(data, 'DurationMinutes', 'Actors')
    
    if not data_clean.empty:
        try:
            data_clean['NumActors'] = data_clean['Actors'].apply(lambda x: len(x.split('|')) if isinstance(x, str) else 0)
            X = data_clean[['NumActors']]
            Y = data_clean['DurationMinutes']
            X = sm.add_constant(X)
            model = sm.OLS(Y, X).fit()
            print("\nRegression analysis results:")
            print(model.summary())

            r_squared = model.rsquared
            p_value = model.f_pvalue
            if p_value < 0.05:
                print(f"\nInterpretation: The number of actors has a significant impact on movie duration, R-squared = {r_squared:.4f}, p-value = {p_value:.4f} (significant).")
            else:
                print(f"\nInterpretation: The number of actors does not have a significant impact on movie duration, R-squared = {r_squared:.4f}, p-value = {p_value:.4f} (not significant).")

            visualize_regression(data_clean, 'NumActors', 'DurationMinutes')

        except Exception as e:
            print(f"Regression analysis failed: {e}")
    else:
        print("The dataset does not contain valid Actors or DurationMinutes columns.")

def perform_duration_sentiment_regression(data):
    print("\nPerforming regression analysis for movie duration vs sentiment...")
    data_clean = clean_data_for_regression(data, 'DurationMinutes', 'Sentiment')
    
    if not data_clean.empty:
        try:
            sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
            data_clean['SentimentScore'] = data_clean['Sentiment'].map(sentiment_map)

            X = data_clean[['DurationMinutes']]
            Y = data_clean['SentimentScore']
            X = sm.add_constant(X)
            model = sm.OLS(Y, X).fit()
            print("\nRegression analysis results:")
            print(model.summary())

            r_squared = model.rsquared
            p_value = model.f_pvalue
            if p_value < 0.05:
                print(f"\nInterpretation: Movie duration has a significant impact on sentiment, R-squared = {r_squared:.4f}, p-value = {p_value:.4f} (significant).")
            else:
                print(f"\nInterpretation: Movie duration does not have a significant impact on sentiment, R-squared = {r_squared:.4f}, p-value = {p_value:.4f} (not significant).")

            visualize_regression(data_clean, 'DurationMinutes', 'SentimentScore')

        except Exception as e:
            print(f"Regression analysis failed: {e}")
    else:
        print("The dataset does not contain valid DurationMinutes or Sentiment columns.")
