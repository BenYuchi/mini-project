import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as ols
import matplotlib.pyplot as plt
import seaborn as sns

def assign_yearcategory(data):
    bins = [1900, 1924.9, 1949.9, 1974.9, 1999.9, 2024]
    labels = ['1900.0-1924.9', '1925.0-1949.9', '1950.0-1974.9', '1975.0-1999.9', '2000.0-2024.9']
    data['YearCategory'] = pd.cut(data['ReleaseYear'], bins=bins, labels=labels, right=False)
    return data

def clean_numactors(data):
    data_clean = data[data['NumActors'] > 0]
    print(f"Cleaned NumActors data distribution:\n{data_clean['NumActors'].value_counts()}")
    return data_clean

def preprocess_numactors(data):
    if 'Actors' in data.columns:
        data['NumActors'] = data['Actors'].apply(lambda x: len(str(x).split('|')))
    else:
        print("The dataset does not contain the 'Actors' column.")
    return data

def balance_sample(data, column, target_sample_size=None):
    try:
        if target_sample_size is None:
            min_sample_size = data[column].value_counts().min()
        else:
            min_sample_size = target_sample_size
        
        if min_sample_size == 0:
            raise ValueError(f"Some categories in {column} have no sample data, balancing is not possible.")

        balanced_data = data.groupby(column).apply(lambda x: x.sample(min_sample_size)).reset_index(drop=True)
        return balanced_data
    except Exception as e:
        print(f"Error occurred while balancing samples: {e}")
        return None

def encode_sentiment(data):
    sentiment_mapping = {
        'Positive': 1,
        'Neutral': 0,
        'Negative': -1
    }
    data['SentimentScore'] = data['Sentiment'].map(sentiment_mapping)
    data['SentimentScore'] = data['SentimentScore'].fillna(0)
    print("\nSentiment column has been converted to numeric (SentimentScore):")
    print(data[['Sentiment', 'SentimentScore']].head())
    return data

def perform_anova(data, independent_var, dependent_var, analysis_type):
    print(f"\nPerforming ANOVA analysis for {analysis_type}...")
    print(data[[independent_var, dependent_var]].head(20))
    print(f"Number of missing values in {independent_var}: {data[independent_var].isna().sum()}")
    print(f"Number of missing values in {dependent_var}: {data[dependent_var].isna().sum()}")
    data_clean = data.dropna(subset=[independent_var, dependent_var])
    if data_clean[independent_var].nunique() > 50:
        print(f"Too many categories in {independent_var} ({data_clean[independent_var].nunique()} categories), sampling will be performed...")
        top_categories = data_clean[independent_var].value_counts().nlargest(50).index
        data_clean = data_clean[data_clean[independent_var].isin(top_categories)]
        print(f"Top 50 most common categories will be retained for analysis.")
    if data_clean.shape[0] < 2:
        print(f"Insufficient data to perform {analysis_type}.")
        return
    formula = f'{dependent_var} ~ C({independent_var})'
    try:
        model = ols.ols(formula, data=data_clean).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print("\nANOVA analysis results:")
        print(anova_table)
        f_value = anova_table['F'].iloc[0]
        p_value = anova_table['PR(>F)'].iloc[0]
        interpret_anova_results(f_value, p_value, independent_var, dependent_var)
        visualize_anova(data_clean, independent_var, dependent_var, analysis_type)
    except Exception as e:
        print(f"ANOVA analysis failed: {e}")

def interpret_anova_results(f_value, p_value, independent_var, dependent_var, threshold=0.05):
    print(f"\nANOVA Results: F-value={f_value}, p-value={p_value}")
    if p_value < threshold:
        print(f"Conclusion: Reject the null hypothesis. {independent_var} has a significant effect on {dependent_var} (p-value < {threshold})")
    else:
        print(f"Conclusion: Cannot reject the null hypothesis. {independent_var} does not have a significant effect on {dependent_var} (p-value >= {threshold})")

def visualize_anova(data, independent_var, dependent_var, analysis_type):
    if dependent_var == "DurationMinutes":
        data_clean = data[data[dependent_var] <= 240]
    else:
        data_clean = data.copy()
    if independent_var == 'YearCategory':
        data_clean = assign_yearcategory(data_clean)
    print(f"Values in {independent_var}: {data_clean[independent_var].unique()}")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=independent_var, y=dependent_var, data=data_clean)
    plt.title(f'Box Plot of {dependent_var} by {independent_var} ({analysis_type})')
    plt.xlabel(independent_var)
    plt.ylabel(dependent_var)
    plt.xticks(rotation=90)
    plt.show()

def director_sentiment_vs_duration(data):
    print("\nPerforming ANOVA analysis for Director vs Sentiment vs Duration...")
    print(data[['Director', 'Sentiment']].head(20))
    print(f"Number of missing values in Director: {data['Director'].isna().sum()}")
    print(f"Number of missing values in Sentiment: {data['Sentiment'].isna().sum()}")
    data_clean = data.dropna(subset=['Director', 'Sentiment', 'DurationMinutes'])
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    data_clean['SentimentScore'] = data_clean['Sentiment'].map(sentiment_map)
    if data_clean['Director'].nunique() > 50:
        print(f"Too many categories in Director ({data_clean['Director'].nunique()} categories), sampling will be performed...")
        top_categories = data_clean['Director'].value_counts().nlargest(50).index
        data_clean = data_clean[data_clean['Director'].isin(top_categories)]
        print(f"Top 50 most common categories will be retained for analysis.")
    if data_clean.shape[0] < 2:
        print("Insufficient data to perform ANOVA for Director vs Sentiment vs Duration.")
        return
    formula = 'DurationMinutes ~ C(Director) + SentimentScore'
    try:
        model = ols.ols(formula, data=data_clean).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print("\nANOVA analysis results:")
        print(anova_table)
        f_value = anova_table['F'].iloc[0]
        p_value = anova_table['PR(>F)'].iloc[0]
        interpret_anova_results(f_value, p_value, 'Director and Sentiment', 'DurationMinutes')
        visualize_anova(data_clean, 'Director', 'DurationMinutes', 'Director vs Sentiment vs Duration')
    except Exception as e:
        print(f"ANOVA analysis failed: {e}")

def writer_vs_duration_analysis(data):
    print("\nPerforming ANOVA analysis for Writer vs Duration...")
    perform_anova(data, 'Writer', 'DurationMinutes', 'Writer vs Duration')

def encode_sentiment(data):
    sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    data['SentimentScore'] = data['Sentiment'].map(sentiment_mapping)
    return data

def year_vs_sentiment_analysis(data):
    print("\nPerforming ANOVA analysis for Year vs Sentiment...")
    data = encode_sentiment(data)
    print(data[['ReleaseYear', 'SentimentScore']].head(20))
    print(f"Number of missing values in ReleaseYear: {data['ReleaseYear'].isna().sum()}")
    print(f"Number of missing values in SentimentScore: {data['SentimentScore'].isna().sum()}")
    data_clean = data.dropna(subset=['ReleaseYear', 'SentimentScore'])
    if data_clean['ReleaseYear'].nunique() > 50:
        print(f"Too many categories in ReleaseYear ({data_clean['ReleaseYear'].nunique()} categories), sampling will be performed...")
        top_years = data_clean['ReleaseYear'].value_counts().nlargest(50).index
        data_clean = data_clean[data_clean['ReleaseYear'].isin(top_years)]
        print(f"Top 50 most common years will be retained for analysis.")
    formula = 'SentimentScore ~ C(ReleaseYear)'
    try:
        model = ols.ols(formula, data=data_clean).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print("\nANOVA analysis results:")
        print(anova_table)
        f_value = anova_table['F'].iloc[0]
        p_value = anova_table['PR(>F)'].iloc[0]
        print(f"\nANOVA Results: F-value={f_value}, p-value={p_value}")
        if p_value < 0.05:
            print(f"Conclusion: Reject the null hypothesis. ReleaseYear has a significant effect on SentimentScore (p-value < 0.05)")
        else:
            print(f"Conclusion: Cannot reject the null hypothesis. ReleaseYear does not have a significant effect on SentimentScore (p-value >= 0.05)")
        visualize_year_vs_sentiment(data_clean)
    except Exception as e:
        print(f"ANOVA analysis failed: {e}")

def visualize_year_vs_sentiment(data):
    plt.figure(figsize=(12, 6))
    sns.barplot(x="ReleaseYear", y="SentimentScore", data=data, estimator=sum, errorbar=None)
    plt.title('Bar Plot of SentimentScore by ReleaseYear (Year vs Sentiment)')
    plt.xlabel('ReleaseYear')
    plt.ylabel('SentimentScore (Sum)')
    plt.xticks(rotation=90)
    plt.show()
