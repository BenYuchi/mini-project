import anova_analysis
import regression_analysis
import sentiment_analysis
import pandas as pd

def convert_duration_to_minutes(duration_str):
    if isinstance(duration_str, str):
        time_parts = duration_str.lower().replace('h', '').replace('min', '').strip().split()
        hours = int(time_parts[0]) if len(time_parts) > 0 else 0
        minutes = int(time_parts[1]) if len(time_parts) > 1 else 0
        return hours * 60 + minutes
    return None

def preprocess_data(data):
    if 'Actors' in data.columns:
        # Count the number of actors separated by '|' and assign to `NumActors`
        data['NumActors'] = data['Actors'].apply(lambda x: len(str(x).split('|')) if pd.notna(x) else 0)

    data['DurationMinutes'] = data['Duration'].apply(convert_duration_to_minutes)

    data = anova_analysis.assign_yearcategory(data)

    print("\nConverted 'Duration' column (DurationMinutes):")
    print(data[['Duration', 'DurationMinutes']].head())
    print("\n'YearCategory' column after preprocessing (first 5 rows):")
    print(data[['ReleaseYear', 'YearCategory']].head())
    
    return data

def load_data():
    file_path = input("Please enter the dataset path (CSV format): ")
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    print("\nBasic information of the dataset:")
    print(data.info())
    print("\nDescriptive statistics of the dataset:")
    print(data.describe())
    
    data = preprocess_data(data)
   
    print("\nPreview of the processed dataset:")
    print(data[['Actors', 'NumActors']].head())  # Print the new 'NumActors' column
    return data

def main():
    data = load_data()
    data = preprocess_data(data)  

    while True:
        print("\nPlease choose an analysis type:")
        print("1. ANOVA Analysis (YearCategory vs ReleaseYear)")
        print("2. ANOVA Analysis (Director vs Duration)")
        print("3. ANOVA Analysis (Writer vs Duration)")
        print("4. ANOVA Analysis (NumActors vs Duration)")
        print("5. Regression Analysis (ReleaseYear vs Duration)")
        print("6. Regression Analysis (NumActors vs Duration)")
        print("7. Regression Analysis (Duration vs Sentiment)")
        print("8. Sentiment Analysis")
        print("9. ANOVA Analysis (Director vs Sentiment vs Duration)")
        print("10. ANOVA Analysis (Year vs Sentiment)")
        print("11. ANOVA Analysis (Writer vs Duration)")
        print("12. Exit Program")

        choice = input("Please enter your choice (1-12): ")

        if choice == '1':
            anova_analysis.perform_anova(data, 'YearCategory', 'ReleaseYear', 'YearCategory vs ReleaseYear')
        elif choice == '2':
            anova_analysis.perform_anova(data, 'Director', 'DurationMinutes', 'Director vs Duration')
        elif choice == '3':
            anova_analysis.perform_anova(data, 'Writer', 'DurationMinutes', 'Writer vs Duration')
        elif choice == '4':  
            anova_analysis.numactors_vs_duration_analysis(data) 
        elif choice == '5':
            regression_analysis.perform_regression(data, 'ReleaseYear', 'DurationMinutes')
        elif choice == '6':
            regression_analysis.perform_actors_duration_regression(data)
        elif choice == '7':
            regression_analysis.perform_duration_sentiment_regression(data)
        elif choice == '8':
            sentiment_analysis.perform_sentiment_analysis(data)
        elif choice == '9':
            anova_analysis.director_sentiment_vs_duration(data)
        elif choice == '10':
            anova_analysis.year_vs_sentiment_analysis(data)  
        elif choice == '11':
            anova_analysis.writer_vs_duration_analysis(data)
        elif choice == '12':
            print("Exiting program.")
            break
        else:
            print("Invalid choice, please try again.")


if __name__ == "__main__":
    main()
