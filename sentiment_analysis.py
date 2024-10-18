import matplotlib.pyplot as plt

def perform_sentiment_analysis(data):
    print("\nPerforming sentiment analysis...")
    
    if 'Sentiment' in data.columns:
        sentiment_counts = data['Sentiment'].value_counts()
        print("\nSentiment analysis results:")
        print(sentiment_counts)

        # Visualize sentiment distribution
        sentiment_counts.plot(kind='bar', title='Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.show()
    else:
        print("The dataset does not contain a 'Sentiment' column.")
