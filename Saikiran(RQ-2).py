import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("C:\\Users\\bhomr\\Desktop\\Software Engineering\\output.csv")  # Replace with your actual file path

# Explore data
print(df.head())

# Handle missing values in the 'prompt' column by filling NaN with an empty string
df['Prompt'].fillna('', inplace=True)

# Preprocess the data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Prompt'])
y = df['WordCount']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Example usage
new_prompt = ["give me summary about heart functioning ", np.nan]  # Include NaN values in new prompts
new_prompt_cleaned = ['' if pd.isna(prompt) else prompt for prompt in new_prompt]
new_prompt_vectorized = vectorizer.transform(new_prompt_cleaned)
predicted_word_count = model.predict(new_prompt_vectorized)

print(f"Predicted Word Count: {predicted_word_count}")
