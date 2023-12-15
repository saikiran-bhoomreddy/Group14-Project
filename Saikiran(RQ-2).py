import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('/content/output.csv')  # Replace with your actual file path

# Explore data
print(df.head())

# Handle missing values in the 'prompt' column by filling NaN with an empty string
df['Prompt'].fillna('', inplace=True)

# Preprocess the data
vectorizer = CountVectorizer()
XP = vectorizer.fit_transform(df['Prompt'])
YP = df['WordCount']

X,y=XP[:100],YP[:100]
xtest,ytest=XP[100:],YP[100:]
# Train the model
model = LinearRegression()
model.fit(X, y)

# Example usage with a new prompt
new_prompt = ["World with single gender "]
new_prompt_vectorized = vectorizer.transform(new_prompt)
predicted_word_count = model.predict(new_prompt_vectorized)

print(f"Expected Word Count for New Prompt: {predicted_word_count[0]}")

# Plot actual vs predicted word count for the test set
ypred = []
for x in xtest:
  ypred.append(model.predict(x)[0])

ytest = list(ytest)
import matplotlib.pyplot as plt

# Plotting the black line
plt.plot(ypred, color='black', label='Predicted')

# Plotting the red line
plt.plot(ytest, color='red', label='Test')

# Adding labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Predicted Vs Test')

# Adding a legend
plt.legend()

# Display the plot
plt.show()