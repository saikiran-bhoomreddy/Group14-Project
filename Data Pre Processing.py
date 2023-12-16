import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

nltk.download('punkt')


def get_token_count(text):
    # Tokenize the text and return the count of tokens
    tokens = word_tokenize(text)
    return len(tokens)

def plot_length_distribution(conversation):
    # Plot the distribution of token lengths in the conversation
    token_counts = [get_token_count(utterance) for utterance in conversation]

    plt.figure(figsize=(10, 6))
    plt.hist(token_counts, bins=20, color='skyblue', edgecolor='black')
    plt.title('Token Length Distribution in Conversation')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.show()


# Example conversation
initial_prompt = "Can you provide a code for the following context?"
user_follow_up_1 = "How does the code handle edge cases?"
model_response_1 = "The code handles edge cases by implementing conditional statements."
user_follow_up_2 = "Can you give an example of a specific edge case?"

# Combine all parts of the conversation
conversation = [initial_prompt, user_follow_up_1, model_response_1, user_follow_up_2]

# Calculate and print the token counts for each part of the conversation
for i, utterance in enumerate(conversation):
    token_count = get_token_count(utterance)
    print(f"Utterance {i + 1}: {token_count} tokens")

# Plot the distribution of token lengths in the conversation
plot_length_distribution(conversation)
