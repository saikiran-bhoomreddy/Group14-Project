
import json
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

nltk.download('punkt')


def get_token_count(text):
    # Tokenize the text and return the count of tokens
    tokens = word_tokenize(text)
    return len(tokens)


def total_token_count(conversation):
    # Calculate the total token count for the entire conversation
    return sum(get_token_count(utterance) for utterance in conversation)


# Read conversation from JSON file
def read_conversation_from_json(file_path):
    with open(file_path, 'r') as file:

     return  json.load(file)


# Plot the distribution of token lengths in the conversation
def plot_length_distribution(conversation):
    token_counts = [get_token_count(utterance) for utterance in conversation]

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(conversation) + 1), token_counts, color='skyblue', edgecolor='black')
    plt.title('Token Length Distribution in Conversation')
    plt.xlabel('Utterance Number')
    plt.ylabel('Number of Tokens')
    plt.show()


# Path to your JSON file
json_file_path = '"C:\\Users\\bhomr\\Downloads\\conversation.json"'

# Read the conversation from the JSON file
conversation = read_conversation_from_json(json_file_path)

# Calculate and print the total token count for the entire conversation
total_tokens = total_token_count(conversation)
print(f"Total Token Count for the Conversation: {total_tokens} tokens")

# Plot the distribution of token lengths in the conversation
plot_length_distribution(conversation)
