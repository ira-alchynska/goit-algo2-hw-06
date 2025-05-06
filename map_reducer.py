import requests
import string
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_text(url):
    """
    Fetches text from a specified URL.

    Parameters:
    url (str): URL from which text is retrieved.

    Returns:
    str or None: The fetched text or None if the request fails.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        logging.info(f"Successfully retrieved text from {url}")
        return response.text
    except requests.RequestException as e:
        logging.error(f"Failed to retrieve text: {e}")
        return None


def remove_punctuation(text):
    """
    Removes punctuation from the given text.

    Parameters:
    text (str): Input text.

    Returns:
    str: Text without punctuation.
    """
    return text.translate(str.maketrans("", "", string.punctuation))


# Map Function
def map_function(word):
    """
    Maps words to a tuple (word, 1) for counting.

    Parameters:
    word (str): Input word.

    Returns:
    tuple: Lowercase word and count of 1.
    """
    return word.lower(), 1


# Shuffle Function
def shuffle_function(mapped_values):
    """
    Shuffles mapped values into groups by word.

    Parameters:
    mapped_values (list): List of mapped word-count tuples.

    Returns:
    iterable: Grouped words and their counts.
    """
    shuffled = defaultdict(list)
    for key, value in mapped_values:
        shuffled[key].append(value)
    return shuffled.items()


# Reduce Function
def reduce_function(item):
    """
    Reduces grouped items to final word counts.

    Parameters:
    item (tuple): Grouped word and list of counts.

    Returns:
    tuple: Word and its total count.
    """
    key, values = item
    return key, sum(values)


# MapReduce implementation
def map_reduce(text):
    """
    Executes the MapReduce algorithm on input text.

    Parameters:
    text (str): Input text to process.

    Returns:
    dict: Dictionary of word counts.
    """
    text = remove_punctuation(text)
    words = text.split()

    with ThreadPoolExecutor() as executor:
        mapped_values = list(executor.map(map_function, words))

    shuffled_values = shuffle_function(mapped_values)

    with ThreadPoolExecutor() as executor:
        reduced_values = dict(executor.map(reduce_function, shuffled_values))

    logging.info("MapReduce computation completed")
    return reduced_values


# Visualization function
def visualize_top_words(word_counts, top_n=10):
    """
    Visualizes the top N most frequent words as a bar chart.

    Parameters:
    word_counts (dict): Dictionary of word counts.
    top_n (int): Number of top frequent words to display.
    """
    sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)[:top_n]
    words, counts = zip(*sorted_words)

    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} most frequent words")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    logging.info(f"Displayed top {top_n} words graphically")


# Main execution
if __name__ == '__main__':
    """
    Main execution block. Fetches text from a URL, performs MapReduce, and visualizes the results.
    """
    url = "https://gutenberg.net.au/ebooks01/0100021.txt"
    text = get_text(url)

    if text:
        word_counts = map_reduce(text)
        visualize_top_words(word_counts, top_n=11)
    else:
        logging.error("Failed to execute MapReduce due to text retrieval failure.")
