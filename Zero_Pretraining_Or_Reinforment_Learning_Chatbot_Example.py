import random
import tkinter as tk

# Corpus is where your text input goes.
corpus = """[Your long corpus text here]"""  # Replace this with your actual text corpus.

# Preprocess the corpus
ngram = {}
corpus = corpus.lower().replace('"','').replace("'",'').replace('\n','').replace(')','').replace('(','').replace('[','').replace(']','').replace('’','').replace("“",'').replace("”",'')

# Build the n-gram model
for sentence in corpus.split('.'):
    words = sentence.split(' ')
    for i in range(2, len(words)):
        word_pair = (words[i - 2], words[i - 1])
        if '' in word_pair:
            continue
        if word_pair not in ngram:
            ngram[word_pair] = []
        ngram[word_pair].append(words[i])

# Function to generate output banter based on a starting bigram
def generate_banter(starting_pair=None):
    if not ngram:
        return "Error: n-gram model is empty."
    
    if starting_pair and starting_pair in ngram:
        word_pair = starting_pair
    else:
        word_pair = random.choice(list(ngram.keys()))  # Pick a random starting pair if none provided

    output = word_pair[0] + ' ' + word_pair[1] + ' '

    while word_pair in ngram:
        third = random.choice(ngram[word_pair])
        output += third + ' '
        word_pair = (word_pair[1], third)
    
    return output.strip()

# Function to update text in the GUI
def update_text():
    starting_bi = input_text.get()  # Get user input from the text box
    # Split the input text into bigrams
    starting_pair = tuple(starting_bi.split())
    # Generate the banter based on the starting bigram
    new_banter = generate_banter(starting_pair)
    text_widget.config(text="Banter: " + new_banter)  # Update the GUI text
    root.after(5000, update_text)  # Update every 5 seconds

# Initialize Tkinter app
root = tk.Tk()
root.title("Banter Generator")
root.configure(bg="black")

# Create a label for the text widget
text_widget = tk.Label(root, text="Banter: ", font=("Arial", 24), fg="white", bg="black", wraplength=root.winfo_screenwidth(), justify="center")
text_widget.pack(expand=True)

# Create an input box for the starting bigram
input_label = tk.Label(root, text="Input Starting Bigram: ", font=("Arial", 14), fg="white", bg="black")
input_label.pack()
input_text = tk.Entry(root, font=("Arial", 14), fg="black")
input_text.pack()

# Start the timer to update text every 5 seconds
update_text()

# Run the Tkinter main loop
root.mainloop()
