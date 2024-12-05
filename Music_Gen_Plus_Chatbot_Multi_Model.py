 corpus =       
        '''

"""



"""

        '''  
import os
import tkinter as tk
import random
import threading
from queue import Queue
import sounddevice as sd
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import time

# Force CPU usage for Bark
os.environ["DEVICE"] = "cpu"

# Preload Bark models
preload_models()

# Initialize global variables
tts_enabled = True
banter_queue = Queue()
audio_queue = Queue()
generation_lock = threading.Lock()  # Ensure only one audio generation at a time

# Replace this with your text corpus


# Preprocess corpus
ngram = {}
for sentence in corpus.lower().split('.'):
    words = sentence.split()
    for i in range(2, len(words)):
        word_pair = (words[i - 2], words[i - 1])
        if word_pair not in ngram:
            ngram[word_pair] = []
        ngram[word_pair].append(words[i])

# Generate banter based on starting bigram
def generate_banter(starting_pair=None):
    if not ngram:
        return "Error: n-gram model is empty."
    word_pair = starting_pair if starting_pair in ngram else random.choice(list(ngram.keys()))
    output = f"{word_pair[0]} {word_pair[1]} "
    while word_pair in ngram and len(output) < 100:  # Limit output size to avoid long processing times
        third = random.choice(ngram[word_pair])
        output += third + " "
        word_pair = (word_pair[1], third)
    return output.strip()

# Generate audio with Bark in a background thread
def generate_and_save_audio(prompt, audio_queue):
    with generation_lock:  # Ensure only one generation at a time
        try:
            print(f"Generating audio for: {prompt}")
            audio_array = generate_audio(prompt)

            # Save the audio to a unique WAV file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"audio_outputs/audio_output_{timestamp}.wav"
            os.makedirs("audio_outputs", exist_ok=True)
            write_wav(filepath, SAMPLE_RATE, audio_array)

            # Send audio back to the main thread for playback
            audio_queue.put(audio_array)
            print(f"Saved audio to {filepath}")
        except Exception as e:
            print(f"Error generating audio: {e}")

# Play audio in the main thread
def play_audio(audio_array):
    try:
        sd.play(audio_array, samplerate=SAMPLE_RATE)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")

# Banter generation loop in a separate thread
def banter_loop():
    while True:
        starting_bi = input_text_var.get().strip()
        starting_pair = tuple(starting_bi.split()) if starting_bi else None
        new_banter = generate_banter(starting_pair)
        banter_queue.put(new_banter)
        if tts_enabled:
            threading.Thread(target=generate_and_save_audio, args=(f"[Music]♪{new_banter}♪", audio_queue), daemon=True).start()
        time.sleep(6)

# Process banter queue
def process_banter_queue():
    if not banter_queue.empty():
        text_widget.config(text="Banter: " + banter_queue.get())
    root.after(100, process_banter_queue)

# Process audio queue
def process_audio_queue():
    if not audio_queue.empty():
        play_audio(audio_queue.get())
    root.after(100, process_audio_queue)

# Toggle TTS
def toggle_tts():
    global tts_enabled
    tts_enabled = not tts_enabled
    tts_status.config(text=f"TTS: {'Enabled' if tts_enabled else 'Disabled'}")

# Initialize Tkinter GUI
root = tk.Tk()
root.title("Banter Generator")
root.configure(bg="black")

text_widget = tk.Label(root, text="Banter: ", font=("Arial", 24), fg="white", bg="black", wraplength=root.winfo_screenwidth())
text_widget.pack(expand=True)

input_label = tk.Label(root, text="Input Starting Bigram: ", font=("Arial", 14), fg="white", bg="black")
input_label.pack()

input_text_var = tk.StringVar()
input_text = tk.Entry(root, textvariable=input_text_var, font=("Arial", 14), fg="black")
input_text.pack()

tts_status = tk.Label(root, text="TTS: Enabled", font=("Arial", 14), fg="white", bg="black")
tts_status.pack()

toggle_button = tk.Button(root, text="Toggle TTS", font=("Arial", 14), command=toggle_tts, bg="gray", fg="black")
toggle_button.pack()

# Queues for handling audio and banter processing
banter_queue = Queue()
audio_queue = Queue()

# Start banter generation in a background thread
threading.Thread(target=banter_loop, daemon=True).start()

# Start processing queues
process_banter_queue()
process_audio_queue()

# Run Tkinter event loop
root.mainloop()
