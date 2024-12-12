import os
import tkinter as tk
import random
import threading
from queue import Queue
import sounddevice as sd
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from collections import Counter
import math
import re
import time
from tkinter import ttk, filedialog
from datetime import datetime
import tkinter.font as tkFont

# Force CPU usage for Bark
os.environ["DEVICE"] = "cpu"

# Preload Bark models
preload_models()

class CorpusManager:
    def __init__(self):
        self.imported_files = {}  # {filename: {"content": content, "weight": 1.0}}
        self.current_corpus = ""
        self.logging_enabled = False
        self.log_file = None
        
    def import_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                filename = os.path.basename(filepath)
                self.imported_files[filename] = {"content": content, "weight": 1.0}
                return True
        except Exception as e:
            print(f"Error importing file: {e}")
            return False
    
    def remove_file(self, filename):
        if filename in self.imported_files:
            del self.imported_files[filename]
            return True
        return False
    
    def get_word_count(self, filename):
        if filename in self.imported_files:
            return len(self.imported_files[filename]["content"].split())
        return 0
    
    def rebuild_corpus(self, selected_files):
        weighted_contents = []
        for filename in selected_files:
            if filename in self.imported_files:
                content = self.imported_files[filename]["content"]
                weight = self.imported_files[filename]["weight"]
                # Repeat content based on weight
                weighted_contents.extend([content] * int(weight))
        self.current_corpus = " ".join(weighted_contents)
        return self.current_corpus
    
    def set_weight(self, filename, weight):
        if filename in self.imported_files:
            self.imported_files[filename]["weight"] = max(0.1, float(weight))
    
    def get_weight(self, filename):
        if filename in self.imported_files:
            return self.imported_files[filename]["weight"]
        return 1.0
    
    def start_logging(self):
        if not self.logging_enabled:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = open(f"banter_log_{timestamp}.txt", "w", encoding="utf-8")
            self.logging_enabled = True
    
    def stop_logging(self):
        if self.logging_enabled and self.log_file:
            self.log_file.close()
            self.log_file = None
            self.logging_enabled = False
    
    def log_banter(self, banter):
        if self.logging_enabled and self.log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_file.write(f"[{timestamp}] {banter}\n")
            self.log_file.flush()

corpus_manager = CorpusManager()

# Initialize empty n-gram dictionary
ngram = {}

def build_ngram_model(corpus_text):
    global ngram
    ngram = {}
    print("Building n-gram model...")
    for sentence in corpus_text.split('.'):
        words = sentence.split()
        for i in range(1, len(words)):
            word_pair = (words[i - 2], words[i - 1]) if i >= 2 else None
            if word_pair and '' not in word_pair:
                if word_pair not in ngram:
                    ngram[word_pair] = []
                ngram[word_pair].append(words[i])
    print("N-gram model built successfully")

# Initialize global variables
tts_enabled = True
banter_queue = Queue()
audio_queue = Queue()
generation_lock = threading.Lock()

def calculate_entropy(distribution):
    total_count = sum(distribution.values())
    if total_count == 0:
        return 0
    entropy = -sum((count / total_count) * math.log2(count / total_count) for count in distribution.values() if count > 0)
    return entropy

def corpus_entropy():
    try:
        bigram_distribution = Counter(ngram.keys())
        output_distribution = Counter(word for pairs in ngram.values() for word in pairs)
        words = corpus_manager.current_corpus.split()
        return {
            "Bigram Entropy": calculate_entropy(bigram_distribution),
            "Corpus Entropy": calculate_entropy(Counter(words)),
            "Output Entropy": calculate_entropy(output_distribution),
        }
    except Exception as e:
        print(f"Error calculating entropy: {e}")
        return {"Bigram Entropy": 0, "Corpus Entropy": 0, "Output Entropy": 0}

# Static corpus entropy calculations
static_entropy_values = None
generated_outputs = []

def update_dynamic_entropy(new_banter):
    try:
        global static_entropy_values, generated_outputs
        generated_outputs.append(new_banter)
        combined_text = corpus_manager.current_corpus + " " + " ".join(generated_outputs)
        combined_words = combined_text.split()
        combined_bigram_distribution = Counter(
            (combined_words[i], combined_words[i + 1]) for i in range(len(combined_words) - 1)
        )
        dynamic_bigram_entropy = calculate_entropy(combined_bigram_distribution)
        dynamic_word_count = Counter(" ".join(generated_outputs).split())
        dynamic_output_entropy = calculate_entropy(dynamic_word_count)

        if static_entropy_values is None:
            static_entropy_values = corpus_entropy()

        root.after(0, lambda: display_entropy(static_values=static_entropy_values, 
                                           dynamic_output_entropy=dynamic_output_entropy, 
                                           dynamic_bigram_entropy=dynamic_bigram_entropy))
    except Exception as e:
        print(f"Error updating entropy: {e}")

def display_entropy(static_values, dynamic_output_entropy, dynamic_bigram_entropy=None):
    try:
        entropy_text = "\n".join([
            f"Bigram Entropy (Static): {static_values['Bigram Entropy']:.4f}",
            f"Corpus Entropy (Static): {static_values['Corpus Entropy']:.4f}",
            f"Bigram Entropy (Dynamic): {dynamic_bigram_entropy:.4f}" if dynamic_bigram_entropy else "",
            f"Output Entropy (Dynamic): {dynamic_output_entropy:.4f}"
        ])
        entropy_label.config(text=f"Entropy:\n{entropy_text}")
    except Exception as e:
        print(f"Error displaying entropy: {e}")

def generate_banter(starting_pair=None, greedy=False):
    if not ngram:
        return "Error: n-gram model is empty."
    word_pair = starting_pair if starting_pair in ngram else random.choice(list(ngram.keys()))
    output = f"{word_pair[0]} {word_pair[1]} "
    while word_pair in ngram and len(output.split()) < 4096:
        if not ngram[word_pair]:
            break
        if greedy:
            next_word = max(set(ngram[word_pair]), key=ngram[word_pair].count)
        else:
            next_word = random.choice(ngram[word_pair])
        output += next_word + " "
        word_pair = (word_pair[1], next_word)
    return output.strip()

def generate_and_save_audio(prompt):
    with generation_lock:
        try:
            root.after(0, lambda: progress_var.set(0))
            root.after(0, lambda: progress_label.config(text=f"Generating Audio: {prompt[:50]}..."))

            audio_array = generate_audio(prompt)

            for i in range(1, 101):
                time.sleep(0.05)
                root.after(0, lambda i=i: progress_var.set(i))
                root.after(0, lambda i=i: progress_label.config(text=f"Progress: {i}% Complete"))

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"audio_outputs/audio_output_{timestamp}.wav"
            os.makedirs("audio_outputs", exist_ok=True)
            write_wav(filepath, SAMPLE_RATE, audio_array)

            text_filepath = filepath.replace(".wav", ".txt")
            with open(text_filepath, "w", encoding="utf-8") as f:
                f.write(prompt)

            root.after(0, lambda: progress_label.config(text=f"Saved audio: {filepath}\nSaved prompt: {text_filepath}"))
            display_prompt = prompt if len(prompt) < 32 else prompt[:100] + "..."
            root.after(0, lambda: update_text_display(f"Banter: {display_prompt}"))

            sd.play(audio_array, SAMPLE_RATE)
            sd.wait()

        except Exception as e:
            root.after(0, lambda: progress_label.config(text=f"Error generating audio: {e}"))
            print(f"Error generating audio: {e}")

def update_text_display(text):
    try:
        # Get current font settings
        font_family = font_var.get()
        font_size = int(float(font_size_var.get()))  # Convert to float first, then int
        
        text_widget.config(state='normal')
        text_widget.delete(1.0, tk.END)
        # Apply the current font settings to the text
        text_widget.configure(font=(font_family, font_size))
        text_widget.insert(tk.END, text, "center")
        text_widget.config(state='disabled')
    except Exception as e:
        print(f"Error updating text display: {e}")

def safe_color_change(widget, **kwargs):
    try:
        root.after(0, lambda: widget.configure(**kwargs))
    except Exception as e:
        print(f"Error changing widget color: {e}")

def check_and_set_color(banter):
    try:
        match = re.search(r'#?[0-9a-fA-F]{6}', banter)
        if match:
            hex_color = match.group(0)
            if not hex_color.startswith("#"):
                hex_color = f"#{hex_color}"
            
            # Schedule color updates using after() method
            root.after(0, lambda: safe_color_change(root, bg=hex_color))
            root.after(0, lambda: safe_color_change(text_frame, bg=hex_color))
            root.after(0, lambda: safe_color_change(text_widget, bg=hex_color))
            root.after(0, lambda: update_text_color(hex_color))
            print(f"Background color changed to {hex_color}")
    except Exception as e:
        print(f"Error in color change: {e}")

def update_text_color(bg_color):
    try:
        bg_color = bg_color.lstrip("#")
        r, g, b = int(bg_color[:2], 16), int(bg_color[2:4], 16), int(bg_color[4:], 16)
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        text_color = "#809980" if luminance > 0.5 else "white"
        root.after(0, lambda: text_widget.config(fg=text_color))
    except Exception as e:
        print(f"Error updating text color: {e}")

def text_loop():
    global static_entropy_values
    if static_entropy_values is None:
        static_entropy_values = corpus_entropy()
        display_entropy(static_entropy_values, 0)

    while True:
        try:
            if corpus_manager.current_corpus:  # Only generate if we have a corpus
                starting_bi = input_text_var.get().strip()
                starting_pair = tuple(starting_bi.split()) if len(starting_bi.split()) == 2 else None
                new_banter = generate_banter(starting_pair, greedy=greedy_mode.get())
                banter_queue.put(new_banter)
                corpus_manager.log_banter(new_banter)  # Log the banter if logging is enabled
                root.after(0, lambda: check_and_set_color(new_banter))
                update_dynamic_entropy(new_banter)
            time.sleep(sleep_duration.get())  # Use the dynamic sleep duration
        except Exception as e:
            print(f"Error in text loop: {e}")
            time.sleep(sleep_duration.get())  # Use the dynamic sleep duration

def audio_loop():
    while tts_enabled:
        try:
            if not banter_queue.empty():
                banter = banter_queue.get()
                audio_done = threading.Event()
                
                def generate_audio_thread():
                    try:
                        generate_and_save_audio(f"[Music]♪[{banter}]♪")
                    finally:
                        audio_done.set()
                
                audio_thread = threading.Thread(target=generate_audio_thread, daemon=True)
                audio_thread.start()
                audio_done.wait()
            else:
                time.sleep(1)
        except Exception as e:
            print(f"Error in audio loop: {e}")
            time.sleep(1)

def process_banter_queue():
    try:
        if not banter_queue.empty():
            banter = banter_queue.get()
            root.after(0, lambda: update_text_display("Banter: " + banter))
    except Exception as e:
        print(f"Error processing banter queue: {e}")
    finally:
        root.after(100, process_banter_queue)

def toggle_tts():
    global tts_enabled
    tts_enabled = not tts_enabled
    root.after(0, lambda: tts_status.config(text=f"TTS: {'Enabled' if tts_enabled else 'Disabled'}"))

def toggle_logging():
    if corpus_manager.logging_enabled:
        corpus_manager.stop_logging()
        logging_button.config(bg="red")
    else:
        corpus_manager.start_logging()
        logging_button.config(bg="green")

weight_entries = {}  # Store weight entry widgets

def add_weight_control(filename):
    weight_frame = tk.Frame(weights_frame, bg="black")
    weight_frame.pack(fill="x", pady=1)
    
    name_label = tk.Label(weight_frame, text=filename, fg="white", bg="black", width=20, anchor="w")
    name_label.pack(side=tk.LEFT, padx=2)
    
    weight_var = tk.StringVar(value="1.0")
    weight_entry = tk.Entry(weight_frame, textvariable=weight_var, width=5)
    weight_entry.pack(side=tk.LEFT, padx=2)
    
    update_btn = tk.Button(weight_frame, text="Update", command=lambda f=filename: update_weight(f))
    update_btn.pack(side=tk.LEFT, padx=2)
    
    weight_entries[filename] = weight_var

def clear_weight_controls():
    for widget in weights_frame.winfo_children():
        widget.destroy()
    weight_entries.clear()

def update_weight(filename):
    try:
        weight = float(weight_entries[filename].get())
        corpus_manager.set_weight(filename, weight)
        update_word_counts()  # Update display to show new weights
    except ValueError:
        print(f"Invalid weight value for {filename}")

def import_corpus():
    filepaths = filedialog.askopenfilenames(
        title="Select Text Files",
        filetypes=[("Text files", "*.txt")]
    )
    
    for filepath in filepaths:
        if corpus_manager.import_file(filepath):
            filename = os.path.basename(filepath)
            corpus_files_listbox.insert(tk.END, filename)
            add_weight_control(filename)  # Add weight control for new file
    
    update_word_counts()

def remove_selected_files():
    selected_indices = corpus_files_listbox.curselection()
    if not selected_indices:
        return
    
    # Remove files in reverse order to maintain correct indices
    for index in sorted(selected_indices, reverse=True):
        filename = corpus_files_listbox.get(index)
        corpus_manager.remove_file(filename)
        corpus_files_listbox.delete(index)
    
    # Rebuild weight controls
    clear_weight_controls()
    for i in range(corpus_files_listbox.size()):
        add_weight_control(corpus_files_listbox.get(i))
    
    update_word_counts()

def update_word_counts():
    word_counts_text = "Word Counts:\n"
    total_words = 0
    for filename in corpus_manager.imported_files:
        count = corpus_manager.get_word_count(filename)
        weight = corpus_manager.get_weight(filename)
        word_counts_text += f"{filename}: {count} words (weight: {weight:.1f})\n"
        total_words += count * int(weight)
    word_counts_text += f"\nTotal Weighted Words: {total_words}"
    word_counts_label.config(text=word_counts_text)

def rebuild_corpus():
    selected_indices = corpus_files_listbox.curselection()
    if not selected_indices:
        status_label.config(text="Please select files to build corpus")
        return
    
    selected_files = [corpus_files_listbox.get(i) for i in selected_indices]
    corpus_manager.rebuild_corpus(selected_files)
    build_ngram_model(corpus_manager.current_corpus)
    
    # Reset entropy calculations
    global static_entropy_values, generated_outputs
    static_entropy_values = None
    generated_outputs = []
    
    files_str = ", ".join(selected_files)
    status_label.config(text=f"Corpus rebuilt using: {files_str}")

def update_font():
    try:
        # Get current text content
        text_widget.config(state='normal')
        current_text = text_widget.get(1.0, tk.END).strip()
        text_widget.config(state='disabled')
        
        # Update the text with new font settings
        if current_text:
            update_text_display(current_text)
    except Exception as e:
        print(f"Error updating font: {e}")

# Initialize Tkinter GUI
root = tk.Tk()
root.title("Banter Generator")
root.configure(bg="black")

# Initialize sleep duration after root window
sleep_duration = tk.DoubleVar(value=3.0)  # Default 3 seconds

# Font controls frame
font_frame = tk.Frame(root, bg="black")
font_frame.pack(fill="x", padx=5, pady=5)

# Font family selection
available_fonts = list(tkFont.families())
available_fonts.sort()
font_var = tk.StringVar(value="Arial")
font_label = tk.Label(font_frame, text="Font:", fg="white", bg="black")
font_label.pack(side=tk.LEFT, padx=5)
font_menu = ttk.Combobox(font_frame, textvariable=font_var, values=available_fonts)
font_menu.pack(side=tk.LEFT, padx=5)
font_menu.bind('<<ComboboxSelected>>', lambda e: update_font())

# Font size control
font_size_var = tk.StringVar(value="14")
font_size_label = tk.Label(font_frame, text="Size:", fg="white", bg="black")
font_size_label.pack(side=tk.LEFT, padx=5)
font_size_scale = ttk.Scale(font_frame, from_=8, to=72, variable=font_size_var, orient="horizontal", length=200)
font_size_scale.pack(side=tk.LEFT, padx=5)
font_size_scale.bind('<Motion>', lambda e: update_font())

# Corpus Management Frame
corpus_frame = tk.Frame(root, bg="black")
corpus_frame.pack(fill="x", padx=5, pady=5)

corpus_buttons_frame = tk.Frame(corpus_frame, bg="black")
corpus_buttons_frame.pack(side=tk.LEFT, padx=5)

import_button = tk.Button(corpus_buttons_frame, text="Import Text Files", command=import_corpus, bg="gray", fg="black")
import_button.pack(fill="x", pady=2)

remove_button = tk.Button(corpus_buttons_frame, text="Remove Selected", command=remove_selected_files, bg="gray", fg="black")
remove_button.pack(fill="x", pady=2)

rebuild_button = tk.Button(corpus_buttons_frame, text="Rebuild Corpus", command=rebuild_corpus, bg="gray", fg="black")
rebuild_button.pack(fill="x", pady=2)

corpus_files_frame = tk.Frame(corpus_frame, bg="black")
corpus_files_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5)

corpus_files_label = tk.Label(corpus_files_frame, text="Available Files (Select to Build Corpus):", font=("Arial", 12), fg="white", bg="black")
corpus_files_label.pack(fill="x")

# Add weight controls frame
weights_frame = tk.Frame(corpus_files_frame, bg="black")
weights_frame.pack(fill="x", pady=2)

corpus_files_listbox = tk.Listbox(corpus_files_frame, selectmode=tk.MULTIPLE, bg="gray", fg="black", height=5)
corpus_files_listbox.pack(fill="both", expand=True)

status_label = tk.Label(root, text="", font=("Arial", 12), fg="white", bg="black")
status_label.pack()

word_counts_label = tk.Label(root, text="Word Counts:", font=("Arial", 12), fg="white", bg="black", justify="left")
word_counts_label.pack()

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100, length=300)
progress_bar.pack()

progress_label = tk.Label(root, text="Progress: 0%", font=("Arial", 14), fg="white", bg="black")
progress_label.pack()

text_frame = tk.Frame(root, bg="black")
text_frame.pack(expand=True, fill="both")

text_widget = tk.Text(text_frame, font=("Arial", 14), fg="white", bg="black", wrap=tk.WORD, height=10)
text_widget.pack(expand=True, fill="both")
text_widget.tag_configure("center", justify='center')

scrollbar = tk.Scrollbar(text_frame, command=text_widget.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
text_widget.config(yscrollcommand=scrollbar.set, state='disabled')

input_label = tk.Label(root, text="Input Starting Bigram: ", font=("Arial", 14), fg="white", bg="black")
input_label.pack()

input_text_var = tk.StringVar()
input_text = tk.Entry(root, textvariable=input_text_var, font=("Arial", 14), fg="black")
input_text.pack()

button_frame = tk.Frame(root, bg="black")
button_frame.pack(fill="x", padx=5, pady=5)

tts_status = tk.Label(button_frame, text="TTS: Enabled", font=("Arial", 14), fg="white", bg="black")
tts_status.pack(side=tk.LEFT, padx=5)

toggle_button = tk.Button(button_frame, text="Toggle TTS", font=("Arial", 14), command=toggle_tts, bg="gray", fg="black")
toggle_button.pack(side=tk.LEFT, padx=5)

logging_button = tk.Button(button_frame, text="Toggle Logging", font=("Arial", 14), command=toggle_logging, bg="red", fg="black")
logging_button.pack(side=tk.LEFT, padx=5)

entropy_label = tk.Label(root, text="Entropy:", font=("Arial", 14), fg="white", bg="black", justify="left")
entropy_label.pack()

# Add sleep duration control frame
sleep_control_frame = tk.Frame(root, bg="black")
sleep_control_frame.pack(fill="x", padx=5, pady=5)

sleep_label = tk.Label(sleep_control_frame, text="Output Interval (seconds):", font=("Arial", 14), fg="white", bg="black")
sleep_label.pack(side=tk.LEFT, padx=5)

sleep_scale = ttk.Scale(sleep_control_frame, from_=0.1, to=10.0, variable=sleep_duration, orient="horizontal", length=200)
sleep_scale.pack(side=tk.LEFT, padx=5)

sleep_value_label = tk.Label(sleep_control_frame, text="3.0s", font=("Arial", 14), fg="white", bg="black")
sleep_value_label.pack(side=tk.LEFT, padx=5)

def update_sleep_label(event=None):
    sleep_value_label.config(text=f"{sleep_duration.get():.1f}s")

sleep_scale.bind("<Motion>", update_sleep_label)

greedy_mode = tk.BooleanVar(value=False)
greedy_checkbox = tk.Checkbutton(root, text="Greedy Mode", variable=greedy_mode, font=("Arial", 14), fg="white", bg="black", selectcolor="black")
greedy_checkbox.pack()

# Start loops in background threads
threading.Thread(target=text_loop, daemon=True).start()
threading.Thread(target=audio_loop, daemon=True).start()

# Start processing queues
process_banter_queue()

# Run Tkinter event loop
root.mainloop()
