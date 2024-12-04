import numpy as np
import cv2
import soundfile as sf
from moviepy.editor import VideoClip
from PIL import Image, ImageDraw
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

def audio_to_hex(audio_path):
    """
    Reads an audio file and converts its samples to 24-bit hex codes.
    """
    try:
        # Read audio file
        audio_data, sample_rate = sf.read(audio_path)

        # If stereo, take the mean to convert to mono
        if len(audio_data.shape) == 2:
            audio_data = audio_data.mean(axis=1)

        duration = len(audio_data) / sample_rate

        # Normalize audio data to range [-1, 1]
        if np.max(np.abs(audio_data)) != 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # Convert to 24-bit signed integers
        max_amplitude = 2**23 - 1
        samples_int = (audio_data * max_amplitude).astype(np.int32)

        # Convert integer samples to hex codes
        hex_codes = [format(sample & 0xFFFFFF, '06x') for sample in samples_int]

        return hex_codes, sample_rate, duration
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process audio file.\n{e}")
        return None, None, None

def generate_color_strip(hex_codes):
    """
    Converts hex codes to RGB tuples.
    """
    colors = []
    for idx, hex_code in enumerate(hex_codes):
        try:
            color = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
            colors.append(color)
        except:
            colors.append((0, 0, 0))  # Fallback to black in case of error
        if (idx + 1) % 100000 == 0:
            print(f"Converted {idx + 1} / {len(hex_codes)} samples to colors")
    return colors

def make_frame(t, colors, samples_per_frame, frame_number, resolution, strip_width, max_blocks):
    """
    Generates a single video frame at time t.
    """
    start_idx = frame_number * samples_per_frame
    end_idx = start_idx + max_blocks

    # Get the samples for the current frame
    current_strip = colors[start_idx:end_idx]

    # If not enough samples, pad with black
    if len(current_strip) < max_blocks:
        padding = [(0, 0, 0)] * (max_blocks - len(current_strip))
        current_strip += padding

    # Create an image with the current color strip
    img = Image.new('RGB', resolution, color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    for i, color in enumerate(current_strip):
        x0 = i * strip_width
        y0 = 0
        x1 = x0 + strip_width
        y1 = resolution[1]
        draw.rectangle([x0, y0, x1, y1], fill=color)

    return np.array(img)

def generate_video_stream(audio_path, colors, sample_rate, duration, output_video, frame_rate=60, resolution=(3840, 2160), strip_width=8):
    """
    Generates a video from colors synchronized with the audio using a color strip.
    """
    try:
        # Calculate the number of color blocks that fit in the frame width
        frame_width, frame_height = resolution
        max_blocks = frame_width // strip_width

        # Calculate samples per frame
        samples_per_frame = max_blocks

        # Total number of frames needed
        total_frames = int(np.ceil(len(colors) / samples_per_frame))

        print("Creating video clip with streaming frames...")

        # Define a VideoClip with a frame generator
        def frame_generator(t):
            frame_number = int(t * frame_rate)
            return make_frame(t, colors, samples_per_frame, frame_number, resolution, strip_width, max_blocks)

        # Create the video clip
        clip = VideoClip(make_frame=frame_generator, duration=duration)

        # Save parameters to metadata file
        base_name, ext = os.path.splitext(output_video)
        metadata_file = f"{base_name}_metadata.txt"
        with open(metadata_file, 'w') as f:
            f.write(f"sample_rate={int(sample_rate)}\n")
            f.write(f"strip_width={strip_width}\n")
            f.write(f"samples_per_frame={samples_per_frame}\n")

        # Write the video file using a lossless codec
        print(f"Writing the video file to {output_video}...")
        clip.write_videofile(
            output_video,
            codec='libx264rgb',
            audio_codec='none',
            fps=frame_rate,
            preset='ultrafast',
            ffmpeg_params=['-pix_fmt', 'rgb24', '-crf', '0']
        )

        print("Video creation completed.")
        messagebox.showinfo("Success", f"Video has been saved as {output_video}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to generate video.\n{e}")

def start_encoding(audio_path, output_video, frame_rate, resolution, strip_width):
    hex_codes, sample_rate, duration = audio_to_hex(audio_path)
    if hex_codes:
        print("Converting hex codes to colors...")
        colors = generate_color_strip(hex_codes)
        print("Starting video generation...")
        generate_video_stream(audio_path, colors, sample_rate, duration, output_video, frame_rate, resolution, strip_width)

def read_metadata(video_path):
    metadata_file = os.path.splitext(video_path)[0] + "_metadata.txt"
    if os.path.exists(metadata_file):
        params = {}
        with open(metadata_file, 'r') as f:
            for line in f:
                key, value = line.strip().split('=')
                params[key] = int(value)
        return params
    else:
        return None

def read_video_extract_samples(video_path, strip_width, samples_per_frame):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Total frames: {total_frames}")
    print(f"Samples per frame: {samples_per_frame}")

    samples = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Ensure the frame is in RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_width = frame.shape[1]
        # Recalculate samples per frame based on frame width and strip_width
        samples_in_frame = frame_width // strip_width

        # For each strip along the horizontal axis
        for i in range(samples_in_frame):
            x = i * strip_width + strip_width // 2
            y = frame.shape[0] // 2  # Middle row

            # Get the color at (x, y)
            R, G, B = frame[y, x]

            # Combine RGB to get 24-bit integer
            sample_int = (R << 16) | (G << 8) | B

            # Convert to signed 24-bit integer
            if sample_int & 0x800000:
                sample_int -= 0x1000000  # Sign extension for negative values

            # Store the integer sample directly
            samples.append(sample_int)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    samples = np.array(samples, dtype=np.int32)

    # Normalize samples to [-1, 1]
    max_amplitude = 2**23 - 1
    samples = samples / max_amplitude

    return samples

def start_decoding(video_path, output_audio_path):
    # Read metadata
    params = read_metadata(video_path)
    if params:
        sample_rate = params.get('sample_rate', None)
        strip_width = params.get('strip_width', None)
        samples_per_frame = params.get('samples_per_frame', None)
    else:
        # Prompt user to input parameters
        def prompt_params():
            sample_rate_input = sample_rate_var.get()
            strip_width_input = strip_width_var.get()
            if not sample_rate_input or not strip_width_input:
                messagebox.showwarning("Warning", "Sample rate and strip width are required.")
                return
            try:
                sample_rate = int(sample_rate_input)
                strip_width = int(strip_width_input)
                decoding_thread = threading.Thread(
                    target=decode_samples,
                    args=(video_path, output_audio_path, sample_rate, strip_width)
                )
                decoding_thread.start()
                param_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Invalid sample rate or strip width entered.")
                return

        # Create a new window for parameter input
        param_window = tk.Toplevel(root)
        param_window.title("Enter Parameters")

        tk.Label(param_window, text="Sample Rate (e.g., 44100):").grid(row=0, column=0)
        sample_rate_var = tk.StringVar()
        tk.Entry(param_window, textvariable=sample_rate_var).grid(row=0, column=1)

        tk.Label(param_window, text="Strip Width (e.g., 8):").grid(row=1, column=0)
        strip_width_var = tk.StringVar()
        tk.Entry(param_window, textvariable=strip_width_var).grid(row=1, column=1)

        tk.Button(param_window, text="OK", command=prompt_params).grid(row=2, column=0, columnspan=2)

        return
    # If parameters are available, start decoding
    decoding_thread = threading.Thread(
        target=decode_samples,
        args=(video_path, output_audio_path, sample_rate, strip_width, samples_per_frame)
    )
    decoding_thread.start()

def decode_samples(video_path, output_audio_path, sample_rate, strip_width, samples_per_frame):
    samples = read_video_extract_samples(video_path, strip_width=strip_width, samples_per_frame=samples_per_frame)
    if samples is not None:
        print(f"Writing audio file to {output_audio_path}")
        try:
            sf.write(output_audio_path, samples, sample_rate)
            print("Audio reconstruction completed.")
            # Schedule the success message in the main thread
            root.after(0, lambda: messagebox.showinfo("Success", f"Audio has been reconstructed and saved to {output_audio_path}"))
        except Exception as e:
            print(f"Failed to write audio file.\n{e}")
            # Schedule the error message in the main thread
            root.after(0, lambda: messagebox.showerror("Error", f"Failed to write audio file.\n{e}"))
    else:
        print("Failed to reconstruct audio.")
        # Schedule the error message in the main thread
        root.after(0, lambda: messagebox.showerror("Error", "Failed to reconstruct audio."))

def select_encode():
    audio_path = filedialog.askopenfilename(
        title="Select an Audio File",
        filetypes=[("Audio Files", "*.wav *.flac *.mp3 *.aiff *.aac"), ("All Files", "*.*")]
    )
    if audio_path:
        output_video = filedialog.asksaveasfilename(
            title="Save Video As",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")]
        )
        if output_video:
            # Optional: Allow user to set frame rate and resolution
            # For simplicity, we'll use default values
            frame_rate = 60  # Default frame rate
            resolution = (3840, 2160)  # Use 4K resolution
            strip_width = 8  # Width of each color block in pixels

            # Run processing in a separate thread to keep the GUI responsive
            encoding_thread = threading.Thread(
                target=start_encoding,
                args=(audio_path, output_video, frame_rate, resolution, strip_width)
            )
            encoding_thread.start()
        else:
            messagebox.showwarning("Warning", "No output video file selected.")
    else:
        messagebox.showwarning("Warning", "No audio file selected.")

def select_decode():
    video_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
    )
    if video_path:
        output_audio_path = filedialog.asksaveasfilename(
            title="Save Audio As",
            defaultextension=".wav",
            filetypes=[("WAV Audio", "*.wav")]
        )
        if output_audio_path:
            # Start decoding process
            start_decoding(video_path, output_audio_path)
        else:
            messagebox.showwarning("Warning", "No output audio file selected.")
    else:
        messagebox.showwarning("Warning", "No video file selected.")

# Initialize Tkinter root
root = tk.Tk()
root.title("Audio-Video Encoder/Decoder")
root.geometry("400x200")

# Add buttons to select encode or decode
encode_button = tk.Button(root, text="Encode Audio to Video", command=select_encode, font=("Helvetica", 16))
encode_button.pack(expand=True, pady=10)

decode_button = tk.Button(root, text="Decode Video to Audio", command=select_decode, font=("Helvetica", 16))
decode_button.pack(expand=True, pady=10)

# Run the Tkinter event loop
root.mainloop()
