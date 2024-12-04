from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# Download and load all models
preload_models()

# Generate audio from text
text_prompt = "[Man][Woman] Hello, this is a test of Mira running on CPU."
audio_array = generate_audio(text_prompt)

# Save the audio to a WAV file
write_wav("output.wav", SAMPLE_RATE, audio_array)

what determines the model used, it generated a male voice. how do i know which model its using? i know there are lots of bark models
