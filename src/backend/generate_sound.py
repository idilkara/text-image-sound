import numpy as np
from PIL import Image
from hilbertcurve.hilbertcurve import HilbertCurve
from pydub import AudioSegment
from pydub.generators import Sine
import random
import io

# Convert PNG image to data points
def image_to_data(image_stream):
    # Open the image from the in-memory bytes
    img = Image.open(image_stream)  # image_stream should be a BytesIO object
    img = img.convert('RGB')  # Convert to RGB (or another format as needed)
    
    # Extract pixel data
    pixel_data = list(img.getdata())  # This gives you a flattened list of RGB values
    img_size = img.size  # This will return (width, height)

    return pixel_data, img_size
# Generate Hilbert curve based on pixel values
def hilbert_mapping(pixel_data, curve_order=6):
    # Create a Hilbert curve
    curve = HilbertCurve(curve_order, 2)
    
    # Ensure the numbers are within the Hilbert curve's range
    max_distance = 4**curve_order - 1

    # Instead of applying modulo on the tuple, handle each RGB component separately
    normalized_data = [
        (r + g + b) % (max_distance + 1)  # Sum the RGB components for normalization
        for r, g, b in pixel_data
    ]
    
    # Map pixel values to Hilbert curve points
    mapped_points = [curve.point_from_distance(value) for value in normalized_data]
    return mapped_points
# Map the Hilbert curve points to musical attributes
def map_to_music(mapped_points, pixel_values):
    music_notes = []
    for (x, y), (r, g, b) in zip(mapped_points, pixel_values):
        # Map pixel brightness to pitch (darker = lower, brighter = higher)
        # Use the average of the RGB values for simplicity
        brightness = (r + g + b) / 3  # Compute the average brightness of the pixel
        
        # MIDI note range from 40 (E2) to 84 (C6)
        pitch = 40 + int((brightness / 255) * 44)
        
        # Map position to duration
        duration = 50 + (x % 150) + (y % 150)  # Between 50ms and 350ms
        
        music_notes.append((pitch, duration))
    return music_notes


# Generate sound
def generate_music(notes, max_length_ms=60000):
    sound = AudioSegment.silent(duration=0)
    total_duration = 0
    
    for pitch, duration in notes:
        # Check if we've reached the maximum length
        if total_duration >= max_length_ms:
            break
            
        # Convert MIDI pitch to frequency
        frequency = 440 * (2 ** ((pitch - 69) / 12))
        
        # Generate a sine wave for the given pitch and duration
        sine_generator = Sine(frequency)
        tone = sine_generator.to_audio_segment(duration=duration)
        
        # Add a small fade to avoid clicks
        tone = tone.fade_in(10).fade_out(10)
        
        # Add the tone to our sound
        sound += tone
        total_duration += duration
    
    return sound

def png_to_music(image_data, output_stream=None, max_length_seconds=60):
    """
    Converts PNG image bytes to music and writes WAV to an output stream.

    Args:
        image_data (bytes or io.BytesIO): PNG image bytes or BytesIO stream.
        output_stream (io.BytesIO): Stream to write WAV data to.
        max_length_seconds (int): Max duration for sound generation.

    Returns:
        io.BytesIO: The WAV audio stream (if output_stream not provided).
    """
    # Ensure image_data is a BytesIO stream
    if isinstance(image_data, bytes):
        image_io = io.BytesIO(image_data)  # Convert bytes to BytesIO
    elif isinstance(image_data, io.BytesIO):
        image_io = image_data
    else:
        raise ValueError("image_data must be bytes or BytesIO")

    # Open the image and extract pixel data
    image = Image.open(image_io)
    pixel_data, img_size = image_to_data(image_io)

    # Map to Hilbert curve
    mapped_points = hilbert_mapping(pixel_data)

    # Create musical notes
    music_notes = map_to_music(mapped_points, pixel_data)

    # Generate the sound
    sound = generate_music(music_notes, max_length_seconds * 1000)

    # Prepare output stream
    if output_stream is None:
        output_stream = io.BytesIO()

    # Export sound to stream
    sound.export(output_stream, format="wav")
    output_stream.seek(0)  # Important for reading later

    # Optional print logs (can remove or log elsewhere)
    print(f"Image size: {img_size[0]}x{img_size[1]} pixels")
    print(f"Generated {len(music_notes)} notes, used {len(sound) / 1000:.2f} seconds")

    return output_stream
