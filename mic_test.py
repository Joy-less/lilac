import pyaudio
import numpy as np

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050

p = pyaudio.PyAudio()

def callback(in_data, frame_count, time_info, status):
    return (in_data, pyaudio.paContinue)

print("\nAvailable audio devices:")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    print(f"[{i}] {dev['name']}")

input_device = int(input("\nEnter input device index (-1 for default): "))
output_device = int(input("Enter output device index (-1 for default): "))

input_device = None if input_device == -1 else input_device
output_device = None if output_device == -1 else output_device

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                input_device_index=input_device,
                output_device_index=output_device,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

print("\nStarting echo... Press Ctrl+C to stop")
stream.start_stream()

try:
    while stream.is_active():
        pass
except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("\nStopped echo")