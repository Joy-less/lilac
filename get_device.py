import sounddevice as sd

def print_audio_devices():
    devices = sd.query_devices()
    print("\nAvailable Audio Devices:")
    print("-" * 80)
    print(f"{'ID':<4} | {'Name':<32} | {'Channels (in/out)':<20} | {'Default':<8}")
    print("-" * 80)
    
    default_input = sd.query_devices(kind='input')
    default_output = sd.query_devices(kind='output')
    
    for i, device in enumerate(devices):
        is_default = []
        if device['name'] == default_input['name'] and device['hostapi'] == default_input['hostapi']:
            is_default.append('input')
        if device['name'] == default_output['name'] and device['hostapi'] == default_output['hostapi']:
            is_default.append('output')
        default_str = ','.join(is_default) if is_default else ''
        
        channels = f"{device['max_input_channels']}/{device['max_output_channels']}"
        
        print(f"{i:<4} | {device['name']:<32} | {channels:<20} | {default_str:<8}")

    print("-" * 80)
    print("\nDefault Devices:")
    print(f"Default Input Device: {default_input['name']}")
    print(f"Default Output Device: {default_output['name']}")

if __name__ == "__main__":
    print_audio_devices()