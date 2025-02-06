import warnings
warnings.filterwarnings('ignore')

import sounddevice as sd
import queue
import threading
import numpy as np
import time
from vc import ToneColorConverter

class RealtimeVoiceConverter:
    def __init__(self, model_path, target_voice_path, device='cpu', input_device=None, output_device=None):
        self.CHUNK = 9984
        self.RATE = 22050
        self.CHANNELS = 1
        self.CROSSFADE_SIZE = int(self.RATE * 0.005)
        self.SPEECH_THRESHOLD = 0.015

        self.SILENCE_CHUNK = np.zeros(self.CHUNK, dtype=np.float32)
        
        self.input_device = input_device
        self.output_device = output_device
        
        self.converter = ToneColorConverter(ckpt_path=model_path, device=device)
        tgt_spec = self.converter.get_spec(fpath=target_voice_path)
        self.target_se = self.converter.model.extract_se(tgt_spec)
        
        self.chunk_buffer = []
        self.chunk_speech_status = []
        self.input_queue = queue.Queue(maxsize=4)
        self.output_queue = queue.Queue(maxsize=4)
        
        self.prev_chunk_end = None
        self.last_was_speech = False
        
        self.is_running = False
        self.drop_count = 0
        self.total_latency = 0
        self.process_count = 0
    
    def is_speech(self, audio_chunk):
        energy = np.mean(np.abs(audio_chunk))
        return energy > self.SPEECH_THRESHOLD
    
    def apply_short_crossfade(self, chunk):
        if self.prev_chunk_end is None:
            self.prev_chunk_end = chunk[-self.CROSSFADE_SIZE:]
            return chunk
            
        fade_in = np.sin(np.linspace(0, np.pi/2, self.CROSSFADE_SIZE))**2
        fade_out = np.cos(np.linspace(0, np.pi/2, self.CROSSFADE_SIZE))**2
        
        chunk_start = chunk[:self.CROSSFADE_SIZE]
        crossfaded = (self.prev_chunk_end * fade_out + chunk_start * fade_in)
        chunk[:self.CROSSFADE_SIZE] = crossfaded
        
        self.prev_chunk_end = chunk[-self.CROSSFADE_SIZE:]
        return chunk

    def _process_audio(self):
        while self.is_running:
            try:
                audio_chunk = self.input_queue.get(timeout=0.1)
                is_current_speech = self.is_speech(audio_chunk)
                
                self.chunk_buffer.append(audio_chunk)
                self.chunk_speech_status.append(is_current_speech)

                if len(self.chunk_buffer) == 3:
                    start_time = time.time()
                    middle_chunk_speech = self.chunk_speech_status[1]
                    force_convert = self.last_was_speech and not middle_chunk_speech

                    if middle_chunk_speech or force_convert:
                        # Convert speech chunks
                        combined_audio = np.concatenate(self.chunk_buffer)
                        src_spec = self.converter.get_spec(wav=combined_audio)
                        converted = self.converter.convert(src_spec, self.target_se)[0]
                        converted = np.nan_to_num(converted)
                        converted = np.clip(converted, -1.0, 1.0)
                        
                        chunk_length = len(converted) // 3
                        middle_chunk = converted[chunk_length:2*chunk_length]
                    else:
                        # Generate silence for non-speech
                        middle_chunk = self.SILENCE_CHUNK
                    
                    middle_chunk = self.apply_short_crossfade(middle_chunk)
                    self.output_queue.put(middle_chunk)
                    
                    self.last_was_speech = middle_chunk_speech
                    self.chunk_buffer.pop(0)
                    self.chunk_speech_status.pop(0)
                    
                    process_time = time.time() - start_time
                    self.total_latency += process_time
                    self.process_count += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing: \"{e}\"")
                continue

    def start(self):
        self.is_running = True
        self.processor_thread = threading.Thread(target=self._process_audio)
        self.processor_thread.start()
        
        def audio_callback(indata, outdata, frames, time, status):
            if status:
                print(status)
            
            try:
                if not self.input_queue.full():
                    self.input_queue.put_nowait(indata.copy().flatten())
                else:
                    self.drop_count += 1
            except queue.Full:
                pass
            
            try:
                if not self.output_queue.empty():
                    output_data = self.output_queue.get_nowait()
                    output_data = output_data.astype(np.float32) * 0.8
                    
                    if output_data.size != frames:
                        if output_data.size > frames:
                            output_data = output_data[:frames]
                        else:
                            temp = np.zeros(frames, dtype=np.float32)
                            temp[:output_data.size] = output_data
                            output_data = temp
                    
                    outdata[:] = output_data.reshape(-1, self.CHANNELS)
                else:
                    outdata.fill(0)
            except queue.Empty:
                outdata.fill(0)
        
        self.stream = sd.Stream(
            channels=self.CHANNELS,
            samplerate=self.RATE,
            blocksize=self.CHUNK,
            dtype=np.float32,
            callback=audio_callback,
            device=(self.input_device, self.output_device),
            latency='high'
        )
        self.stream.start()

    def stop(self):
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        if hasattr(self, 'processor_thread'):
            self.processor_thread.join()

    def get_stats(self):
        avg_latency = (self.total_latency / self.process_count) * 1000 if self.process_count > 0 else 0
        return {
            'dropped_frames': self.drop_count,
            'average_latency': f"{avg_latency:.1f}ms",
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'processed_chunks': self.process_count,
            'buffer_size': len(self.chunk_buffer),
            'is_speech': self.last_was_speech
        }

def main():
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    
    input_device =  int(input("\nEnter input device index (-1 for default): "))
    output_device = int(input("Enter output device index (-1 for default): "))
    
    input_device = None if input_device == -1 else input_device
    output_device = None if output_device == -1 else output_device
    
    converter = RealtimeVoiceConverter(
        model_path='vc/model.pth',
        target_voice_path='samples/tsu.wav',
        device='cpu',
        input_device=input_device,
        output_device=output_device
    )
    
    print("\nStarting voice conversion... Press Ctrl+C to stop")
    converter.start()
    
    try:
        while True:
            time.sleep(1)
            stats = converter.get_stats()
            print(f"Status: {stats}")
    except KeyboardInterrupt:
        print("\nStopping voice conversion...")
        converter.stop()

if __name__ == "__main__":
    main()