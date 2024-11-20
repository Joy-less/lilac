import warnings
warnings.filterwarnings('ignore')

import sounddevice as sd
import queue
import threading
import numpy as np
import time
from VoiceConversion import ToneColorConverter

class RealtimeVoiceConverter:
    def __init__(self, model_path, target_voice_path, device='cpu', input_device=None, output_device=None):
        self.CHUNK = 9984
        self.RATE = 22050
        self.CHANNELS = 1
        
        # 아주 짧은 크로스페이드 구간 (약 5ms)
        self.CROSSFADE_SIZE = int(self.RATE * 0.005)  
        
        self.input_device = input_device
        self.output_device = output_device
        
        self.converter = ToneColorConverter(ckpt_path=model_path, device=device)
        
        tgt_spec = self.converter.get_spec(fpath=target_voice_path)
        self.target_se = self.converter.model.extract_se(tgt_spec)
        
        # 버퍼 크기 증가
        self.chunk_buffer = []
        self.input_queue = queue.Queue(maxsize=4)  # 버퍼 크기 증가
        self.output_queue = queue.Queue(maxsize=4)
        
        # 이전 청크의 마지막 부분만 아주 짧게 저장
        self.prev_chunk_end = None
        
        self.is_running = False
        self.drop_count = 0
        self.total_latency = 0
        self.process_count = 0
        
    def apply_short_crossfade(self, chunk):
        """아주 짧은 크로스페이드만 적용"""
        if self.prev_chunk_end is None:
            self.prev_chunk_end = chunk[-self.CROSSFADE_SIZE:]
            return chunk
            
        # 부드러운 사인 커브를 사용한 크로스페이드
        fade_in = np.sin(np.linspace(0, np.pi/2, self.CROSSFADE_SIZE))**2
        fade_out = np.cos(np.linspace(0, np.pi/2, self.CROSSFADE_SIZE))**2
        
        # 현재 청크의 시작 부분에만 아주 짧은 크로스페이드 적용
        chunk_start = chunk[:self.CROSSFADE_SIZE]
        crossfaded = (self.prev_chunk_end * fade_out + chunk_start * fade_in)
        chunk[:self.CROSSFADE_SIZE] = crossfaded
        
        # 다음 크로스페이드를 위해 현재 청크의 끝부분 저장
        self.prev_chunk_end = chunk[-self.CROSSFADE_SIZE:]
        
        return chunk
    
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
                    outdata[:] = output_data.reshape(-1, 1)
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
    
    def _process_audio(self):
        while self.is_running:
            try:
                audio_chunk = self.input_queue.get(timeout=0.1)
                self.chunk_buffer.append(audio_chunk)
                
                if len(self.chunk_buffer) == 3:
                    start_time = time.time()
                    
                    # 3개의 청크를 연결
                    combined_audio = np.concatenate(self.chunk_buffer)
                    
                    # 변환 수행
                    src_spec = self.converter.get_spec(wav=combined_audio)
                    converted_audio, _ = self.converter.convert(src_spec, self.target_se)
                    
                    # 변환된 오디오 처리
                    converted_audio = np.nan_to_num(converted_audio)
                    converted_audio = np.clip(converted_audio, -1.0, 1.0)
                    
                    # 중간 청크 추출
                    chunk_length = len(converted_audio) // 3
                    middle_chunk = converted_audio[chunk_length:2*chunk_length]
                    
                    # # 볼륨 정규화
                    # if np.max(np.abs(middle_chunk)) > 0:
                    #     middle_chunk = middle_chunk / np.max(np.abs(middle_chunk)) * 0.9
                    
                    # 아주 짧은 크로스페이드 적용
                    middle_chunk = self.apply_short_crossfade(middle_chunk)
                    
                    # 출력 큐에 전송
                    self.output_queue.put_nowait(middle_chunk)
                    
                    # 버퍼 관리 개선 - 한 번에 하나의 청크만 제거
                    self.chunk_buffer = self.chunk_buffer[1:]
                    
                    process_time = time.time() - start_time
                    self.total_latency += process_time
                    self.process_count += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing: {e}")
                continue
    
    def stop(self):
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        if hasattr(self, 'processor_thread'):
            self.processor_thread.join()
    
    def get_stats(self):
        if self.process_count == 0:
            avg_latency = 0
        else:
            avg_latency = (self.total_latency / self.process_count) * 1000
        
        return {
            'dropped_frames': self.drop_count,
            'average_latency': f"{avg_latency:.1f}ms",
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'processed_chunks': self.process_count,
            'buffer_size': len(self.chunk_buffer)
        }

def main():
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    
    input_device = -1  # int(input("\nEnter input device index (-1 for default): "))
    output_device = -1  # int(input("Enter output device index (-1 for default): "))
    
    input_device = None if input_device == -1 else input_device
    output_device = None if output_device == -1 else output_device
    
    converter = RealtimeVoiceConverter(
        model_path='VoiceConversion/model.pth',
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