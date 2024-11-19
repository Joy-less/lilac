import warnings
warnings.filterwarnings('ignore')

import pyaudio
import queue
import threading
import numpy as np
import time
from VoiceConversion import ToneColorConverter

class RealtimeVoiceConverter:
    def __init__(self, model_path, target_voice_path, device='cpu', input_device=None, output_device=None):
        # 오디오 설정
        self.CHUNK = 8192  # 1024에서 2048로 증가
        self.FORMAT = pyaudio.paFloat32
        self.RATE = 22050
        self.CHANNELS = 1
        
        # 디바이스 설정
        self.input_device = input_device
        self.output_device = output_device
        
        # PyAudio 초기화
        self.p = pyaudio.PyAudio()
        
        # 음성 변환 모델 초기화
        self.converter = ToneColorConverter(ckpt_path=model_path, device=device)
        
        # 타겟 보이스 미리 처리
        tgt_spec = self.converter.get_spec(fpath=target_voice_path)
        self.target_se = self.converter.model.extract_se(tgt_spec)
        
        # 큐 초기화
        self.input_queue = queue.Queue(maxsize=3)  # 3에서 10으로 증가
        self.output_queue = queue.Queue(maxsize=3)  # 3에서 10으로 증가
        
        # 상태 및 통계
        self.is_running = False
        self.drop_count = 0
        self.total_latency = 0
        self.process_count = 0
        
    def _input_callback(self, in_data, frame_count, time_info, status):
        try:
            if not self.input_queue.full():
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                self.input_queue.put_nowait(audio_data)
            else:
                self.drop_count += 1
        except queue.Full:
            pass
        return (None, pyaudio.paContinue)

    def _output_callback(self, in_data, frame_count, time_info, status):
        try:
            if not self.output_queue.empty():
                output_data = self.output_queue.get_nowait()
                # float32로 변환하고 볼륨 조정
                output_data = output_data.astype(np.float32) * 0.8
                return (output_data.tobytes(), pyaudio.paContinue)
            else:
                return (np.zeros(frame_count, dtype=np.float32).tobytes(), pyaudio.paContinue)
        except queue.Empty:
            return (np.zeros(frame_count, dtype=np.float32).tobytes(), pyaudio.paContinue)
    
    def start(self):
        """스트리밍 시작"""
        self.is_running = True
        self.processor_thread = threading.Thread(target=self._process_audio)
        self.processor_thread.start()
        
        # 입력 스트림 시작
        self.input_stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            input_device_index=self.input_device,
            frames_per_buffer=self.CHUNK,
            stream_callback=self._input_callback
        )
        
        # 출력 스트림 시작
        self.output_stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            output=True,
            output_device_index=self.output_device,
            frames_per_buffer=self.CHUNK,
            stream_callback=self._output_callback
        )
        
        self.input_stream.start_stream()
        self.output_stream.start_stream()
    
    def _process_audio(self):
        while self.is_running:
            try:
                audio_chunk = self.input_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                src_spec = self.converter.get_spec(wav=audio_chunk)
                converted_audio, _ = self.converter.convert(src_spec, self.target_se)
                
                # 오디오 데이터 정규화
                converted_audio = np.nan_to_num(converted_audio)  # NaN 값 처리
                converted_audio = np.clip(converted_audio, -1.0, 1.0)  # 클리핑
                
                # Smooth transition을 위한 페이드 인/아웃
                fade_len = min(512, len(converted_audio) // 4)
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)
                
                converted_audio[:fade_len] *= fade_in
                converted_audio[-fade_len:] *= fade_out
                
                self.output_queue.put_nowait(converted_audio)
                
                process_time = time.time() - start_time
                self.total_latency += process_time
                self.process_count += 1
                
            except queue.Empty:
                continue
    
    def stop(self):
        """변환 중지"""
        self.is_running = False
        
        if hasattr(self, 'input_stream'):
            self.input_stream.stop_stream()
            self.input_stream.close()
        
        if hasattr(self, 'output_stream'):
            self.output_stream.stop_stream()
            self.output_stream.close()
            
        if hasattr(self, 'processor_thread'):
            self.processor_thread.join()
            
        self.p.terminate()
    
    def get_stats(self):
        """현재 통계 반환"""
        if self.process_count == 0:
            avg_latency = 0
        else:
            avg_latency = (self.total_latency / self.process_count) * 1000  # ms로 변환
        
        return {
            'dropped_frames': self.drop_count,
            'average_latency': f"{avg_latency:.1f}ms",
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'processed_chunks': self.process_count
        }

def main():
    # 사용 가능한 오디오 디바이스 출력
    p = pyaudio.PyAudio()
    print("\nAvailable audio devices:")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        print(f"[{i}] {dev['name']}")
    p.terminate()
    
    # 사용자로부터 디바이스 선택 받기
    input_device = int(input("\nEnter input device index (-1 for default): "))
    output_device = int(input("Enter output device index (-1 for default): "))
    
    # -1이면 None으로 설정 (기본 디바이스 사용)
    input_device = None if input_device == -1 else input_device
    output_device = None if output_device == -1 else output_device
    
    # 초기화
    converter = RealtimeVoiceConverter(
        model_path='VoiceConversion/model.pth',
        target_voice_path='buk.wav',
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