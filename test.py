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
        
        self.input_device = input_device
        self.output_device = output_device
        
        # Voice Conversion 모델 초기화
        print("Voice Conversion 모델 로딩 중...")
        self.converter = ToneColorConverter(ckpt_path=model_path, device=device)
        
        # 타겟 보이스 특성 추출
        print("타겟 보이스 특성 추출 중...")
        tgt_spec = self.converter.get_spec(fpath=target_voice_path)
        self.target_se = self.converter.model.extract_se(tgt_spec)
        
        self.input_queue = queue.Queue(maxsize=3)
        self.output_queue = queue.Queue(maxsize=3)
        
        self.is_running = False
        self.drop_count = 0
        self.total_latency = 0
        self.process_count = 0
        
        pass
        
    def start(self):
        self.is_running = True
        self.processor_thread = threading.Thread(target=self._process_audio)
        self.processor_thread.start()
        
        def audio_callback(indata, outdata, frames, time, status):
            if status:
                print(status)
            
            try:
                if not self.input_queue.full():
                    audio_data = indata.copy().flatten()
                    self.input_queue.put_nowait(audio_data)
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
            latency='low'
        )
        self.stream.start()
        print("실시간 음성 변환 시작...")
    
    def _process_audio(self):
        while self.is_running:
            try:
                audio_chunk = self.input_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Voice Conversion 수행
                src_spec = self.converter.get_spec(wav=audio_chunk)
                converted_audio, _ = self.converter.convert(src_spec, self.target_se)
                    
                # 후처리
                processed_audio = np.nan_to_num(converted_audio)
                processed_audio = processed_audio.copy()
                
                # 페이드 인/아웃 효과
                fade_len = min(512, len(processed_audio) // 4)
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)
                
                processed_audio[:fade_len] *= fade_in
                processed_audio[-fade_len:] *= fade_out
                
                processed_audio = np.clip(processed_audio, -1.0, 1.0)
                
                self.output_queue.put_nowait(processed_audio)
                
                process_time = time.time() - start_time
                self.total_latency += process_time
                self.process_count += 1
                
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"음성 변환 오류: {e}")
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
            'processed_chunks': self.process_count
        }

def main():
    try:
        print("\n사용 가능한 오디오 장치:")
        print(sd.query_devices())
        
        input_device = -1
        output_device = -1
        
        input_device = None if input_device == -1 else input_device
        output_device = None if output_device == -1 else output_device
        
        converter = RealtimeVoiceConverter(
            model_path='VoiceConversion/model.pth',
            target_voice_path='samples/tsu.wav',
            device='cpu',
            input_device=input_device,
            output_device=output_device
        )
        
        print("\n음성 변환 시작... Ctrl+C로 중지")
        converter.start()
        
        while True:
            time.sleep(1)
            stats = converter.get_stats()
            print(f"상태: {stats}")
            
    except KeyboardInterrupt:
        print("\n음성 변환 중지 중...")
        if 'converter' in locals():
            converter.stop()
    except Exception as e:
        print(f"오류 발생: {e}")
        if 'converter' in locals():
            converter.stop()

if __name__ == "__main__":
    main()