import torch
from vc.models import SynthesizerTrn
import torchaudio
from torchaudio.transforms import Resample as AudioResample


class ToneColorConverter:
    def __init__(self, ckpt_path, device='cpu'):
        hps = {
            "data": {
                "sampling_rate": 22050,
                "filter_length": 1024,
                "hop_length": 256,
                "win_length": 1024,
                "n_speakers": 0
            },
            "model": {
                "spec_channels": 513,
                "inter_channels": 192,
                "hidden_channels": 192,
                "resblock": "1",
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_rates": [8, 8, 2, 2],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [16, 16, 4, 4],
                "gin_channels": 256,
                "zero_g": True,
                "device": device
            }
        }

        model = SynthesizerTrn(**hps['model'], **hps['data']).to(device)
        model.eval()
        self.model = model
        self.hps = hps
        self.device = device
        self.sampling_rate = self.hps['data']['sampling_rate']

        model_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))

        dequantized_dict = {}
        for key, value in model_dict.items():
            if isinstance(value, dict) and 'quantized' in value:
                dequantized_dict[key] = self.dequantize_tensor(
                    value['quantized'], 
                    value['scale'], 
                    value['zero_point']
                )
            else:
                dequantized_dict[key] = value
        self.model.load_state_dict(dequantized_dict, strict=False)


    def dequantize_tensor(self, quantized, scale, zero_point):
        return scale * quantized.float() + zero_point
    

    def spectrogram_torch(self, y, n_fft, sampling_rate, hop_size, win_size, center=False):
        hann_window = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
        y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect")
        y = y.squeeze(1)
        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center, pad_mode="reflect", normalized=False, onesided=True, return_complex=False)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        return spec


    def load_audio(self, fpath):
        desired_channels = 1
        audio, sr = torchaudio.load(fpath)
        
        if sr != self.sampling_rate:
            audio = AudioResample(orig_freq=sr, new_freq=self.sampling_rate)(audio)

        if audio.shape[0] != desired_channels:
            audio = audio.mean(dim=0, keepdim=True)
        return audio
    

    # def get_spec(self, fpath):
    #     hps = self.hps
    #     audio = self.load_audio(fpath)
    #     with torch.no_grad():
    #         y = audio.to(self.device)
    #         y = self.spectrogram_torch(y, hps['data']['filter_length'],
    #                                     hps['data']['sampling_rate'], hps['data']['hop_length'], hps['data']['win_length'],
    #                                     center=False).to(self.device)
    #     return y
    

    def load_audio_from_mic(self, audio_data):
        """
        마이크 입력 데이터를 처리하는 함수
        audio_data: numpy array from microphone input
        """
        audio = torch.from_numpy(audio_data).float()
        audio = audio.unsqueeze(0)  # [1, samples]로 차원 추가
        return audio


    def get_spec(self, wav=None, fpath=None):
        hps = self.hps
        
        if fpath is not None:
            audio = self.load_audio(fpath)
        else:
            audio = self.load_audio_from_mic(wav)
            
        with torch.no_grad():
            y = audio.to(self.device)
            y = self.spectrogram_torch(y, hps['data']['filter_length'],
                                        hps['data']['sampling_rate'], hps['data']['hop_length'], hps['data']['win_length'],
                                        center=False).to(self.device)
        return y
    
    
    def convert(self, src_spec, g_tgt):
        with torch.no_grad():
            audio = self.model(src_spec=src_spec, g_tgt=g_tgt).data.cpu().float().numpy()
        return audio, self.sampling_rate