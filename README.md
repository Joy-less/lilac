# Lilac
Lilac is a ZeroShot VC Client software that uses AI to perform real-time voice conversion into a single voice file for more than 5 seconds without additional learning.


# Lilac's Features
- ### Run in a cpu environment
    It can also be used in home PCs without GPUs.

- ### No model training required
    Only voice of more than 5 seconds is required for real-time conversion.


# Pre-requisites
- A Windows/Mac/Linux system with a minimum of 4GB RAM.
- Anaconda installed.
- PyTorch installed.


# Installation
```
conda create -n lilac python=3.9
conda activate lilac
git clone https://github.com/kdrkdrkdr/lilac.git
pip install -r requirements.txt
```

# Run
```
python main.py
```


# What's New!
- v1.0.0
    - new feature:
        - Openvoice VC Code Optimization
        - VAD with threshold
    - bugfix:
        - Mitigate sound breakage during chunk conversion


# Future Plan
Once the feature is stabilized, we will deploy it as Standalone.


# Disclaimer 
We are not responsible for any direct, indirect, ripple, consequential or special damages caused by the use or unavailability of this Software.


# Reference
[myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice)  
[w-okada/voice-changer](https://github.com/w-okada/voice-changer)
