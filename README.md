# 即時多語言語音辨識與語者標記

使用 Openai Whisper 模型與 Diart 即時語者分段標記模型(Speaker diarization)，實作即時多語言語音辨識。
We use the Openai's Whisper model and Diart real-time speaker diarization model to implement the real-time ASR and speaker diarization.

# Demo

![demo](rt_asr_sd_demo.gif)

## 功能
此專案的功能包括：

- 實時多語言語音識別：透過 whisper 庫，能夠自動辨使語言即時識別出該語言的語音。
- 語者區分：透過 diart 庫，能夠區分出語音中的不同語者。

## 安裝方式
### 安裝 Diart 
[[diart installation]](https://github.com/juanmc2005/diart#-installation)
  1. 建立環境
  ```
  conda create -n diart python=3.8
  conda activate diart
  ```
  2. 安裝 audio libraries
  ```
  conda install portaudio pysoundfile ffmpeg -c conda-forge
  ```
  3. 安裝 diart (v0.7)
  ```
  pip install diart==0.7
  ```
### 登入 pyannote 取得 access token
[[get access to pyannote]](https://github.com/juanmc2005/diart#get-access-to--pyannote-models)  
Diart 是基於儲存在 huggingface hub 中的 pyannote.audio 模型。為了讓 diart 能使用這些模型，你需要按照以下步驟操作：
  1. [接受用戶條款](https://huggingface.co/pyannote/segmentation)以使用 pyannote/segmentation 模型 
  2. [接受用戶條款](https://huggingface.co/pyannote/embedding)以使用 pyannote/embedding 模型
  3. 安裝 [huggingface-cli](https://huggingface.co/docs/huggingface_hub/quick-start#install-the-hub-library) 並用你的用戶訪問令牌(access token) [登錄](https://huggingface.co/docs/huggingface_hub/quick-start#login)（或者在 diart 的命令列介面或 API 中手動提供）。

### 安裝 Whisper
[[whisper setup]](https://github.com/openai/whisper#setup)
```
pip install git+https://github.com/openai/whisper.git
```

### 下載程式執行

執行程式前請先確保麥克風設定能夠被存取，中斷程式請輸入`ctrl+C`。
```
python real-time_asr_spkd.py
```



