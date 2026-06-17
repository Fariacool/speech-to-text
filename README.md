# FunASR Chinese Subtitle CLI

用 FunASR Paraformer 把中文长音频/视频转换成字幕。默认流水线是
`paraformer-zh + fsmn-vad + ct-punc`，可选说话人标注。

## 安装

一键安装 `uv`、`ffmpeg`、Python 环境和项目依赖：

```bash
./install.sh
```

如果 GPU 镜像需要指定 PyTorch CUDA wheel 源：

```bash
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./install.sh
```

CPU 测试环境：

```bash
./install.sh --cpu
```

## 基本使用

```bash
uv run --no-sync funasr-subtitle input.mp4 \
  --device cuda:0 \
  --hotword person_a \
  --hotword person_b
```

输出默认写到 `outputs/`：

- `*.srt`：给播放器或剪辑软件用
- `*.vtt`：给网页播放器用
- `*.txt`：快速阅读
- `*.segments.json`：标准化字幕片段
- `*.raw.json`：FunASR 原始输出，方便排查或二次处理

## 先本地抽音频再上传

可以直接把 MP3 传到 GPU 服务器上处理，不一定要上传很大的 MP4。脚本支持音频或视频输入；传入 MP3 后，脚本仍会先用 FFmpeg 转成 FunASR 更适合的 16 kHz 单声道 WAV 再识别。

你已经生成的高质量 MP3 可以直接使用：

```bash
ffmpeg -i "/path/to/source-video.mp4" \
  -q:a 0 \
  -map a \
  example-6h.mp3
```

服务器上运行：

```bash
uv run --no-sync funasr-subtitle example-6h.mp3 \
  --device cuda:0 \
  --hotword person_a \
  --hotword person_b \
  --hotword organization_a \
  --hotword organization_b
```

不过 ASR 不需要音乐级音质。后续更推荐在本地直接导出体积更小的 16 kHz 单声道 MP3，上传更快，服务器预处理也更少：

```bash
ffmpeg -i "/path/to/source-video.mp4" \
  -vn \
  -ac 1 \
  -ar 16000 \
  -c:a libmp3lame \
  -b:a 64k \
  example-6h-16k.mp3
```

如果上传体积不是问题，也可以直接生成 FunASR 最终会使用的 WAV，省掉服务器上的转码步骤：

```bash
ffmpeg -i "/path/to/source-video.mp4" \
  -vn \
  -ac 1 \
  -ar 16000 \
  -c:a pcm_s16le \
  example-6h-16k.wav
```

6 小时 WAV 通常会有几百 MB；综合看，`16k/64k mono MP3` 更适合临时上传到云 GPU。

## 6 小时访谈推荐命令

正式跑长任务前，建议先抽 5 分钟样本验证模型、字幕格式和说话人标签：

```bash
uv run --no-sync funasr-subtitle input.mp3 \
  --device cuda:0 \
  --spk \
  --sample-minutes 5 \
  --output-dir outputs-sample-spk \
  --prefix sample-spk
```

如果样本没问题，再跑完整音频。普通字幕可以不物理切分，让 FunASR 的 VAD 做内部长音频切分：

```bash
uv run --no-sync funasr-subtitle input.mp4 \
  --device cuda:0 \
  --hotword person_a \
  --hotword person_b \
  --hotword organization_a \
  --hotword organization_b
```

需要说话人标签时加 `--spk`：

```bash
uv run --no-sync funasr-subtitle input.mp4 \
  --device cuda:0 \
  --spk \
  --chunk-minutes 30 \
  --hotword person_a \
  --hotword person_b
```

长任务建议开启物理分块，这样日志能看到全局进度，且每个 chunk 完成后会更新 `*.partial.srt`、`*.partial.vtt`、`*.partial.txt`、`*.partial.segments.json`：

```bash
uv run --no-sync funasr-subtitle input.mp4 \
  --device cuda:0 \
  --chunk-minutes 45 \
  --batch-size-s 120 \
  --batch-threshold-s 30 \
  --max-single-segment-ms 30000 \
  --hotword person_a \
  --hotword person_b
```

后台运行可以用一行 `nohup`：

```bash
nohup bash -lc 'cd /path/to/speech-to-text && uv run --no-sync funasr-subtitle /path/to/input.mp3 --device cuda:0 --spk --chunk-minutes 30 --output-dir outputs-spk' > /path/to/speech-to-text/funasr-spk.log 2>&1 &
```

查看日志和中间结果：

```bash
tail -f /path/to/speech-to-text/funasr-spk.log
ls -lh /path/to/speech-to-text/outputs-spk/
```

## 说明

- 默认模型源是 Hugging Face：`--hub hf`，通常更适合海外服务器。
- 如果 ModelScope 更快，可以加 `--hub ms`。
- 使用 `--spk` 时，脚本会自动切到支持时间戳的 ModelScope Paraformer preset；说话人分离依赖句子时间戳。
- 长视频默认不需要先切分；脚本会先用 FFmpeg 抽取 16 kHz 单声道 WAV。
- 如果加了 `--chunk-minutes`，脚本会按块显示进度并持续写入 partial 输出。
- `--no-sync` 会让 uv 使用安装脚本准备好的 `.venv`，避免运行时重新解析依赖或覆盖手动安装的 PyTorch wheel。
