# Image-Captioner-GUI

A lightweight GUI tool for batch-captioning images and videos using local vision models. 
Drag your files in, let the LLM analyze them, and it automatically saves the results directly to the filenames or the metadata.

## Features
- Captions images (.jpg, .png) and videos (.mp4, .mkv, etc.).
- Extracts video frames and sends them to the AI to describe temporal progression and action.
- Append keywords to existing file names or replace entirely (in case filename contains randomness or gibberish). It automatically handles illegal characters and file path lengths.
- Can write captions into the metadata of JPEG (EXIF) and PNG files (ImageDescription/Comment tags).
- Uses concurrent threading and a preprocessing of the next image so GPU stays saturated while the CPU handles file I/O.
- Drag & Drop priority - No CLI needed — just drop selected files onto the GUI.

## Screenshot
<img width="893" height="845" alt="Screen2" src="https://github.com/user-attachments/assets/8e9830b4-13cc-4952-8be8-4d0e2a7aa0d4" />

## Recommended models & Prompts
For the best results, Qwen3-VL or the latest Qwen3.5 is recommended, but you can choose any with visual-support in LM studio.

### Basic prompts for image captioning:
Keyword tagging: 
```Output exactly 10 keywords describing the image. Comma separated. Stop after 10.```

Fluent sentence-like brief caption: ```Write a descriptive caption within 15 words for this image.```

For video analysis there is one basic prompt inside the script already that you can change to your own liking for the best result: 
```Describe the cause and effect sequence observed across these frames.```

## Usage
- Launch your backend: LM Studio or your preferred API that can host vision models and load a it.
- Ensure the API URL in the app matches your server (default is http://127.0.0.1:1234/v1/chat/completions).
- Choose whether you want to rename your files, write to metadata, or both or simply preview what it generates ("None" radio button)
- Drag your images or videos onto the "Drag & drop files here" zone.

## Requirements
Python 3.10+
```bash
pip install opencv-python Pillow piexif requests tkinterdnd2-universal numpy
```
