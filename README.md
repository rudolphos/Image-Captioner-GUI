
A lightweight GUI tool for batch-captioning media library using local vision models. Drag your files in, let the LLM analyze them, and automatically save the results directly to the filenames or the metadata.

- Captions images (.jpg, .png) and videos (.mp4, .mkv, etc.).
- Extracts frames and provides them to the AI to describe temporal progression and action.
- Append keywords to existing file names or replace them entirely (in case filename contains randomness or gibberish). It automatically handles illegal characters and file path lengths.
- Can write captions into the metadata of JPEG (EXIF) and PNG files (ImageDescription/Comment tags).
- Uses concurrent threading and a preprocessing so GPU stays saturated while the CPU handles file I/O.
- Drag & Drop priority - No command line needed—just drop selected files onto the interface.

<img width="893" height="845" alt="Screen2" src="https://github.com/user-attachments/assets/8e9830b4-13cc-4952-8be8-4d0e2a7aa0d4" />

- Recommended Models & Prompts
For the best results with vision tasks, Qwen3-VL (or the latest Qwen-VL iterations) is highly recommended for its accuracy and keyword adherence.

Basic prompts for image captioning:
Keyword tagging: Output exactly 10 keywords describing the image. Comma separated. Stop after 10.
Fluent sentence-like brief caption: Write a descriptive caption within 15 words for this image.

For video analysis there is one basic prompt inside the script already that you can change to your own liking: 
Describe the cause and effect sequence observed across these frames.

Tutorial:
- Launch your backend: LM Studio or your preferred API that can host vision models and load a it.
- Ensure the API URL in the app matches your server (default is http://127.0.0.1:1234/v1/chat/completions).
- Choose whether you want to rename your files, write to metadata, or both or simply preview what it generates ("None" radio button)
- Drag your images or videos onto the "Drag & drop files here" zone.

Requirements
Python 3.10+
pip install opencv-python Pillow piexif requests tkinterdnd2
