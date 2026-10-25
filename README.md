# Offline AI (v7)

A local AI assistant for Windows. Runs a 9B model on your own computer.
Everything stays on your machine. The internet is only used when a question
needs something recent (news, prices, who holds a position today).

Built in spare time over two months by directing a Claude agent.
Project page: https://axelpfannenstill.lovable.app/

## Download and run

1. Download the app (zip, 307 MB):https://drive.google.com/file/d/1s8foxfzpr7fiEjVWyCb1fvBrZ7bttkKG/view?usp=sharing , then unzip it anywhere, for example C:\OfflineAI\
2. Download the language model (5.3 GB) from Hugging Face:
   https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf
3. Put the file in the `models\` folder next to the exe. The folder should look like this:

```
OfflineAI\
  AIPrototype_v7.exe
  README.md
  models\
    Qwen3.5-9B-Q4_K_M.gguf        <- you download this one
    bge-small-en-v1.5-f16.gguf    <- included
```

4. Double-click `AIPrototype_v7.exe`. The first start takes 20 to 40 seconds while the model loads.

## Requirements

- Windows 10 or 11, 64-bit
- 16 GB RAM
- A GPU with 6 GB VRAM or more (NVIDIA, AMD or Intel, via Vulkan). Without a GPU it runs on the CPU, slowly.
- 6 GB free disk space

## Windows warning on first start

The exe is not code-signed, so Windows SmartScreen shows "Windows protected your PC".
Click **More info** and then **Run anyway**. This is expected for small unsigned programs.

## What it does

- Answers questions, explains things, writes and fixes code, translates
- Looks up recent facts on the web when needed (Brave Search, Wikipedia, Wikidata)
- Says so when it cannot verify an answer, instead of guessing
- 94% correct on a 300 question test in 14 languages

## Web search

Web search works out of the box. If it stops working, get a free key from
https://api-dashboard.search.brave.com/ and set it as an environment variable:
`setx BRAVE_API_KEY "your-key"`, then restart the app.

## Source

The Python source is in the GitHub repository. Main file: `AIPrototype.py`.
