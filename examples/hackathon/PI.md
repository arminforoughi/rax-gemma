# RAX + Gemma on a Raspberry Pi 5

Everything on one board: the SO-101 and its wrist camera plug into the Pi, and Gemma 4
E2B runs on the Pi's CPU through LiteRT-LM (Google AI Edge). No laptop and no network
are needed while it runs.

Google's own figures for Gemma 4 E2B on a Pi 5 with LiteRT-LM: about 99 tokens/s reading
the prompt, about 9 tokens/s writing, about 1.4 GB of memory. Gemma is asked once to
look, once to plan and once per placed tube; the arm's own control loop never waits on
it.

## Hardware

- Raspberry Pi 5, 8 GB, with the official 27 W supply and an active cooler (the CPU runs
  flat out during each Gemma answer)
- Raspberry Pi OS 64-bit (Bookworm or later)
- SO-101 bus servo adapter → Pi USB; OAK-D wrist camera → Pi USB 3 (blue port)
- A speaker or headphones, for `--speak`

## Install

```bash
sudo apt install -y python3-venv espeak-ng
python3 -m venv ~/rax-env && source ~/rax-env/bin/activate
git clone https://github.com/arminforoughi/rax-gemma.git && cd rax-gemma
pip install -e ".[robot]"
pip install litert-lm-api "huggingface_hub[cli]"
```

Get the model (2-3 GB) from the LiteRT community on Hugging Face. Accept the Gemma licence
on the model page first and log in with `hf auth login`:

```bash
hf download litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm --local-dir ~/models
```

## Check the model on its own

```bash
python -c "from rax.gemma import LiteRtGemma; g = LiteRtGemma(); print(g.warm()); print(g.ask('Say hi in three words.'), g.last_s)"
```

## Run

```bash
python examples/hackathon/gemma_lab.py --pi --port /dev/ttyACM0 --places places.json \
    --hint "sample tubes with coloured caps lying on a black mat" --speak
```

`--pi` uses LiteRT-LM with `~/models/gemma-4-E2B-it.litertlm` and finds the tubes by cap
colour (YOLO-World is too slow on the Pi's CPU). `--model` points at another
`.litertlm` file, for example Gemma 4 E4B if it is fast enough for you.

Before any motion, type `look` and `plan <request>`: these use only the camera, so you
can check what Gemma sees on the Pi before the arm moves.

## What changes from the laptop

| | Laptop | Pi 5 |
|---|---|---|
| Runtime | Ollama (a local server) | LiteRT-LM (in the process) |
| Model | Gemma 4 E4B | Gemma 4 E2B |
| Finding tubes | YOLO-World or colour | colour |
| Speech | macOS `say` | `espeak-ng` |

E2B is the smaller model, so it misnames things more often. In colour mode, what the cap
finder sees is added to Gemma's list, which covers for that.
