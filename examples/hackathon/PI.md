# RAX + Gemma on a Raspberry Pi 5

Everything on one board: the SO-101 and its wrist camera plug into the Pi, and Gemma 4
E2B runs on the Pi's CPU through LiteRT-LM (Google AI Edge). No laptop and no network
are needed while it runs.

Google's own figures for Gemma 4 E2B on a Pi 5 with LiteRT-LM: about 99 tokens/s reading
the prompt, about 9 tokens/s writing, about 1.4 GB of memory. Gemma is asked once to
look, once to plan and once per placed tube; the arm's own control loop never waits on
it.

## Hardware

- Raspberry Pi 5 (8 GB; 4 GB also runs E2B, which uses about 0.6 GB), with the official
  27 W supply and an active cooler (the CPU runs flat out during each Gemma answer)
- Raspberry Pi OS 64-bit (Bookworm or later)
- SO-101 bus servo adapter → Pi USB; OAK-D wrist camera → Pi USB 3 (blue port), or a
  powered USB hub
- A speaker or headphones, for `--speak`

The supply matters more than anything else here. A phone or laptop charger keeps the
Pi up while idle, then browns it out as soon as Gemma, the camera and the arm run
together; the Pi cuts out with no error, and a cut mid-write can kill the SD card.
Check it before anything else, under load:

```bash
vcgencmd get_throttled             # 0x0 is good; bit 0 (0x1) or 16 (0x10000) = undervoltage
vcgencmd get_config usb_max_current_enable   # 1 on a supply the Pi recognises as 5 A
```

## Install

```bash
sudo apt install -y python3-venv espeak-ng
python3 -m venv ~/rax-env && source ~/rax-env/bin/activate
git clone https://github.com/arminforoughi/rax-gemma.git && cd rax-gemma
export TMPDIR=~/tmp && mkdir -p $TMPDIR    # /tmp is in RAM on Pi OS; torch does not fit
pip install -e ".[robot]" --extra-index-url https://download.pytorch.org/whl/cpu
pip install litert-lm-api "huggingface_hub[cli]"
```

The extra index matters: LeRobot pins an older torch, and without it pip pulls PyPI's
aarch64 CUDA build and several GB of NVIDIA libraries a Pi cannot use. With it, pip
takes the `+cpu` build.

Let a normal user open the OAK-D (once; Luxonis' udev rule):

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Copy the arm's calibration from the machine that calibrated it, or LeRobot asks for a
new one on the first connect (and a server with no terminal dies on that prompt):

```bash
scp ~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json \
    pi:.cache/huggingface/lerobot/calibration/robots/so_follower/
```

Stock LeRobot has no OAK-D camera; `rax.robots.oakd` provides it, so the RAX LeRobot
fork is not needed on the Pi.

Get the model (2-3 GB) from the LiteRT community on Hugging Face. If the download is
refused, accept the Gemma licence on the model page and log in with `hf auth login`:

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

### The chat page

The RAX UI with its Gemma chat page, served by the Pi:

```bash
pip install ultralytics --extra-index-url https://download.pytorch.org/whl/cpu   # YOLO-World
RAX_GEMMA_BACKEND=litert RAX_GEMMA_VISION=0 RAX_GEMMA_DETECTOR=prompt \
    python examples/server.py --arm so101 --no-tubes \
    --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_<serial>-if00
```

Open `http://<pi>:8484/ui/chat.html` (`/chat` itself is the page's JSON endpoint).

- `RAX_GEMMA_VISION=0`: Gemma answers text-only. YOLO-World, prompted with the pick
  page's query ("red cube, green cube"), says what is on the table, so set the query to
  what you have. Looking at the picture is most of E2B's time on a Pi.
- `RAX_GEMMA_DETECTOR=prompt`: picks find their object with YOLO-World (about 2 s a
  frame on a Pi 5; the tracker holds it in between). Without ultralytics, the colour a
  label names is used instead.
- Use the `/dev/serial/by-id/` path: `/dev/ttyACM0` becomes `ttyACM1` when the arm is
  replugged.

A question or small talk ("what do you see?", "hello") never moves the arm, whatever
Gemma plans; E2B text-only plans moves for both. A request ("put the red cube in the
left rack", "red left") does.

## What changes from the laptop

| | Laptop | Pi 5 |
|---|---|---|
| Runtime | Ollama (a local server) | LiteRT-LM (in the process) |
| Model | Gemma 4 E4B | Gemma 4 E2B |
| Finding tubes | YOLO-World or colour | colour |
| Speech | macOS `say` | `espeak-ng` |

E2B is the smaller model, so it misnames things more often. In colour mode, what the cap
finder sees is added to Gemma's list, which covers for that.
