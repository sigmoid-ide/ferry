# Ferry

**Run your Hunch experiments on a GPU machine, and bring every run home.**

```bash
ferry add mybox user@gpu-box
ferry run --on mybox -m "bigger batch on the 3090" -- python train.py
```

Ferry sends your code to the machine, builds the environment there, starts the run so it survives a dropped
connection, and copies it back into `.hunch/` while it trains. `hunch log` and the PyCharm and VS Code plugins show
it like any other run.

Status: 0.1, for your own machines over SSH. Part of [Sigmoid IDE](https://github.com/sigmoid-ide).

## Install

```bash
pip install sigmoid-ferry
```

That gives you the `ferry` command, with Hunch. It also installs as a tool of its own: `uv tool install sigmoid-ferry`.

## What the machine needs

Linux (x86_64 or aarch64), SSH and `python3`. Nothing is installed system-wide: Ferry keeps everything in `~/.ferry`,
including [uv](https://github.com/astral-sh/uv), which it downloads once and checks against a pinned checksum.

## Running

`ferry add` shows the machine's host key the first time and pins it; a changed key stops Ferry until you remove the
old one. Ferry uses your SSH setup as it is: `~/.ssh/config` (with `ProxyJump` for a machine behind a jump host),
your keys and ssh-agent. A key with a passphrase is asked for in the terminal.

`ferry run` records the run on your computer first (code, git branch, the run it builds on), then:

- builds the environment on the machine from `requirements.txt`, or else from the packages in the project's own
  Python (an activated environment, or `.venv`), with torch matched to the machine's driver. Each environment is
  built once and reused;
- sends only the code files the machine does not have yet;
- starts the run in the same project folder you started it from, so relative paths work;
- copies the run home while it trains, and checks every file when it ends.

## Tokens and GPUs

Give a run what it needs from your environment with `--env`, and pick its GPUs with `--gpus`:

```bash
ferry run --on mybox --env HF_TOKEN --env WANDB_MODE=offline --gpus 1 -- python train.py
```

`--env NAME` sends the value `NAME` has here, so a token is never typed out or left in your shell history. It is
handed to the run in memory and never saved in a file, on the machine or in `.hunch/`. `--gpus` takes the numbers
`nvidia-smi` shows. Ferry refuses a GPU the machine does not have, which CUDA would answer by quietly running on the
CPU, and says when another program already uses a GPU the run gets.

## Data and outputs

Only your code goes to the machine on its own. Send data with `--data`:

```bash
ferry run --on mybox --data data/ -- python train.py
```

Data goes into a folder the project keeps on the machine and appears in every run's folder at the same path, so
`open("data/train.bin")` works there as it does here. It is sent once: next time only new or changed files go.

What a run writes (checkpoints, plots) stays on the machine. Ferry lists it when the run ends; bring it home with:

```bash
ferry pull a7f3                       # what run a7f3 wrote
ferry pull a7f3 checkpoints --to out/
```

## When the connection drops

The run keeps going. Ctrl+C stops copying, never the run. `ferry attach a7f3` picks the copy up again,
`ferry stop a7f3` stops the run (and everything it started; after 30 seconds it is killed), and `ferry ps` asks each
machine how its runs are doing. A run whose process on the machine dies without an end is marked crashed. If a
machine is gone for good, `ferry forget a7f3` stops counting its run as running, and Hunch shows it as lost; should the
machine come back, `ferry attach a7f3` picks the run up again.

## Renting a GPU on Vast.ai

No GPU machine of your own? Rent one with your own [Vast.ai](https://vast.ai) account. Ferry talks to Vast.ai
directly, and you pay Vast.ai; nothing goes through anyone else.

```bash
ferry login vast                        # once: your API key, kept in the OS keychain
ferry gpus --gpu "RTX 4090"             # what you can rent now, cheapest first
ferry run --rent "RTX 4090" --for 2h -m "bigger batch" -- python train.py
```

`--rent` takes an offer from `ferry gpus`, or a GPU name for its cheapest offer. Ferry says what it picked and the
most it can cost, and asks first. `--for` is required: the machine destroys itself when that time is up, even with
your laptop shut. Once the run is home, Ferry gives the machine back, and the run records what it cost
(`cost.json`).

To keep a machine for several runs, rent it on its own and give it back when you are done:

```bash
ferry rent "RTX 4090" --for 3h          # adds it as a machine, such as vast-12345
ferry run --on vast-12345 -- python train.py
ferry release vast-12345
```

`ferry machines` shows each rented machine's time left and its cost so far.

- Only hosts Vast.ai has verified, unless you ask with `ferry gpus --any-host`. Like any host, a host can see the
  code and data you send it.
- Your Vast.ai key stays on your computer, in the keychain or in `VAST_API_KEY`. The machine gets an SSH key of its
  own and Vast.ai's key for that one machine, which can do nothing else; it uses that key to destroy itself on time.
- When a run has ended and no laptop has brought it home for 15 minutes, the machine stops itself: the GPU stops
  costing, and only its disk does (40 GB costs about $0.30 to $0.90 a day, depending on the host). `ferry attach`
  starts it again and brings the run home; `ferry release` gives it back.
- A machine whose SSH port does not answer within 15 seconds goes back at once, and Ferry takes the next offer.

## Disk space

```bash
ferry clean mybox             # environments no run has asked for in 14 days
ferry clean mybox --outputs   # and what this project's ended runs wrote there, except runs you marked keep
```

Both say what they would remove and how much space that frees, and ask first. An environment is built again when a run
needs it.

## What stays where

Nothing secret goes into `.hunch/` or to the machine: no keys, no passwords. Tokens passed with `--env` exist only in
the run's memory. Your Vast.ai key is in the OS keychain. Ferry's own settings (the machines
you added and their host keys, and each rented machine's own SSH key until it goes back) are in `%APPDATA%\ferry` on
Windows and `~/.config/ferry` elsewhere.
