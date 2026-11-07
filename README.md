# Ferry

**Run your Hunch experiments on a GPU machine, and bring every run home.**

```bash
ferry add mybox user@gpu-box
ferry run --on mybox -m "bigger batch on the 3090" -- python train.py
```

Ferry sends your code to the machine, builds the environment there, starts the run so it survives a dropped
connection, and copies it back into `.hunch/` while it trains. `hunch log` and the PyCharm and VS Code plugins show
it like any other run.

Status: early development (v0: your own machines over SSH). Part of [Sigmoid IDE](https://github.com/sigmoid-ide).

## What the machine needs

Linux (x86_64 or aarch64), SSH and `python3`. Nothing is installed system-wide: Ferry keeps everything in `~/.ferry`,
including [uv](https://github.com/astral-sh/uv), which it downloads once and checks against a pinned checksum.

## Running

`ferry add` shows the machine's host key the first time and pins it; a changed key stops Ferry until you remove the
old one. Ferry uses your SSH setup as it is: `~/.ssh/config`, your keys and ssh-agent. A key with a passphrase is
asked for in the terminal.

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

## Disk space

```bash
ferry clean mybox             # environments no run has asked for in 14 days
ferry clean mybox --outputs   # and what this project's ended runs wrote there, except runs you marked keep
```

Both say what they would remove and how much space that frees, and ask first. An environment is built again when a run
needs it.

## What stays where

Nothing secret goes into `.hunch/` or to the machine: no keys, no passwords. Tokens passed with `--env` exist only in
the run's memory. Ferry's own settings (the machines
you added and their host keys) are in `%APPDATA%\ferry` on Windows and `~/.config/ferry` elsewhere.
