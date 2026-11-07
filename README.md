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
