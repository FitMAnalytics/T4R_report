# Converting a Jupyter Notebook to an Executable .py Script

## Step 1: Export the notebook to .py

```bash
jupyter nbconvert --to script optuna_hpsearch.ipynb --output optuna_hpsearch
```

This produces `optuna_hpsearch.py` with all the cell contents. Markdown cells become `# comments`.

## Step 2: Clean up the .py file

The exported script will have some artifacts you'll want to tidy:

- Remove `# In[...]` cell markers that nbconvert inserts
- Remove any `display()` or notebook-specific calls
- The `matplotlib` visualization cell — either drop it or add `plt.savefig("optuna_plots.png")` instead of `plt.show()`, since there's no GUI on a scheduled job

## Step 3: Redirect all output to a log file

Two options:

### Option A — at the shell level (easiest):

```bash
python optuna_hpsearch.py 2>&1 | tee optuna_run.log
```

This captures both stdout and stderr to a file while still printing to terminal.

### Option B — inside the script (more control):

Add this near the top of the .py file, after imports:

```python
import sys

time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
log_file = open(f"optuna_run_{time_stamp}.log", "w")

class TeeOutput:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            s.write(msg)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = TeeOutput(sys.stdout, log_file)
sys.stderr = TeeOutput(sys.stderr, log_file)
```

Every `print()` then goes to both the terminal and the log file automatically.

## Step 4: Make hyperparameters easy to change

The exported script will already have the search ranges as plain variables at the top (from the config cell). You can edit them directly. If you want to also accept command-line overrides:

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n-trials", type=int, default=200)
parser.add_argument("--num-epochs-max", type=int, default=20)
args = parser.parse_args()

N_TRIALS = args.n_trials
```

Then run: `python optuna_hpsearch.py --n-trials 50`

## Summary

```bash
jupyter nbconvert --to script optuna_hpsearch.ipynb --output optuna_hpsearch
# edit the .py: remove notebook artifacts, add logging, save plots to file
python optuna_hpsearch.py 2>&1 | tee optuna_run.log
```

The SQLite storage (`xlnet_hpsearch_{timestamp}.db`) already persists the study, so even if the job dies and restarts, `load_if_exists=True` picks up where it left off.
