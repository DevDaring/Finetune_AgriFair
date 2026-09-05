# Deployment

Running the study on a rented GPU, safely enough that an interruption costs minutes rather
than the run.

## Order of operations

```
1. bootstrap.sh                     prepare the VM, once per instance
2. run_verification.sh <tiers>      pre-flight on real weights, must pass before step 3
3. run_study.sh <tiers>             the study, with autopush already running
4. push_models_to_huggingface.py    adapters, once the study has finished
```

## Why the verification step is not optional

The smoke pipeline proves the plumbing on a 1M-parameter random model. It cannot prove what
only real weights can: that the gated checkpoints downloaded, that Gemma 3's multimodal
wrapper loads and its text tower is reachable, that a 12B model fits at the configured
micro-batch, that the pre-built FlashAttention wheel actually installed rather than silently
falling back, and that real outputs parse.

`check_verification.py` gates on exactly those, and exits non-zero if any is unproven. A
"completed without error" run is not evidence.

## Splitting across two instances

Wall clock halves at the same total cost, because the four models are independent until the
analysis stage.

```
instance A:  bash deploy/run_study.sh small-instruct,broad-instruct
instance B:  bash deploy/run_study.sh general-instruct,general-instruct-2
```

Both push to the same artifacts branch. Run the analysis stage once, after both finish, on
whichever instance has restored both sets of artifacts.

## Secrets

`.env` is gitignored and is never fetched by any script here. Copy it in separately:

```
scp -P <port> Codes/.env root@<host>:/workspace/Finetune_AgriFair/Codes/.env
```

`bootstrap.sh` refuses to continue without it, and `artifact_sync.py` injects the GitHub
token into the remote URL for the duration of a single push, never writing it to disk and
redacting anything token-shaped out of git output before it reaches a log.

## What runs in the background

| Script | Interval | Purpose |
|---|---|---|
| `autopush.sh` | 30 min | force-push results, checkpoints and data to the artifacts branch |
| `monitor.sh` | 60 min | progress, GPU state, parse-failure rates, stall and error detection |

`health_report.py` is the single check behind the monitor and can be run on its own at any
time. It exits non-zero when something needs attention, so it can drive an alert.

It reports progress rather than liveness on purpose: a process that is alive but has written
nothing for 45 minutes is the failure worth catching, and "still running" would hide it.

## Resumability

Nothing here has to be restarted from the beginning.

- Training skips any arm whose `final/` adapter and `train_summary.json` exist.
- An interrupted arm resumes from its last saved epoch's adapter weights.
- Evaluation reuses stored per-item predictions unless `FORCE_EVAL=1`.
- The hosted-model panel caches every API response per item, so a re-run pays nothing.
- `restore_artifacts.py` pulls a previous instance's state onto a fresh VM.

This is what makes interruptible instances the right choice: they cost 40 to 50 percent less
and the pipeline is built to survive preemption.
