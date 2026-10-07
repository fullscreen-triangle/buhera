# Jobs on AppHub

AppHub is the university's Code-Server: machines with many CPUs and GPUs, reached through a browser. It accepts no connection from outside, so a job cannot be sent to it — but AppHub can pull from the university's Gitea. lattice uses that: it wraps a task of your repository into a **unit**, pushes it to Gitea, and you run the unit in an AppHub session with one command; the unit builds its own environment, runs its shards in parallel, and pushes its results back, where Buhera reads them.

**Time:** 20 minutes, plus however long the job runs.

**Before you start:**

- lattice installed on your computer: `cargo install --path lattice` in `pylon` (the server looks in `~/.cargo/bin`, or `LATTICE_CLI`).
- A repository with a remote on `git.uni-greifswald.de` (`git remote add uni https://git.uni-greifswald.de/<user>/<repo>.git`), and the task in `.vscode/tasks.json` or pixi.
- long-grass running on your computer — jobs work on repositories on the machine the server runs on, so the hosted site refuses them.

---

## 0. The path of a job

| Where | What happens |
|---|---|
| here | **tasks** → **plan** (what the task needs on AppHub; writes nothing) → **wrap** (writes `.lattice/<unit>/`, commits only that, pushes it to Gitea) |
| AppHub | you start a session; in its terminal: `git pull`, then `bash .lattice/<unit>/run.sh --detach` |
| here | **results** (what the session pushed back) → **log** of any shard → **get** (copy the outputs into your working tree) |

lattice cannot start an AppHub session; nothing outside AppHub can. Everything else is automatic.

## 1. Your repositories

```
apphub
```

Add a repository by its path on your computer. Each one gets **tasks** and **units & results**.

## 2. Plan

Click **tasks**, then a task. Say how to split it — `seed=1..3` runs it three times, filling `{seed}` in the command; **once per file** runs it per tracked file matching a glob; **send back** names the files to return — and press **plan it**. For the lipid series' scoring script, a task `python3 score.py {seed}` split over three seeds:

```text
unit         score — 3 shard(s) (seed×3)
from         task "score"
runs         ( cd "${LATTICE_ROOT}" && python3 score.py {seed} )
environment  venv in .; built on first run, kept in $HOME after
compute      CPU is enough: no GPU library found
profile      any CPU profile
parallel     one shard per CPU of the session
results      results/**, plus each shard's $LATTICE_OUT

ON APPHUB (CODE-SERVER → TERMINAL)
1. get it   git clone <the repository's Gitea URL>
2. run      cd <repo> && bash .lattice/score/run.sh --detach
```

lattice read only what git tracks — exactly what AppHub will get. It worked out the environment (here a virtualenv from `requirements.txt`), whether a GPU is needed and which line of code decided it, which AppHub profile to start, the models and secrets the code reads, and how many shards to run at once. Every verdict shows where it came from; `.lattice/<unit>/unit.toml` is there to correct one. If the repository has no remote on the university's Gitea, the plan says so and how to add one.

## 3. Wrap

**wrap it…** asks once more — *this commits `.lattice/score/` on your current branch, and nothing else, and pushes it to Gitea* — then **wrap and push**:

```text
lattice: wrote .lattice/score/
lattice: committed the unit as 37063f6cc4 (only its own files)
lattice: pushed to …; AppHub can pull it now
wrapped and pushed — AppHub can pull it now.
```

Your other staged and unstaged work is left as it was.

## 4. On AppHub

Start a session with the profile the plan named (CPU, or a GPU profile such as “Deep Learning → Advanced: RTX 4090”). In its terminal:

```text
git clone https://git.uni-greifswald.de/<user>/<repo>.git     # the first time; afterwards: cd <repo> && git pull
cd <repo> && bash .lattice/score/run.sh --detach
```

`--detach` keeps it running after the browser tab closes. If the unit needs a secret (the plan lists them), store it once on AppHub in `~/.lattice/secrets.env`; secrets are never committed. Pushing results needs your Gitea token in the session, saved the first time you clone (`git config --global credential.helper store`).

## 5. Results

Back here: **units & results** → **results**. From a run of this unit in a Linux session:

```text
every shard is done.
results at 82ebbd45ba (13s ago)
part 1/1 · done on d1dfb17ba78b · 3 shard(s), 3 at a time, 8 CPU, 0 GPU

shard    state   exit   took   host
seed-1   done    0      0s     d1dfb17ba78b   log
seed-2   done    0      0s     d1dfb17ba78b   log
seed-3   done    0      0s     d1dfb17ba78b   log
3 done, 0 failed, 0 running, 0 not reported, of 3
```

Before the session has pushed anything it says “nothing pushed back yet — is the unit running on AppHub?”; while it runs, **look again**. **log** shows a shard's output:

```text
shard seed-1
seed 1: mean chain length 34.122
```

**get the outputs…** copies the files the unit sent back into your working tree. Running `run.sh` again on AppHub only retries the shards that did not succeed.

## 6. The job belongs to the experiment

On the wrapped frame, or next to a unit, choose the plan the job belongs to — or add it on the plan item under **jobs on AppHub**. The item then lists the unit with a **results** button, and an experiment that was only an idea becomes “running”.

That closes the loop the tutorials walked: Mara's mail about the robot, the protocol in your notes, the plan with its evidence, and the scoring on AppHub — all on one screen, if you cut them onto it.
