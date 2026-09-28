# FlyBench — dumbflies.py as a research bench

![pic](pic.png)

The flies in `dumbflies.py` are a system whose rules we wrote and then forgot.
That makes them a rare thing: a black box with the answer key in the drawer.
This bench runs the **original classes unchanged**, watches the flies from outside,
and asks what an observer can recover — then checks against the code.

## What's here

| file | what it does |
|---|---|
| `flybench.py` | Loads `BugConfig / ThinkingField / EnhancedBug` straight from `dumbflies.py` (exec, untouched). Replaces the webcam with reproducible synthetic scenes and the Tk loop with a headless copy of `BugGUI.process_frame`. Logs observables (x, y, heading) and hidden truth (energy, decision-field centroid, pattern count). |
| `analyze.py` | Four behavioural tests + observer models (gradient boosting) that predict turning and speed from observables only; train on one run, test on another. |
| `ablate.py` | Splits the observer's score into "momentum", "where it is", "where it has been". |
| `fig.py` | `flybench_findings.png` |
| `results.json`, `ablation.json` | the numbers |

Scenes: `room` (textured room, bright window, dark "person" who walks in, sits, leaves on a 900-frame loop), `room_static` (same, nobody), `gray` (flat 128), `black` (all zeros). 3000 frames, 2 seeds each.

Speed patches (the original per-pixel Python loops, and `cv2.bitwise_and` with a mask which costs 40 ms/call on the test machine) were replaced with vectorized versions; `check_equivalence()` and `check_cone_equivalence()` verify they match the originals (max diff 2e-16; vision cone pixel-identical).

## What the flies actually are

The fly's "vision" is the whole camera frame masked to its cone and then **resized to 32×32**. So the 32×32 fields are not an eye's view — they are a **map of the entire screen** with a tiny cone stamped at the fly's absolute position. Perception → memory → decision each dilate and smear that stamp, so the decision field ends up as a **blurred trail of where the fly has been**. The fly then steers toward "trail centroid minus screen centre", scaled by its distance from centre. The input is also min-max normalized per frame, which erases brightness.

## Results

| test | prediction | result |
|---|---|---|
| Does the camera change movement? | no | **Confirmed.** Room vs flat gray vs static room differ no more than two seeds of the same scene (KS 0.02–0.03 vs 0.01–0.035 seed-to-seed). An observer trained on flat gray predicts flies in the room exactly as well (turn R² 0.82, speed 0.99). |
| Does a walking person attract them? | no | **Confirmed.** 9.9% of flies within 250 px of the person vs 10.4% at the same spot and frames with nobody there. |
| What feeds them? | their own motion (cone shifting counts as "motion") | **Wrong.** Intake is ~0.02/frame whether still or moving (r = 0.10), in every scene. Burn is 0.05/frame, so everyone starves on a timer; births come from the mating lottery. |
| In the dark? | stop and face right | **Half wrong.** In pure black they don't stop — they move *faster* (median step 9.8 px vs 5). Other flies are drawn as white dots into each fly's view, so the field holds 2–3 cells at other flies' positions, whose centroid is far from centre → high speed. Heading bias to the right is present but weak (resultant 0.35 at +3°). All die at exactly frame 2000 = 100 / 0.05: zero intake. |

### What an observer can predict (train seed 0, test seed 1, R²)

| features | turning | speed |
|---|---|---|
| camera (brightness + change ahead of the fly) | 0.00 | 0.01 |
| where it is now | 0.02 | 0.07 |
| where it has been (position EMAs, no momentum) | 0.15 | 0.54 |
| its own last two moves (momentum) | 0.82 | 0.99 |
| all observables | 0.82 | 0.99 |
| hidden decision-field centroid (oracle) | 0.90 | 0.99 |

Read honestly: most of the predictability is inertia (momentum 0.5 in the code). The rule underneath shows up as *history beats present*: where the fly has been explains 54% of its speed; where it is now explains 7%; the camera explains nothing. The remaining gap to the oracle on turning (0.82 → 0.90) is what a better observer would need to find: the field's spatial trail and the unwrapped heading the code steers with.

## Why this is the useful bench

- It's a real mixed system (fields, thresholds, momentum, social drawing, a lottery) with **complete ground truth**.
- Every "rule-recovery" method in the Genealogy corpus (probe tomography, law extraction, SilentPing-style reading of hidden state) can be scored here against the answer key.
- It already exposes the classic trap: a flexible model scores 0.99 by learning inertia, while telling you almost nothing about the mechanism. Ablation is what found the trail.

## Next

1. **Harder observer.** Give it only positions and the video, ask for the rule as a sentence or a small symbolic model, then score it against the code (does it name "trail centroid vs screen centre"?).
2. **Fix the eye, then re-run the bench.** Make the cone a real egocentric crop, so the camera can matter. The bench becomes the test: does the camera-only R² rise above zero?
3. **Windowed brain.** Swap `ThinkingField` for a time-window brain, same bench, same observers. Does it do anything the observer can detect that the old flies couldn't?

## Reproduce

```
python flybench.py room,gray,room_static,black 3000 0,1   # ~10 min on 2 cores
python analyze.py && python ablate.py && python fig.py
```
Needs numpy, scipy, opencv-python, scikit-learn, matplotlib. Put `dumbflies.py` in this folder or the folder above, or point to it with `set DUMBFLIES=C:\path\to\dumbflies.py`.

---

# Part 2 — Resonator flies (next steps 2 and 3, done)

`resonator_brain.py` swaps only the brain. Body, movement, repulsion, energy decay, mating and inheritance are the original code.

- **Real eye.** 9 points ahead of the fly (3 distances × 3 angles), sampled from the camera. No more whole-screen map.
- **Windowed resonator per eye point:** `z ← r·e^{iω}·z + (1−r)·d`. The tuning ω and window r are genes, inherited and mutated by the original `inherit_traits`.
- **Hop-and-listen.** Eyes are shut during a 10-frame hop. On landing the fly stands still for ≥ 1.5/(1−r) frames, cancelling the body's own wander turn (an efference copy). It stays and eats while it hears its rate; otherwise it hops again. This was needed: flying over plain room texture registers *more* activity (9–12) than a mistuned flicker (4–7), so a fly that listened in flight could live on its own motion.
- **Control brain `window`.** Identical eye, window and behaviour, but the unit is `e ← r·e + (1−r)·|d|` (flicker energy, no tuning). ω is then an unused gene.

Scene `rhythm`: the room plus two flickering patches, A at 0.05 cycles/frame (1.5 Hz at 30 fps) and B at 0.15 cycles/frame (4.5 Hz).

![resonator results](resonator_findings.png)

## Ledger

| claim | test | verdict |
|---|---|---|
| Flies go to the rhythm they're tuned to | No breeding (no families), 120 fresh random flies, 3 seeds, 2000 frames | **Holds.** Tuned to A: 52% of life at A, 7% at B. Tuned to B: 29% at B, 14% at A. Untuned: 15% / 18%. Tuning predicts place preference, Spearman ρ = 0.42, p = 1.5e-6. |
| …and it's the tuning, not the window | Same test, control brain | **Holds.** Control flies sit at both patches (30–40% each) with no link to ω (ρ = −0.09, p = 0.31). The control is *better* at finding flicker in general; the resonator trades that for selectivity. |
| The camera now matters | Observer predicts stop/go from video only, train seed 0 → test seed 1 | **Holds, with a twist.** Original flies 0.04, resonator 0.17, window control 0.23. The untuned control is the most camera-predictable, because it reacts to any flicker the same way. Turning is still ~0 for all brains. |
| Behaviour = camera × who the fly is | Observer given the fly's ω and r as well | **Inconclusive.** The score *dropped* for both brains (0.17 → 0.12, 0.23 → 0.12). With breeding, ω labels families; the model memorised seed-0 families that seed 1 doesn't have. The no-breeding test above is the evidence instead. |
| The population evolves toward the room's rhythms | Share of flies tuned to either rate, start vs last 1000 frames | **Not shown.** It rises in the rhythm room (42%→67%, 62%→83%), but it also rises in a room with *no* rhythm (41%→66%). That's drift in small bottlenecked populations; the window control goes 38%→30% and 60%→61%. It needs many seeds and bigger populations before it's a claim. |
| (confound found on the way) | Sorting measured *with* breeding | Even the control brain "sorts" (B-tuned at B 23% vs 8% at A) because children are born beside parents and inherit ω. Any sorting number from a breeding run is lineage-confounded. |

## Files

| file | what |
|---|---|
| `resonator_brain.py` | the two brains (`make_bug_class(EnhancedBug, "resonator" \| "window")`) |
| `resonator_flies.py` | **live webcam**: runs the original dumbflies GUI with the new brain. Rings show each fly's tuning (blue slow → red fast, filled = listening and hearing); top-left strip is the population's tuning histogram in Hz. `--brain window` for the control, `--floor 8` if a noisy camera feeds everyone. |
| `sort_nobreed.py` | the clean sorting test |
| `res_eval.py`, `cam_identity.py`, `res_fig.py` | scoring and figure; numbers in `res_results.json`, `sort_nobreed.json` |

```
python sort_nobreed.py                                        # ~10 min, the key result
python flybench.py rhythm 6000 0,1 resonator                  # long runs (also: window; original at 3000)
python flybench.py room 6000 0 resonator                      # evolution control
python res_eval.py && python res_fig.py
python resonator_flies.py                                     # live, needs dumbflies.py + webcam
```
The live launcher was smoke-tested here with stubbed Tk (brain swap, inheritance, drawing), not with a real webcam.
