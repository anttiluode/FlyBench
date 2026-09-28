# FlyBench — dumbflies.py as a research bench

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

> **Bug found and fixed Sep 28.** In the headless bench, children were built by the original `try_mate`, which looks up the class name `EnhancedBug`, so every child of a resonator or window fly got the *original* ThinkingField brain (carrying a tuning gene it didn't use). `flybench.run` now points that name at the brain being run. The live launchers were never affected, and neither was the no-breeding test. All breeding runs were redone; the corrected lines below are marked. Numbers in `rescored_fixed.json`.

| claim | test | verdict |
|---|---|---|
| Flies go to the rhythm they're tuned to | No breeding (no families), 120 fresh random flies, 3 seeds, 2000 frames | **Holds.** Tuned to A: 52% of life at A, 7% at B. Tuned to B: 29% at B, 14% at A. Untuned: 15% / 18%. Tuning predicts place preference, Spearman ρ = 0.42, p = 1.5e-6. |
| …and it's the tuning, not the window | Same test, control brain | **Holds.** Control flies sit at both patches (30–40% each) with no link to ω (ρ = −0.09, p = 0.31). The control is *better* at finding flicker in general; the resonator trades that for selectivity. |
| The camera now matters | Observer predicts stop/go from video only, train seed 0 → test seed 1 | **Holds.** Original flies 0.04, resonator 0.16, window control 0.09. Turning is still ~0 for all brains. *(Corrected Sep 28: the first version said the window control scored 0.23 and was "the most camera-predictable". That run had original-brain children; see the bug note below.)* |
| Behaviour = camera × who the fly is | Observer given the fly's ω and r as well | **Withdrawn, not rerun.** The first result came from runs with original-brain children. The no-breeding test above is the evidence instead. |
| The population evolves toward the room's rhythms | Share of flies tuned to either rate, start vs last 1000 frames | **Supported, 2 seeds.** Rhythm room 44%→100% and 67%→98%. Room with *no* rhythm 38%→35%. Window control (ω unused) 38%→36% and 58%→82%, so drift alone can move it too, just not to 100%. More seeds needed before it's a firm claim. *(Corrected Sep 28: the first version said "not shown"; its runs had original-brain children.)* |
| Sorting with breeding | Where tuned flies spend their time in breeding runs | Resonator: tuned flies at their own patch 51–70% of the time, the other patch 5–7%. Window control is mixed (one seed has A-"tuned" flies at B 70%) because children are born beside their parents and ω is only a family label there. Breeding sorting numbers are lineage-confounded; use the no-breeding test. |

## Files

| file | what |
|---|---|
| `resonator_brain.py` | the two brains (`make_bug_class(EnhancedBug, "resonator" \| "window")`) |
| `resonator_flies.py` | **live webcam**: runs the original dumbflies GUI with the new brain. Rings show each fly's tuning (blue slow → red fast, filled = listening and hearing); top-left strip is the population's tuning histogram in Hz. `--brain window` for the control, `--floor 8` if a noisy camera feeds everyone. |
| `sort_nobreed.py` | the clean sorting test |
| `rescore_fixed.py` | rescoring of the breeding runs after the child-class fix |
| `res_eval.py`, `cam_identity.py`, `res_fig.py` | scoring and figure; numbers in `res_results.json`, `sort_nobreed.json` |

```
python sort_nobreed.py                                        # ~10 min, the key result
python flybench.py rhythm 6000 0,1 resonator                  # long runs (also: window; original at 3000)
python flybench.py room 6000 0 resonator                      # evolution control
python res_eval.py && python res_fig.py
python resonator_flies.py                                     # live, needs dumbflies.py + webcam
```
The live launcher was smoke-tested here with stubbed Tk (brain swap, inheritance, drawing), not with a real webcam.

## Live notes (first overnight run, Sep 27–28)

- **Lights on = baby boom for the slow flies.** A light switch is a step, the slowest signal there is. Measured on the bench (fly held in place, 300 frames after the switch): food 40 for ω = 0.1, 15 for ω = 0.3, 6 for ω = 0.5, under 2 for ω ≥ 0.8; zero at every tuning in the dark. The morning run went to 100 flies, all blue.
- **The loop slowed from ~12.5 to ~6 fps overnight.** The original's population graph replots every point since start on every frame. `resonator_flies.py` now redraws every 30 frames and keeps the last 3000 points. Frame rate matters: a fly's tuning in Hz is ω·fps/2π, so if fps halves, every fly's Hz halves.
- **`flies_log.csv`** gets a row every 10 s: fps, population, births, deaths, brightness, and the tuning histogram. `python plot_log.py` turns it into `flies_log.png`.

### Morning population (saved Sep 28, 106 flies) — `python analyze_population.py pop.json 6.3`

- **Two families own the colony.** Children are clones of one parent: `try_mate` calls `inherit_traits` twice and the second call overwrites the first, so there is no mixing, only rare small mutations. Body size is random at birth and useless to a resonator fly, so it works as a family name. Size 4.241: 90 flies, tuned 0.21 Hz. Size 4.735: 14 flies, 0.27 Hz. The rest is one mutant of the big family and one hungry loner (0.46 Hz, energy 24).
- **Both winners are slow.** Their tunings sit at 9% and 14% of the random starting range; their sizes (the neutral gene) sit at 41% and 58%, unremarkable. If winning had nothing to do with tuning, both landing that low has a chance of about 2%. That fits the light-switch mechanism, but it's one run.
- **"Frequencies went down" was replacement, not adaptation.** Within a family, tuning varies only 0.19–0.23; mutation is far too small to move a family. The slow founders were there from the start (random refills overnight) and took over when the light came on.
- **Everyone was born in the boom** (ages 2138–4140 frames). Nobody from the night survived.
- **The colony was capped, not limited by food.** 105 of 106 were full and ready to mate; only the 100-fly cap stopped births.

---

# Part 3 — Clicker neuron: a colony taught with clicks

![pic](pic.png)

One colony becomes one neuron. The flies are the dendrite: each listening fly is a tuned temporal filter sitting at a place. The soma sums what they hear, and a teacher's click is the modulator. `neuron.py`:

| part | rule | clock |
|---|---|---|
| weight | a fly's energy (÷100, clipped) | medium |
| soma | V = Σ weight × what it hears; fires when V > θ, and θ adapts so it fires ~10% of frames | fast |
| eligibility | a decaying trace of each fly's contribution, boosted when the soma fired | fast |
| click (modulator +) | food = 2 × eligibility, max 40 per fly and 120 per click. Absolute: a click when nobody was listening feeds nobody | — |
| no click (modulator −) | a spike with no click within 30 frames costs 3 energy in total, split by contribution | — |
| anchoring | a fly fed by a click stays put for 400 frames even when it hears nothing (a rewarded synapse is stabilised). Counts down only while learning | slow |
| growth | a fly over 130 energy buds: splits in place, the copy's tuning mutated ±12%. No mating. Unfed flies starve | slow |

Flies eat nothing on their own; only clicks feed them.

**Bench** (`neuron_bench.py`): the room with one lamp that always flickers, switching between A = 0.15 cycles/frame (target) and B = 0.05 (distractor) in 120–240-frame segments, plus the room light switching every 400–700 frames. The teacher clicks 8 frames into each A segment and every 30 frames while it lasts (about 65 clicks in 4000 frames). Then comes a 900-frame test that is identical for every condition, with learning, energy decay and growth off. Four seeds per condition.

![clicker neuron results](neuron_findings.png)

## Ledger (v3)

| claim | test | verdict |
|---|---|---|
| Clicks teach it to fire on the clicked thing | taught A: fires on A minus fires on B | **Holds, 4/4 seeds.** +0.13 (per seed +0.10 to +0.16). Fires on A 14%, on B 0.8%. |
| It's the teaching, not the world | same world, same test, taught B instead | **Holds, 4/4 seeds, reversed.** −0.15 (−0.13 to −0.18). Fires on A 1%, on B 16%. |
| It's the click timing | same number of clicks at random times | **Holds.** −0.03, mixed signs (−0.15 to +0.08). |
| It beats doing nothing | no teacher, flies eat what they hear | **Holds.** +0.07 (this world has a built-in lean toward A: the faster flicker). The colony also shrinks to 5–14 flies, against 32–41 with a teacher. |
| What it learned can be read from the colony | tuning of the colony's flies after training | **Holds.** Taught A: median ω 0.90 (target 0.94), 47% of flies within ±0.15 of A, 3% near B. Taught B: median 0.42, 65% near B (0.31), 1% near A. Same test, different history, different neuron, and the difference is visible in the substrate. |
| It learns to ignore light switches | firing in the 60 frames after a switch, minus the same rhythm without one | **No.** Taught neurons fire +0.14 more after a switch, the untaught one +0.18. Switches were rarely clicked but also rarely punished enough. Open. |
| It beats a standard learner | 40 fixed random resonators + logistic readout trained on the true label of **every** training frame, threshold matched to the neuron's firing rate | **No.** The bank scores +0.18 (random places) and +0.18 (at the lamp), against the neuron's +0.13. The neuron gets ~65 clicks and only local rules, while the bank gets 4000 labelled frames and a global optimiser. The neuron's case is that it learns online from sparse clicks and that its structure is readable, not that it's more accurate. |

## What failed on the way (same bench, same seeds)

- **v0** used the original mating for growth. Fed flies rarely met, and the colony starved to 5–10 flies that were mostly random refills. Not selective: taught A fired more on A in 1 of 3 seeds, and all 3 fired hard on light switches.
- **v1** added budding and made click-food absolute. A > B in 3 of 4 seeds, but clearly so in only one; seed 2 was backwards. Instrumenting seed 2 showed the colony was 1–2 flies at the lamp, because every B segment made the A-tuned flies hear nothing and hop away. The test then caught a fresh random refill.
- **v2** added anchoring. The colony's energy still drained. Over 2500 frames on seed 2, clicks brought in 2234 energy and 1334 left through the false-alarm cost plus budding (the measurement lumped the two; a bud halves its parent). The cost was charged to every contributor per spike, with the soma held near 10% firing.
- **v3** made each unconfirmed spike cost a fixed 3 energy split by share, and doubled click-food. The colony holds 32–41 flies, and this is the ledger above.

Raw runs: `neuron_runs/`; v0/v1 runs in `neuron_runs_v0/`, `neuron_runs_v1/`; summary `neuron_results.json`.

## Live

```
python clicker_neuron.py                 # SPACE = click
python clicker_neuron.py --self-feed 1   # control: flies also eat what they hear
```
The soma is drawn bottom-left, flashing yellow when it fires. Lines from flies to the soma are the dendrite, meaning who is feeding it right now. The bottom-right strip shows soma voltage (white) against threshold (red). Press SPACE whenever the thing you want detected is happening: a steady blink on a phone, a hand wave. If nobody clicks, the colony starves in a few minutes and is replaced by random flies. Save Population keeps a trained neuron. `neuron_log.csv` gets a row every 10 s. Smoke-tested here with a stubbed GUI, not a real webcam.

```
python neuron_bench.py clicker,clicker_B,shuffled,no_credit 0,1,2,3   # ~40 min on 2 cores
python bank_control.py 0,1,2,3
python neuron_report.py
```
