"""Rescore breeding runs after the child-class fix (Sep 28)."""
import json
import res_eval as E
R = {}
for b in ("resonator", "window"):
    for s in (0, 1):
        d = E.load(f"{b}_rhythm_s{s}")
        R[f"sorting_with_breeding_{b}_s{s}__LINEAGE_CONFOUNDED"] = E.sorting(d)
        R[f"evolution_{b}_rhythm_s{s}"] = E.evolution(d)
R["evolution_resonator_room_NO_RHYTHM_s0"] = E.evolution(E.load("resonator_room_s0"))
for b, tag in (("resonator", "resonator_"), ("window", "window_")):
    R[f"camera_observer_{b}"] = E.camera_scores(f"{tag}rhythm_s0", f"{tag}rhythm_s1")
    print(b, R[f"camera_observer_{b}"], flush=True)
json.dump(R, open("rescored_fixed.json", "w"), indent=2)
print(json.dumps({k: v for k, v in R.items() if k.startswith("evolution")}, indent=1))
