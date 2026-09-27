"""Camera alone vs camera + the fly's own tuning (omega, r): does knowing WHO the
fly is add to what the video explains? Should help resonator flies only."""
import sys, json, numpy as np
import res_eval as E, analyze as A
brain = sys.argv[1]
dtr, dte = E.load(f"{brain}_rhythm_s0"), E.load(f"{brain}_rhythm_s1")
s_tr, _ = A.steps(dtr); s_te, _ = A.steps(dte)
Ctr = np.c_[A.img_features(dtr, "rhythm", 0), E.spectral_features(dtr, "rhythm", 0)]
Cte = np.c_[A.img_features(dte, "rhythm", 1), E.spectral_features(dte, "rhythm", 1)]
Gtr, Gte = np.c_[dtr["omega"], dtr["r"]], np.c_[dte["omega"], dte["r"]]
out = {"camera": A.fit_score(Ctr, s_tr, Cte, s_te),
       "tuning_only": A.fit_score(Gtr, s_tr, Gte, s_te),
       "camera_plus_tuning": A.fit_score(np.c_[Ctr, Gtr], s_tr, np.c_[Cte, Gte], s_te)}
json.dump(out, open(f"cam_identity_{brain}.json", "w"), indent=2); print(brain, out)
