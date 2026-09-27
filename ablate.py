import json, numpy as np, analyze as A
Itr,Htr,Otr,s_tr,t_tr = A.features(A.runs['room_s0'],'room',0)
Ite,Hte,Ote,s_te,t_te = A.features(A.runs['room_s1'],'room',1)
sets = {
 'momentum_only (prev 2 turns/speeds)': (Htr[:,10:14], Hte[:,10:14]),
 'where_it_is_and_has_been (no momentum)': (Htr[:,:10], Hte[:,:10]),
 'where_it_is_now_only': (Htr[:,:4], Hte[:,:4]),
 'full_history': (Htr, Hte)}
out={}
for tgt,(ya,yb) in {'turn':(t_tr,t_te),'speed':(s_tr,s_te)}.items():
    out[tgt]={k:A.fit_score(a,ya,b,yb) for k,(a,b) in sets.items()}
print(json.dumps(out,indent=2))
json.dump(out,open('ablation.json','w'),indent=2)
