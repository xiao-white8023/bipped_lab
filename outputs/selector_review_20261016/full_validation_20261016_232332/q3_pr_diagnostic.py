from pathlib import Path
import numpy as np,json
p=np.load('/home/cjh/桌面/validation_predictions.npz'); y=p['labels'][:,0]+2*p['labels'][:,1]; branch=p['current_branch']; N=len(y)

def auc(s,t):
 order=np.argsort(s,kind='stable'); ss=s[order]; tt=t[order]; ends=np.r_[np.flatnonzero(np.diff(ss)),len(ss)-1]; starts=np.r_[0,ends[:-1]+1]; ranks=np.empty(len(s));
 for a,b in zip(starts,ends):ranks[a:b+1]=(a+b)/2+1
 P=int(t.sum()); Q=len(t)-P
 return float((np.dot(tt,ranks)-P*(P+1)/2)/(P*Q))

def calc(q,mask=None):
 if mask is None: mask=np.ones(N,bool)
 s=q[mask,3]; yy=y[mask]; b=branch[mask]; t=yy==3; order=np.argsort(-s,kind='stable'); ss=s[order]; ts=t[order]; ends=np.r_[np.flatnonzero(np.diff(ss)),len(ss)-1]; flags=ends+1; tp=np.cumsum(ts)[ends]; prec=tp/flags; rec=tp/t.sum()
 def point(idx):
  threshold=float(ss[ends[idx]]); chosen=s>=threshold; classes=np.bincount(yy[chosen],minlength=4)
  return {'threshold':threshold,'flagged':int(chosen.sum()),'true_positive':int(tp[idx]),'false_positive':int(chosen.sum()-tp[idx]),'precision':float(prec[idx]),'recall':float(rec[idx]),'flagged_class_counts':classes.tolist()}
 feasible=prec>=.5
 i=np.where(feasible)[0][np.argmax(rec[feasible])]
 j=np.where(rec>=.8)[0][np.argmax(prec[rec>=.8])]
 ap=np.sum(np.diff(np.r_[0,rec])*prec)
 byneg={}
 for c in range(3):
  pair=(yy==c)|(yy==3); byneg[str(c)]=auc(s[pair],yy[pair]==3)
 return {'samples':len(s),'positive':int(t.sum()),'prevalence':float(t.mean()),'roc_auc':auc(s,t),'average_precision':float(ap),'max_recall_at_precision_50':point(i),'max_precision_at_recall_80':point(j),'pairwise_auc_vs_negative_class':byneg,'q3_class_quantiles':{str(c):np.quantile(s[yy==c],[0,.1,.25,.5,.75,.9,1]).tolist() for c in range(4)}}

out={}
for mode in ['raw','temperature','temperature_bias']:
 q=p['probabilities_'+mode]; out[mode]=calc(q); out[mode]['by_current_branch']={name:calc(q,branch==code) for name,code in [('VP',0),('LP',1)]}
 json.dump(out,open(str(Path(__file__).with_name('q3_pr_diagnostic.json')),'w'),indent=2)
 print(mode,json.dumps(out[mode],ensure_ascii=False))
