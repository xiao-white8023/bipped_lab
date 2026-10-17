from pathlib import Path
import json,numpy as np
f=np.load('/home/cjh/桌面/validation_predictions.npz',allow_pickle=False)
y=f['labels'].astype(bool); cur=f['current_branch']; c=y[:,0].astype(int)+2*y[:,1].astype(int)
cf=np.where(cur==1,y[:,0],y[:,1]); af=np.where(cur==1,y[:,1],y[:,0]); rescue=cf&~af; harmful=~cf&af; single=c==1; single|=c==2
base={'samples':len(c),'class_counts':np.bincount(c,minlength=4).tolist(),'rescue':int(rescue.sum()),'harmful_opportunities':int(harmful.sum()),'single':int(single.sum()),'rescue_current_lp':int((rescue&(cur==1)).sum()),'rescue_current_vp':int((rescue&(cur==0)).sum())}
def n(x):return int(x.sum())
def metric(sw):
 miss=rescue&~sw; harm=harmful&sw
 return dict(request=n(sw),both_safe_request=n(sw&(c==0)),both_safe_rate=n(sw&(c==0))/n(c==0),both_fail_request=n(sw&(c==3)),both_fail_rate=n(sw&(c==3))/n(c==3),correct_rescue=n(sw&rescue),rescue_rate=n(sw&rescue)/n(rescue),missed_rescue=n(miss),harmful=n(harm),dangerous=n(miss|harm),dangerous_rate=n(miss|harm)/n(single),lp_to_vp_rescued=n(sw&rescue&(cur==1)),vp_to_lp_rescued=n(sw&rescue&(cur==0)))
out={'base':base,'profiles':{}}
for mode in ('raw','temperature','temperature_bias'):
 q=f['probabilities_'+mode].astype(float); pred=q.argmax(1); arg=f['switch_request_'+mode]; delta=np.where(cur==1,q[:,1]-q[:,2],q[:,2]-q[:,1])
 assert np.array_equal(arg, ((pred==1)&(cur==1))|((pred==2)&(cur==0)))
 rules={'argmax':arg,'min_risk_delta_gt_0':delta>0,'argmax_and_delta_gt_0.01':arg&(delta>.01)}
 for t in (.01,.05,.1): rules[f'all_delta_gt_{t}']=delta>t
 sub={'rules':{k:metric(v) for k,v in rules.items()},'missed_rescue_by_argmax_class':{str(i):n(rescue&~arg&(pred==i)) for i in range(4)}}
 sub['q0_argmax_missed_rescue_delta']={str(t):n(rescue&~arg&(pred==0)&(delta>t)) for t in (0,.01,.05,.1,.2)}
 sub['q0_argmax_delta']={str(t):{'all':n((pred==0)&(delta>t)),'true_safe':n((pred==0)&(delta>t)&(c==0)),'true_dual':n((pred==0)&(delta>t)&(c==3)),'true_rescue':n((pred==0)&(delta>t)&rescue),'true_harmful':n((pred==0)&(delta>t)&harmful)} for t in (.01,.05,.1)}
 sub['raw_vs_argmax_checks']={'precomputed_rescue_match':n(f['correct_rescue_'+mode])==sub['rules']['argmax']['correct_rescue'],'precomputed_wrong_match':n(f['single_failure_wrong_'+mode])==sub['rules']['argmax']['dangerous']}
 # delta fixed decisions partition by deterministic sample id hash; groups are repeatability check, NOT holdout for calibration/model
 groups=((f['sample_id'].astype(np.uint64)*np.uint64(11400714819323198485))>>np.uint64(63)).astype(bool)
 sub['fixed_half_counts']=[n(~groups),n(groups)]
 sub['q0_delta_examples']=[]
 ix=np.where(rescue&~arg&(pred==0)&(delta>.1))[0]
 for i in ix[np.argsort(-delta[ix])[:3]]:
  sub['q0_delta_examples'].append({'row':int(i),'sample_id':int(f['sample_id'][i]),'current_branch':int(cur[i]),'labels':y[i].astype(int).tolist(),'q':q[i].tolist(),'delta':float(delta[i])})
 out['profiles'][mode]=sub
print(json.dumps(out,indent=2))
open(str(Path(__file__).with_name('decision_diagnostic.json')),'w').write(json.dumps(out,indent=2))
