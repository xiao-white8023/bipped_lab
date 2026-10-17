import json,numpy as np
from pathlib import Path
z=np.load('/home/cjh/桌面/validation_predictions.npz')
y=z['labels']; t=y[:,0]+2*y[:,1]; b=z['current_branch']; N=len(t); ii=np.arange(N); sf=(t==1)|(t==2); cf=np.where(b==1,y[:,0],y[:,1]).astype(bool); need=sf&cf
out={'rows':N,'class_counts':np.bincount(t,minlength=4).tolist(),'current_counts':np.bincount(b,minlength=2).tolist(),'profiles':{},'paired':{}}
for mode in ['raw','temperature','temperature_bias']:
 q=z['probabilities_'+mode]; p=q.argmax(1); desired=np.where(p==1,0,np.where(p==2,1,b)); sw=desired!=b; sel=np.where(desired==1,y[:,0],y[:,1]).astype(bool); cr=need & ~sel; miss=need&sel; harm=sf&~cf&sel
 assert np.array_equal(p,z['predicted_class_'+mode]); assert np.array_equal(desired,z['desired_branch_'+mode]); assert np.array_equal(cr,z['correct_rescue_'+mode]); assert np.array_equal(miss,z['missed_rescue_'+mode]); assert np.array_equal(harm,z['harmful_switch_'+mode])
 cm=np.zeros((4,4),dtype=int);np.add.at(cm,(t,p),1)
 d={'confusion':cm.tolist(),'class_argmax_counts':np.bincount(p,minlength=4).tolist(),'correct_rescue':int(cr.sum()),'needed_rescue':int(need.sum()),'missed_rescue':int(miss.sum()),'harmful_switch':int(harm.sum()),'single_failure_wrong':int((sf&sel).sum()),'single_failure_count':int(sf.sum()),'both_safe_switch':int((sw&(t==0)).sum()),'both_safe_count':int((t==0).sum()),'both_failure_switch':int((sw&(t==3)).sum()),'both_failure_count':int((t==3).sum()),'both_failure_warning_tp':int(((p==3)&(t==3)).sum()),'warning_count':int((p==3).sum()),'directions':{}}
 for branch in [0,1]:
  mask=need&(b==branch); safe_sf=sf&~cf&(b==branch)
  d['directions']['current_'+('VP' if branch==0 else 'LP')]={'needed_rescue':int(mask.sum()),'correct_rescue':int((mask&cr).sum()),'missed_rescue':int((mask&miss).sum()),'missed_as_safe':int((mask&(p==0)).sum()),'missed_as_both_failure':int((mask&(p==3)).sum()),'missed_as_wrong_single':int((mask&np.isin(p,[1,2])&sel).sum()),'safe_current_single_cases':int(safe_sf.sum()),'harmful_switch':int((safe_sf&harm).sum()),'both_safe_switch':int(((b==branch)&(t==0)&sw).sum()),'both_failure_switch':int(((b==branch)&(t==3)&sw).sum())}
 out['profiles'][mode]=d
 if mode!='raw':
  rp=z['predicted_class_raw']; rc=z['correct_rescue_raw']; rh=z['harmful_switch_raw']; mat=np.zeros((4,4),int); np.add.at(mat,(rp,p),1)
  x={'transition':mat.tolist(),'class_changed':int((rp!=p).sum()),'desired_changed':int((desired!=z['desired_branch_raw']).sum()),'lost':int((rc&~cr).sum()),'gained':int((~rc&cr).sum()),'harmful_added':int((~rh&harm).sum()),'harmful_removed':int((rh&~harm).sum()),'by_branch':{}}
  for branch in [0,1]:
   mask=b==branch;x['by_branch']['VP' if branch==0 else 'LP']={'lost':int((mask&rc&~cr).sum()),'gained':int((mask&~rc&cr).sum()),'harmful_added':int((mask&~rh&harm).sum()),'harmful_removed':int((mask&rh&~harm).sum())}
  out['paired'][mode]=x
print(json.dumps(out,indent=2));Path(str(Path(__file__).with_name('profiles_diagnostic.json'))).write_text(json.dumps(out,indent=2))
