import numpy as np,json
from pathlib import Path
p=np.load('/home/cjh/桌面/validation_predictions.npz');a=np.load(str(Path(__file__).with_name('validation_metadata_join.npz')));assert np.array_equal(p['sample_id'],a['sample_id']);assert np.array_equal(p['labels'],a['labels']);assert np.array_equal(p['current_branch'],a['original_current_branch'])
y=p['labels'];b=p['current_branch'];t=y[:,0]+2*y[:,1];pred=p['predicted_class_raw'];cf=np.where(b==1,y[:,0],y[:,1]).astype(bool);sf=(t==1)|(t==2);need=sf&cf;miss=p['missed_rescue_raw'];f=a['fault_metadata'];planned=a['fault_source'];active=f[:,6]+2*f[:,7];ex=a['execution_metadata'];first=np.where(b==1,a['future_metrics'][:,0,0],a['future_metrics'][:,1,0]);flags=np.where(b==1,a['first_failure_flags'][:,0],a['first_failure_flags'][:,1]);
def count_dist(x,m):
 u,c=np.unique(x[m],return_counts=True);return {str(v.item()):int(n)for v,n in zip(u,c)}
def summary(mask):
 return dict(rows=int(mask.sum()),planned_source=count_dist(planned,mask),active_source=count_dist(active,mask),lp_kind=count_dist(f[:,0],mask),vp_kind=count_dist(f[:,1],mask),severity=count_dist(f[:,2],mask),phase=count_dist(f[:,3],mask),terrain_level=count_dist(a['terrain_level'],mask),terrain_type=count_dist(a['terrain_type'],mask),current_branch=count_dist(b,mask),predicted_class=count_dist(pred,mask),dwell_lt1=int((mask&(ex[:,1]<1)).sum()),dwell_le1=int((mask&(ex[:,1]<=1)).sum()),blend_positive=int((mask&(ex[:,2]>0)).sum()),dwell_quantiles=np.quantile(ex[mask,1],[0,.25,.5,.75,1]).tolist()if mask.any()else[],first_failure_le5=int((mask&(first>0)&(first<=5)).sum()),first_failure_quantiles=np.quantile(first[mask],[0,.25,.5,.75,1]).tolist()if mask.any()else[],first_failure_flags=count_dist(flags,mask))
def breakdown(base,field):
 out={}
 for v in np.unique(field[base]):
  m=base&(field==v);out[str(v.item())]={'need':int(m.sum()),'missed':int((m&miss).sum()),'missed_q0':int((m&miss&(pred==0)).sum()),'missed_q3':int((m&miss&(pred==3)).sum()),'missed_wrong_single':int((m&miss&np.isin(pred,[1,2])).sum())}
 return out
out={'metadata_integrity':dict(labels_match=True,branch_match=True,sample_id_match=True),'groups':{}}
for name,m in {'all_need':need,'all_missed':miss,'miss_q0':miss&(pred==0),'miss_q3':miss&(pred==3),'vp_need':need&(b==0),'vp_missed':miss&(b==0),'vp_missed_q0':miss&(b==0)&(pred==0),'vp_missed_q3':miss&(b==0)&(pred==3),'lp_missed':miss&(b==1)}.items():out['groups'][name]=summary(m)
for name,mask in [('all',need),('VP',need&(b==0)),('LP',need&(b==1))]:
 out[name+'_breakdown']={k:breakdown(mask,v)for k,v in {'planned':planned,'active':active,'terrain_level':a['terrain_level'],'terrain_type':a['terrain_type'],'vp_kind':f[:,1],'lp_kind':f[:,0],'severity':f[:,2],'phase':f[:,3]}.items()}
Path(str(Path(__file__).with_name('profiles_metadata_diagnostic.json'))).write_text(json.dumps(out,indent=2))
for name in ['all_need','all_missed','miss_q0','miss_q3','vp_need','vp_missed','vp_missed_q0','vp_missed_q3','lp_missed']: print(name,json.dumps(out['groups'][name]))
for name in ['VP','LP']:print(name+'_breakdown',json.dumps(out[name+'_breakdown']))
