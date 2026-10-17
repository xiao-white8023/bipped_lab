from pathlib import Path
import numpy as np,json
p=np.load('/home/cjh/桌面/validation_predictions.npz'); m=np.load(str(Path(__file__).with_name('validation_metadata_join.npz'))); q=p['probabilities_raw']; y=p['labels'][:,0]+2*p['labels'][:,1]; pos=y==3; arg=q.argmax(1); flag=q[:,3]>=np.float32(.4340755045413971); f=m['fault_metadata']; n=len(y)
kind_names={0:'NONE',1:'FREEZE',2:'DELAY',3:'VP_OCCLUSION',4:'VP_SCALE',5:'VP_BIAS',6:'VP_SHIFT',7:'VP_COLLAPSE',8:'LP_RANDOM_DROPOUT',9:'LP_BLOCK_DROPOUT',10:'LP_HEIGHT_BIAS',11:'LP_REGISTRATION',12:'LP_DRIFT',13:'LP_COLLAPSE'}

def summary(mask):
 z=mask&pos; zz=mask&~pos; fn=z&~flag; fp=zz&flag
 return {'rows':int(mask.sum()),'class_counts':np.bincount(y[mask],minlength=4).tolist(),'double_failure_positive':int(z.sum()),'argmax_missed_double_failure':int((z&(arg!=3)).sum()),'argmax_double_failure_recall':float((z&(arg==3)).sum()/z.sum()) if z.any() else None,'threshold_tp':int((z&flag).sum()),'threshold_fn':int(fn.sum()),'threshold_recall':float((z&flag).sum()/z.sum()) if z.any() else None,'negative_rows':int(zz.sum()),'threshold_fp':int(fp.sum()),'threshold_false_positive_rate':float(fp.sum()/zz.sum()) if zz.any() else None,'threshold_fp_true_class_counts':np.bincount(y[fp],minlength=4).tolist(),'argmax_missed_true3_predictions':np.bincount(arg[z&(arg!=3)],minlength=4).tolist()}

def group(values,decoder=str):
 return {decoder(k):summary(values==k) for k in sorted(np.unique(values))}

out={'q3_threshold':float(np.float32(.4340755045413971)),'overall':summary(np.ones(n,bool)), 'by_fault_source':group(m['fault_source'],lambda k:['none','lp_only','vp_only','dual'][k]),'by_lp_kind':group(f[:,0],lambda k:kind_names.get(int(k),str(k))),'by_vp_kind':group(f[:,1],lambda k:kind_names.get(int(k),str(k))),'by_terrain_type_column':group(m['terrain_type']),'by_terrain_level':group(m['terrain_level']),'by_fault_phase':group(f[:,3]),'by_severity':group(f[:,2]),'by_lp_active':group(f[:,6]),'by_vp_active':group(f[:,7])}
pairs=f[:,0].astype(np.int64)*100+f[:,1]; out['by_fault_pair']=group(pairs,lambda k:kind_names.get(int(k//100),str(k//100))+'+'+kind_names.get(int(k%100),str(k%100)))
for section in ['by_fault_source','by_lp_kind','by_vp_kind','by_terrain_type_column','by_terrain_level','by_fault_phase','by_severity']:
 print('\n'+section)
 for k,v in out[section].items():print(k,'N',v['rows'],'positive',v['double_failure_positive'],'argmiss',v['argmax_missed_double_failure'],'thrFN',v['threshold_fn'],'thrFP',v['threshold_fp'],'negative',v['negative_rows'],'fpclasses',v['threshold_fp_true_class_counts'],'R',v['threshold_recall'])
print('\nFaultpairpositive>=30 sortedFN')
for k,v in sorted(out['by_fault_pair'].items(),key=lambda x:x[1]['threshold_fn'],reverse=True):
 if v['double_failure_positive']>=30:print(k,v)
json.dump(out,open(str(Path(__file__).with_name('q3_metadata_diagnostic.json')),'w'),indent=2)
out['by_active_fault_source']=group(m['active_fault_source'],lambda k:['none','lp_only','vp_only','dual'][k])
print('\nby_active_fault_source')
for k,v in out['by_active_fault_source'].items(): print(k,v)
event,evindex=np.unique(m['event_identity'],axis=0,return_inverse=True); ecount=len(event)
negcount=np.bincount(evindex[~pos],minlength=ecount); fpcount=np.bincount(evindex[(~pos)&flag],minlength=ecount); poscount=np.bincount(evindex[pos],minlength=ecount); fncount=np.bincount(evindex[pos&~flag],minlength=ecount)
out['error_event_concentration']={'event_count':ecount,'fp_event_count':int((fpcount>0).sum()),'fn_event_count':int((fncount>0).sum()),'top10_fp_sum':int(np.sort(fpcount)[-10:].sum()),'top10_fn_sum':int(np.sort(fncount)[-10:].sum())}
for name,counts in [('fp',fpcount),('fn',fncount)]:
 rows=[]
 for ei in np.argsort(-counts)[:15]:
  if not counts[ei]:continue
  mask=evindex==ei; entry={'event_identity':event[ei].tolist(),'group_ids':list(np.unique(m['group_ids'][mask])),'fault_pairs':np.unique(f[mask,:2],axis=0).tolist(),**summary(mask)}; rows.append(entry)
 out['top_'+name+'_events']=rows
json.dump(out,open(str(Path(__file__).with_name('q3_metadata_diagnostic.json')),'w'),indent=2)
print('event concentration',out['error_event_concentration'])
for e in out['top_fp_events'][:5]: print('topFPevent',e)
t=m['future_metrics'][:,:,0].astype(float); maxfirst=t.max(1)
out['by_later_branch_first_failure_time']={f'{lo}_to_{hi}_steps':summary(pos&(maxfirst>=lo)&(maxfirst<=hi)) for lo,hi in [(1,10),(11,25),(26,50)]}
out['first_failure_step_medians']={name:{'rows':int(mask.sum()),'lp_vp':np.median(t[mask],axis=0).tolist(),'later_branch':float(np.median(maxfirst[mask]))} for name,mask in [('all_true3',pos),('argmax_missed_true3',pos&(arg!=3)),('argmax_detected_true3',pos&(arg==3))]}
json.dump(out,open(str(Path(__file__).with_name('q3_metadata_diagnostic.json')),'w'),indent=2)
print('Failure timing',json.dumps(out['by_later_branch_first_failure_time']))
