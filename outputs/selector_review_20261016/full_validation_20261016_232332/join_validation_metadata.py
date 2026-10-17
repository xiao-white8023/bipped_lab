from pathlib import Path
import numpy as np, torch, json,hashlib
root=Path('/home/cjh/TienKung-Lab/outputs/g1_renet_selector_v4_1s_normal_20261012_170014/dataset')
run_report=json.loads(Path('/home/cjh/桌面/joint_training_report.json').read_text())
assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()==run_report['training_sources']['original_manifest_sha256']
p=np.load('/home/cjh/桌面/validation_predictions.npz'); target_id=p['sample_id']; wanted=['scenario_seed','labels','fault_metadata','terrain_level','terrain_type','execution_metadata','future_metrics','candidate_time','feature_end_time','event_identity','fault_parameters','first_failure_flags','first_failure_cause_stable','candidate_safety']
parts={k:[] for k in wanted}; parts.update({'original_current_branch':[],'original_shard_index':[],'original_row_in_shard':[],'group_ids':[]})
shards=sorted((root/'validation').glob('shard_*.pt'))
assert len(shards)==26
for si,path in enumerate(shards):
 d=torch.load(path,map_location='cpu',weights_only=False); n=len(d['scenario_seed'])
 for k in wanted:parts[k].append(d[k].numpy())
 parts['original_current_branch'].append(d['features'][:,-1,-2].numpy().astype(np.uint8)); parts['original_shard_index'].append(np.full(n,si,np.int32)); parts['original_row_in_shard'].append(np.arange(n,dtype=np.int32)); parts['group_ids'].append(np.asarray(d['group_ids']))
 del d
parts={k:np.concatenate(v) for k,v in parts.items()}; fullid=parts['scenario_seed']; assert len(np.unique(fullid))==len(fullid); assert len(np.unique(target_id))==len(target_id)
sort=np.argsort(fullid); where=np.searchsorted(fullid[sort],target_id); assert (where<len(fullid)).all(); join=sort[where]; assert np.array_equal(fullid[join],target_id)
fullf=parts['fault_metadata']; fully=parts['labels']; clean_reject=((fullf[:,6]!=0)&(fullf[:,7]==0)&(fully[:,1]!=0))|((fullf[:,7]!=0)&(fullf[:,6]==0)&(fully[:,0]!=0)); assert np.array_equal(np.sort(fullid[~clean_reject]),np.sort(target_id))
out={k:v[join] for k,v in parts.items()}; assert np.array_equal(out['labels'],p['labels']); assert np.array_equal(out['original_current_branch'],p['current_branch'])
f=out['fault_metadata']; out['fault_source']=((f[:,0]!=0).astype(np.uint8)+2*(f[:,1]!=0).astype(np.uint8)); out['active_fault_source']=(f[:,6].astype(np.uint8)+2*f[:,7].astype(np.uint8)); out['sample_id']=target_id
np.savez_compressed(str(Path(__file__).with_name('validation_metadata_join.npz')),**out)
report={'original_manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),'source_validation_rows':len(fullid),'joined_rows':len(target_id),'labels_match_all':True,'current_branch_match_all':True,'sample_ids_unique':True,'exact_clean_filter_sample_id_set_match':True,'original_rows_not_in_cleaned_predictions':len(fullid)-len(target_id),'output':str(Path(__file__).with_name('validation_metadata_join.npz')),'fault_source_codes':{'0':'none','1':'lp_only','2':'vp_only','3':'dual'},'fault_metadata_order':['lp_kind','vp_kind','severity','phase','remaining_frames','duration_frames','lp_active','vp_active'],'execution_metadata_order':['mode','dwell_s','remaining_blend_steps'],'fields':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in out.items()}}
json.dump(report,open(str(Path(__file__).with_name('validation_metadata_join.json')),'w'),indent=2); print(json.dumps(report))
