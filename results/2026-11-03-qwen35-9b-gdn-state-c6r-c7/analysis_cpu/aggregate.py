"""CPU-only exact quantiles, paired comparisons and plots, one layer at a time."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
from pathlib import Path
import csv
import gc
import itertools
import json
import resource
import time
import numpy as np
from common import (ROOT,LAYERS,METHODS,SAMPLES,NAMES,IDX,PAIRS,CSVStream,jd,sha,
                    summarize,metadata,load_layer,paired,pearson,top_indices,histograms)
from plots import distributions,key_maps,cross_layer,relationship

def sign(x):return int(np.sign(x))

def high_groups(exp,layer,values,peak_indices,positions,writer):
    for policy in ['Native_selected','self_selected']:
        selection_methods=['Native'] if policy=='Native_selected' else METHODS
        for selector in selection_methods:
            chosen=top_indices(values[selector][...,IDX['peak_rms']])
            output_methods=METHODS if policy=='Native_selected' else [selector]
            for rank,flat in enumerate(chosen,1):
                event,head,value=np.unravel_index(flat,values[selector].shape[:3])
                sample=SAMPLES[event//129];token=int(positions[event%129])
                for method in output_methods:
                    m=values[method][event,head,value]
                    writer.write({'experiment':exp,'model_layer_id':layer,'selection_policy':policy,'selection_method':selector,
                                  'method':method,'rank':rank,'sample_id':sample,'token_idx':token,'head_idx':int(head),'value_idx':int(value),
                                  'largest_absolute_value_Key_index_in_this_basis':int(peak_indices[method][event,head,value]),
                                  'absmax':float(m[2]),'peak_rms':float(m[4]),'local_qdq_relative_L2':float(m[7]),
                                  'out3_key_element_count':int(m[9]),'out6_key_element_count':int(m[10]),
                                  'alignment':'same_FP_State_group' if exp=='A' else 'same_input_position_only_independent_states'})

def historical_comparison(summary,writer):
    current={(r['experiment'],r['method'],r['model_layer_id']):r for r in summary}
    fields=['peak_rms_median','peak_rms_p95','peak_rms_p99','qdq_rel_l2_median','qdq_rel_l2_p95','qdq_rel_l2_p99',
            'out3_element_fraction','out6_element_fraction','absmax_p99','kurtosis_median','scale_median','pooled_qdq_relative_L2']
    old_rows=list(csv.DictReader((ROOT/'historical_comparison/per_layer_summary.csv').open()))
    deltas=[];old_index={}
    for old in old_rows:
        method='Hadamard' if 'H128' in old['condition'] else 'Native';layer=int(old['model_layer_id']);exp=old['experiment']
        old_index[exp,method,layer]=old
        new=current[exp,method,layer]
        for field in fields:
            a=float(old[field]);b=float(new[field]);d=b-a;relative=abs(d)/(abs(a)+1e-12)
            row={'experiment':exp,'method':method,'model_layer_id':layer,'metric':field,'historical_value':a,
                 'current_value':b,'current_minus_historical':d,'relative_absolute_difference':relative,'numeric_exact':a==b}
            writer.write(row);deltas.append(row)
    trends=[]
    for exp in ['A','B']:
        for field in ['peak_rms_median','out3_element_fraction','qdq_rel_l2_median']:
            changes=[]
            for layer in LAYERS:
                old_sign=sign(float(old_index[exp,'Hadamard',layer][field])-float(old_index[exp,'Native',layer][field]))
                new_sign=sign(current[exp,'Hadamard',layer][field]-current[exp,'Native',layer][field])
                if old_sign!=new_sign:changes.append(layer)
            trends.append({'experiment':exp,'metric':field,'changed_Hadamard_vs_Native_direction_layers':changes})
    return {'compared_numeric_fields':len(deltas),'numeric_exact_fields':sum(x['numeric_exact'] for x in deltas),
            'maximum_relative_absolute_difference':max(x['relative_absolute_difference'] for x in deltas),
            'largest_differences':sorted(deltas,key=lambda r:r['relative_absolute_difference'],reverse=True)[:12],
            'trend_comparisons':trends,'scope':'descriptive comparison, no new arbitrary admission tolerance or accuracy claim'}

def findings(summary,ranking):
    index={(r['experiment'],r['method'],r['model_layer_id']):r for r in summary};result={}
    fields=['peak_rms_median','out3_element_fraction','out6_element_fraction','qdq_rel_l2_median','kurtosis_median','scale_median']
    for exp in ['A','B']:
        presence={m:{'out3_elements':sum(index[exp,m,l]['out3_count'] for l in LAYERS),
                     'out6_elements':sum(index[exp,m,l]['out6_count'] for l in LAYERS),
                     'layers_with_out3':[l for l in LAYERS if index[exp,m,l]['out3_count']>0],
                     'layers_with_out6':[l for l in LAYERS if index[exp,m,l]['out6_count']>0]} for m in METHODS}
        comparisons={}
        for candidate,reference in PAIRS:
            comparisons[candidate+'_vs_'+reference]={}
            for field in fields:
                delta={l:index[exp,candidate,l][field]-index[exp,reference,l][field] for l in LAYERS}
                comparisons[candidate+'_vs_'+reference][field]={'candidate_lower_layers':[l for l,v in delta.items() if v<0],
                    'candidate_higher_layers':[l for l,v in delta.items() if v>0],'equal_layers':[l for l,v in delta.items() if v==0]}
        rs=[r for r in ranking if r['experiment']==exp]
        complete_peak=[l for l in LAYERS if all(r['peak_vs_error_same_sign'] for r in rs if r['model_layer_id']==l)]
        complete_tail=[l for l in LAYERS if all(r['out3_vs_error_same_sign'] for r in rs if r['model_layer_id']==l)]
        opposite=[r for r in rs if r['peak_error_strict_opposite'] or r['out3_error_strict_opposite']]
        result[exp]={'outlier_presence':presence,'comparisons':comparisons,'peak_and_error_full_order_same_layers':complete_peak,
                     'out3_and_error_full_order_same_layers':complete_tail,'strict_opposite_order_cases':opposite}
    concordance=[]
    for candidate,reference in PAIRS:
        for field in fields:
            same=[];different=[]
            for l in LAYERS:
                a=sign(index['A',candidate,l][field]-index['A',reference,l][field])
                b=sign(index['B',candidate,l][field]-index['B',reference,l][field])
                (same if a==b else different).append(l)
            concordance.append({'candidate':candidate,'reference':reference,'metric':field,'same_direction_layers':same,'different_direction_layers':different})
    result['A_B_direction_concordance']=concordance
    return result

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
    assert json.loads((ROOT/'formal/RESULT.json').read_text())['status']=='PASS'
    parent=json.loads((ROOT/'operations/PARENT_COMPLETE.json').read_text());assert parent['status']=='PASS_GPU_CAPTURE_READY_FOR_CPU_ANALYSIS'
    assert json.loads((ROOT/'operations/WORKER_WAIT.json').read_text())['owning_Popen_wait_integer']==0
    source_files=sorted(Path(__file__).resolve().parent.glob('*.py'))
    source_rows=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in source_files]
    jd(ROOT/'operations/CPU_ANALYSIS_CLAIM.json',{'pid':os.getpid(),'sources':source_rows,'GPU_worker_restarted':False})
    out=ROOT/'analysis';out.mkdir(exist_ok=False)
    figs=ROOT/'figures';figs.mkdir(exist_ok=False)
    names=['per_layer_summary','per_layer_head_summary','per_sample_layer_summary','per_token_layer_summary',
           'A_paired_group_effects','high_outlier_groups','metric_order_comparison','historical_native_hadamard_comparison']
    writers={n:CSVStream(out/(n+'.csv')) for n in names};all_summary=[];ranking=[]
    prompts={sid:len(json.loads((ROOT/'inputs'/f'{sid}.json').read_text())['prompt_ids']) for sid in SAMPLES}
    start=time.monotonic()
    for exp in ['A','B']:
        hist=histograms(exp);summ={};figure_out=figs/exp;figure_out.mkdir()
        for layer in LAYERS:
            print(json.dumps({'phase':'CPU_ANALYSIS_LAYER','experiment':exp,'layer':layer}),flush=True)
            values,peaks,positions=load_layer(exp,layer)
            for method in METHODS:
                m=values[method];base=metadata(exp,method,layer)
                row={**base,**summarize(m),'valid_samples':4,'sampled_sample_token_events':516,
                     'group_peak_QDQ_Pearson':pearson(m[...,4],m[...,7]),
                     'group_out3_count_QDQ_Pearson':pearson(m[...,9],m[...,7])}
                assert row['group_observations']==2113536 and row['element_observations']==270532608
                writers['per_layer_summary'].write(row);all_summary.append(row);summ[method,layer]=row
                for head in range(32):
                    writers['per_layer_head_summary'].write({**base,'head_idx':head,**summarize(m[:,head]),
                                                             'valid_samples':4,'sampled_sample_token_events':516})
                for sample_idx,sample in enumerate(SAMPLES):
                    part=m[sample_idx*129:(sample_idx+1)*129]
                    writers['per_sample_layer_summary'].write({**base,'sample_id':sample,**summarize(part),
                                                               'prompt_tokens':prompts[sample],'sampled_token_events':129})
                    for pos,metrics in zip(positions,part):
                        writers['per_token_layer_summary'].write({**base,'sample_id':sample,'token_idx':int(pos),
                            'full_input_position_one_based':prompts[sample]+int(pos)+1,**summarize(metrics)})
            if exp=='A':
                for candidate,reference in PAIRS:
                    for sid_index,sample in [(None,'ALL4')]+list(enumerate(SAMPLES)):
                        a=values[candidate] if sid_index is None else values[candidate][sid_index*129:(sid_index+1)*129]
                        b=values[reference] if sid_index is None else values[reference][sid_index*129:(sid_index+1)*129]
                        writers['A_paired_group_effects'].write({'experiment':'A','model_layer_id':layer,'candidate':candidate,
                            'reference':reference,'sample_id':sample,'sample_count':4 if sid_index is None else 1,
                            'pair_unit':'sample_token_layer_head_value_same_FP_State',**paired(a,b)})
            high_groups(exp,layer,values,peaks,positions,writers['high_outlier_groups'])
            for candidate,reference in PAIRS:
                a=summ[candidate,layer];b=summ[reference,layer]
                dp=a['peak_rms_median']-b['peak_rms_median'];dq=a['qdq_rel_l2_median']-b['qdq_rel_l2_median'];dt=a['out3_element_fraction']-b['out3_element_fraction']
                rr={'experiment':exp,'model_layer_id':layer,'candidate':candidate,'reference':reference,
                    'median_peak_candidate_minus_reference':dp,'median_QDQ_candidate_minus_reference':dq,'out3_fraction_candidate_minus_reference':dt,
                    'peak_vs_error_same_sign':sign(dp)==sign(dq),'out3_vs_error_same_sign':sign(dt)==sign(dq),
                    'peak_error_strict_opposite':sign(dp)*sign(dq)<0,'out3_error_strict_opposite':sign(dt)*sign(dq)<0}
                ranking.append(rr);writers['metric_order_comparison'].write(rr)
            distributions(exp,layer,summ,hist,figure_out);key_maps(exp,layer,values,figure_out)
            del values,peaks;gc.collect()
        cross_layer(exp,summ,figure_out);relationship(exp,summ,figure_out)
    historic=historical_comparison(all_summary,writers['historical_native_hadamard_comparison'])
    for w in writers.values():w.close()
    counts={n:w.count for n,w in writers.items()}
    expected={'per_layer_summary':192,'per_layer_head_summary':6144,'per_sample_layer_summary':768,
              'per_token_layer_summary':99072,'A_paired_group_effects':720,'high_outlier_groups':1920,'metric_order_comparison':288,
              'historical_native_hadamard_comparison':1152}
    assert counts==expected,(counts,expected)
    jd(out/'FINDINGS.json',findings(all_summary,ranking));jd(out/'HISTORICAL_COMPARISON.json',historic)
    assert len(list(figs.rglob('*.png')))==100
    for row in source_rows:assert sha(ROOT/row['path'])==row['sha256']
    jd(out/'ANALYSIS_COMPLETE.json',{'status':'PASS','rows':counts,'PNG_count':100,'layer_count':24,'methods':METHODS,
        'legacy_exact_quantile_source_sha256':sha(Path(__file__).with_name('prior_analysis_source.py')),
        'analysis_sources':source_rows,'elapsed_seconds':time.monotonic()-start,
        'peak_host_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'statistical_scope':'4 samples only; group/token/head/value observations are dependent repeated measurements; no significance claim'})
    print('ANALYSIS_COMPLETE_PASS',flush=True)

if __name__=='__main__':main()
