"""CPU-only edge cases and independent exact-quantile checks; no formal metrics produced."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import math
import numpy as np
from common import ROOT,paired,top_indices,summarize,jd,safe_json

def linear_quantile_python(values,p):
    x=sorted(float(v) for v in values if math.isfinite(float(v)))
    q=(len(x)-1)*p;lo=math.floor(q);hi=math.ceil(q)
    return x[lo]+(x[hi]-x[lo])*(q-lo)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
    reference=np.zeros((4,16),dtype=np.float64);candidate=reference.copy()
    for column in [4,7,9,10]:
        reference[:,column]=[0,1,2,3];candidate[:,column]=[0,2,1,3]
    result=paired(candidate,reference)
    for name in ['peak_rms','qdq_rel_l2','out3_count','out6_count']:
        assert result[name+'_candidate_lower']==1 and result[name+'_candidate_higher']==1 and result[name+'_equal']==2
        assert result[name+'_lower_fraction']==.25 and result[name+'_equal_fraction']==.5
        assert result[name+'_candidate_minus_reference_median']==0
        assert abs(result[name+'_candidate_minus_reference_p95']-.85)<1e-12
    assert result['qdq_reference_zero_pairs']==1 and result['qdq_ratio_valid_positive_reference_pairs']==3
    assert result['qdq_ratio_candidate_over_positive_reference_median']==1
    z=paired(np.zeros((4,16)),np.zeros((4,16)))
    assert z['qdq_reference_zero_pairs']==4 and z['qdq_ratio_valid_positive_reference_pairs']==0
    assert safe_json(z)['qdq_ratio_candidate_over_positive_reference_median'] is None
    assert np.array_equal(top_indices(np.array([2,4,4,4,1,4]),3),np.array([1,2,3]))
    source=ROOT/'gates/cpu_collector/A_ALL_B_NATIVE_H/A/groups/synthetic_layer_mapping__A_Native__L00.npz'
    with np.load(source) as data:metrics=data['metrics'].copy()
    summary=summarize(metrics);flat=metrics.reshape(-1,16)
    for metric,column in [('peak_rms',4),('qdq_rel_l2',7)]:
        for title,p in [('median',.5),('p95',.95),('p99',.99)]:
            expected=linear_quantile_python(flat[:,column],p)
            assert abs(summary[metric+'_'+title]-expected)<1e-12
    independent=math.sqrt(math.fsum(float(x) for x in flat[:,11]))/(math.sqrt(math.fsum(float(x) for x in flat[:,12]))+1e-12)
    assert abs(summary['pooled_qdq_relative_L2']-independent)<1e-14
    assert summary['undefined_kurtosis_groups']==int(flat[:,14].sum())>0
    assert summary['group_observations']==2*32*128
    jd(ROOT/'gates/CPU_ANALYSIS_QUALIFICATION.json',{'status':'PASS','synthetic_pairs':'1 lower,1 higher,2 equal; candidate-reference and ratio directions exact',
        'zero_denominator':'counted explicitly; undefined ratios serialize null, not zero','top_ties':'descending metric, then canonical flat coordinates',
        'actual_synthetic_collector_source':str(source),'independent_Python_sorted_linear_quantiles':'PASS',
        'independent_math_fsum_pooled_L2':'PASS','zero_variance_Pearson_undefined_count':'PASS','new_GPU_work':False,'formal_data_used':False})
    print('CPU_ANALYSIS_QUALIFICATION_PASS',flush=True)

if __name__=='__main__':main()
