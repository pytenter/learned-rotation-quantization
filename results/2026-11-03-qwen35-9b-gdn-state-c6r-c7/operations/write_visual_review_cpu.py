"""Record completed manual visual inspection with fresh matching bitmap hashes."""
from pathlib import Path
import hashlib
import json
import os

ROOT=Path('/data/zypan/experiments/QWEN_GDN_STATE_OUTLIER_FOUR_ROTATIONS_C6R_C7_20261103_V1')
assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
assert not (ROOT/'visual_qa/VISUAL_REVIEW.json').exists()
bitmap=json.loads((ROOT/'visual_qa/BITMAP_CHECKS.json').read_text())
layout=json.loads((ROOT/'operations/CPU_TICK_LAYOUT_RESULT.json').read_text())
assert bitmap['status']=='PASS' and bitmap['PNG_count']==100
assert layout['status']=='PASS_CPU_TICK_LAYOUT_V2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for row in bitmap['files']:assert sha(ROOT/row['path'])==row['sha256']
created=json.loads((ROOT/'analysis/REPORT_CREATED.json').read_text())
assert sha(ROOT/'REPORT_zh.md')==created['report_sha256']
assert sha(ROOT/'DATA_DICTIONARY_zh.md')==created['dictionary_sha256']
full=['figures/A/cross_layer_summary.png','figures/B/cross_layer_summary.png',
      'figures/A/L00_key_outlier_locations.png','figures/B/L30_key_outlier_locations.png',
      'figures/A/L00_distribution.png','figures/B/L30_distribution.png',
      'figures/A/outlier_error_relationship.png','figures/B/outlier_error_relationship.png']
contacts=['visual_qa/A_all50_contact_sheet.png','visual_qa/B_all50_contact_sheet.png']
result={'status':'PASS','reviewer':'Codex actual image inspection','main_PNG_count':100,
        'scope':'All100 via both contact sheets; representative full figures inspected for shared layout and labels',
        'full_figures_inspected':[{'path':p,'sha256':sha(ROOT/p)} for p in full],
        'all100_contact_sheets_inspected':[{'path':p,'sha256':sha(ROOT/p)} for p in contacts],
        'observations':['All24 GDN layers present in A and B, no blank or visibly corrupted panels',
            'All32 heads and128 Keys in heatmaps, common colour ranges and different-basis warning visible',
            'Final signed/absolute symlog tick labels readable, common bins and full ranges unchanged',
            'All4 methods present in cross-layer and distribution figures with units and A/B scopes',
            'Relationship labels readable after retaining preset Native labels only; all data points preserved',
            'Main report and dictionary hashes unchanged from original report owner'],
        'original_layout_retained':True,'raw_data_and_original_sources_unchanged':True,
        'new_generation_or_GPU_work':False}
with (ROOT/'visual_qa/VISUAL_REVIEW.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':'PASS_ACTUAL_VISUAL_REVIEW','main_PNG_count':100}),flush=True)
