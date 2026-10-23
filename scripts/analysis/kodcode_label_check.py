import json,random,sys,concurrent.futures
sys.path.insert(0,'/path/to/iar_code/scripts/analysis')
import label_seed_rollouts as L
from iar_campaign.reward_alignment import _score_payload
from pathlib import Path
if __name__=='__main__':
    A=Path('/path/to/runs/evaluation_audits_and_reproducibility/12_final_test_analysis_reconciliation/correlations/nonofficial_reward_alignment_attempt001')
    arch={r['call_id']:r for r in L.read_jsonl(A/'rows_kodcode_p_own.jsonl')}
    s23=[r for r in L.read_jsonl(L.reward_log('kodcode','seed23')) if r.get('status')=='ok']
    sample=random.Random(23).sample(s23,200)
    ex={str(r['id']):r for r in L.read_jsonl(L.DATASETS['kodcode'])}
    pay=[('kodcode',ex[str(r['task_id'])],L.candidate_text(r),60,None) for r in sample]
    with concurrent.futures.ThreadPoolExecutor(2) as pool: sc=list(pool.map(_score_payload,pay))
    mis=[(r['call_id'],arch[r['call_id']]['correct'],int(bool(s['correct'])),s.get('returncode'),s.get('timed_out'),(s.get('stdout_tail') or '')[-160:].replace('\n','|')) for r,s in zip(sample,sc) if int(bool(s['correct']))!=arch[r['call_id']]['correct']]
    print(len(mis))
    for m in mis: print(m)
