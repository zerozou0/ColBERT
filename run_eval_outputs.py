import os

import pandas as pd
import pdb


ROOT = '/home/yid/yid-efs/ColBERT'
# EXP_NAME = 'from_qa_mpnet_data'
EXP_NAME = 'using_colbertv2'
# EXP_DATE_PATH = '2023-11/27/22.55.55'
EXP_DATE_PATH = '2023-11/28/23.06.51'
EVAL_SET_NAME = 'ir_test_data'
EVAL_CONFIG_NAME = 'nbits=2'
INDEX_NAME = 'nodeid2claudesummary_231122'

if not os.path.isfile(f'{ROOT}/data/eval/qid2pid_gt.json'):
    qid2gt_df = pd.read_json(f'{ROOT}/data/eval/qid2gt_{EVAL_SET_NAME}.json', orient='records', lines=False)
    nodeid2pid_df = pd.read_json(f'{ROOT}/data/eval/nodeid2pid_{INDEX_NAME}.json', orient='records', lines=False)
    qid2pid_df = pd.merge(qid2gt_df, nodeid2pid_df, on='nodeid', how='left')
    qid2pid_df.to_json(f'{ROOT}/data/eval/qid2pid_gt.json', orient='records')
else:
    qid2pid_df = pd.read_json(f'{ROOT}/data/eval/qid2pid_gt.json', orient='records', lines=False)

qid2pid_df['eval_rank'] = None

eval_output_df = pd.read_csv(
    f'{ROOT}/experiments/{EXP_NAME}/{EXP_DATE_PATH}/eval_results/'
    f'{EXP_NAME}.{EVAL_SET_NAME}.{EVAL_CONFIG_NAME}.ranking.tsv',
    sep='\t', header=None
).rename(
    columns={0: 'qid', 1: 'pid', 2: 'rank', 3: 'score'}
)
# pdb.set_trace()  # Check column value types & join

for i, r in qid2pid_df.iterrows():
    qid = r.qid
    pid_gt = r.pid
    if (not isinstance(qid, int)) or (not isinstance(pid_gt, int)):
        print(f'Missing value: qid {qid} pid_gt {pid_gt}')
        continue
    
    ranks_df = eval_output_df[eval_output_df.qid == qid]
    if len(ranks_df) == 0:
        print(f'Missing eval output: qid {qid} pid_gt {pid_gt}')
        continue
    ranks_df = ranks_df[ranks_df.pid == pid_gt]['rank']
    if len(ranks_df) == 1:
        qid2pid_df.at[i, 'eval_rank'] = ranks_df.iloc[0]
    else:
        qid2pid_df.at[i, 'eval_rank'] = 99
    # pdb.set_trace()

# Write eval output
qid2pid_df.to_json(
    f'{ROOT}/experiments/{EXP_NAME}/{EXP_DATE_PATH}/eval_results/'
    f'{EXP_NAME}.{EVAL_SET_NAME}.{EVAL_CONFIG_NAME}.gt_ranking.json'
)

# print stats
evaluated_df = qid2pid_df[~qid2pid_df.eval_rank.isnull()]
for k in range(1, 7):
    print(f'Recall @ {k}: {len(evaluated_df[evaluated_df.eval_rank <= k]) / len(evaluated_df)}')

pdb.set_trace()
