from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer


ROOT = '/home/yid/yid-efs/ColBERT'
EXP_NAME = 'from_qa_mpnet_data'
SUFFIX = 'nodeid2claudesummary_231122'

if __name__=='__main__':
    with Run().context(RunConfig(nranks=4, experiment=EXP_NAME)):

        config = ColBERTConfig(
            nbits=2,
            root=f'{ROOT}/experiments/{EXP_NAME}',
        )
        indexer = Indexer(
            checkpoint=f'{ROOT}/experiments/{EXP_NAME}/2023-11/27/22.55.55/checkpoints/colbert',
            config=config,
        )
        indexer.index(
            name=f'{SUFFIX}.mpnet_train.nbits=2', 
            collection=f'{ROOT}/data/eval/{SUFFIX}.tsv',
            overwrite=True
        )
