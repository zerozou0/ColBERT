from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer


ROOT = '/home/yid/yid-efs/ColBERT'
EXP_NAME = 'from_qa_mpnet_data'
SUFFIX = 'mpnet_231127'
TRIPLES_FILE_PATH = f'{ROOT}/data/triples_{SUFFIX}.tsv'
QUERIES_FILE_PATH = f'{ROOT}/data/queries_{SUFFIX}.tsv'
ARTICLES_FILE_PATH = f'{ROOT}/data/collections_{SUFFIX}.tsv'

if __name__=='__main__':
    with Run().context(RunConfig(
        nranks=4, experiment=EXP_NAME, 
        root=f'{ROOT}/experiments',
    )):

        config = ColBERTConfig(
            bsize=32,
            resume=True,
            lr=1.5e-6,
            kmeans_niters=10,  # the number of iterations of k-means clustering; 4 is a good and fast default. Consider larger numbers for small datasets
        )
        trainer = Trainer(
            triples=TRIPLES_FILE_PATH,
            queries=QUERIES_FILE_PATH,
            collection=ARTICLES_FILE_PATH,
            config=config,
        )

        checkpoint_path = trainer.train(
            # checkpoint='colbert-ir/colbertv2.0'  # Epoch 1
            # checkpoint='2023-11/29/01.40.20'  # Epoch 2
            checkpoint='2023-11/29/02.06.32'  # Epoch 3
        )

        print(f"Saved checkpoint to {checkpoint_path}...")