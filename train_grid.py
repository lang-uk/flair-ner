import argparse
from pathlib import Path

import flair
import torch
from flair.datasets import ColumnCorpus
from flair.data import Corpus

from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter, SequenceTaggerParamSelector
from flair.embeddings import (
    StackedEmbeddings,
    FlairEmbeddings,
    TokenEmbeddings,
    CharacterEmbeddings,
)


def choochoochoo(embeddings: TokenEmbeddings) -> None:
    # define columns
    columns = {0: "text", 1: "ner"}

    # this is the folder in which train, test and dev files reside
    data_folder = "./fixed-split"

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns, train_file="train.iob", test_file="test.iob")

    search_space = SearchSpace()
    search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[embeddings()])

    search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[64, 128, 256])
    search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
    search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
    search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.25])
    search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8, 16, 32])

    param_selector = SequenceTaggerParamSelector(
        corpus, "ner", base_path=Path("./ner-tests/flair.grid.charemb/"), training_runs=2, max_epochs=150
    )

    # start the optimization
    param_selector.optimize(search_space, max_evals=15)


if __name__ == "__main__":
    flair.device = torch.device("cuda:1")

    parser = argparse.ArgumentParser(
        description="""That is the hyperparam opt trainer that can accept a base dir with embeddings"""
    )

    parser.add_argument("--embeddings-dir", type=Path, help="Path base dir with embeddings", default=Path("/data/"))

    args = parser.parse_args()

    choochoochoo(
        lambda: StackedEmbeddings(
            [
                CharacterEmbeddings(path_to_char_dict=args.embeddings_dir / "flair/uk/flair_dictionary.pkl"),
                FlairEmbeddings(args.embeddings_dir / "flair/uk/backward/best-lm.pt"),
                FlairEmbeddings(args.embeddings_dir / "flair/uk/forward/best-lm.pt"),
            ]
        )
    )
