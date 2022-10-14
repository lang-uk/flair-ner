import argparse
from typing import Union
from pathlib import Path

import flair
import torch

from flair.datasets import ColumnCorpus
from flair.data import Corpus
from flair.models import SequenceTagger
from flair.embeddings import FastTextEmbeddings, StackedEmbeddings, FlairEmbeddings, TokenEmbeddings
from flair.trainers import ModelTrainer


flair.device = torch.device("cpu")


class UKR_NER_CORP(ColumnCorpus):
    def __init__(
        self,
        data_folder: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ):
        """
        Initialize a preprocessed version of the Ukrainian Named Entity
        Recognition Corpus(lang-uk ner) dataset available
        from https://github.com/lang-uk/ner-uk

        # TODO: re-enable downloader as soon as we publish the corpus somewhere
        :param tag_to_bioes: NER by default, need not be changed
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if type(data_folder) == str:
            data_folder: Path = Path(data_folder)

        # column format
        columns = {0: "text", 1: "ner"}

        super(UKR_NER_CORP, self).__init__(
            data_folder,
            columns,
            # tag_to_bioes=tag_to_bioes,
            encoding="utf-8",
            in_memory=in_memory,
            **corpusargs,
        )


def choochoo(hidden_size: int, rnn_layers: int, embeddings: TokenEmbeddings, config_name: str) -> None:
    # define columns
    columns = {0: "text", 1: "ner"}

    # this is the folder in which train, test and dev files reside
    data_folder = "./fixed-split"

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns, train_file="train.iob", test_file="test.iob")

    # 2. what tag do we want to predict?
    tag_type = "ner"

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_label_dictionary(tag_type)

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=hidden_size,
        rnn_layers=rnn_layers,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True,
    )

    # 6. initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    results_path = Path(f"./ner-tests/{config_name}/")
    checkpoint_path = results_path / "checkpoint.pt"
    tensorboard_path = results_path / "tensorboard"
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists():
        trained_model = SequenceTagger.load(checkpoint_path)

        trainer.resume(
            trained_model,
            base_path=checkpoint_path.parent,
            learning_rate=0.1,
            mini_batch_size=32,
            checkpoint=True,
            train_with_dev=True,
            monitor_test=True,
            max_epochs=150,
            embeddings_storage_mode="cpu",
            use_tensorboard=True,
            tensorboard_log_dir=tensorboard_path,
            tensorboard_comment=f"Flair UK: {config_name}",
        )
    else:
        # 7. start training
        trainer.train(
            checkpoint_path.parent,
            learning_rate=0.1,
            mini_batch_size=32,
            checkpoint=True,
            train_with_dev=True,
            monitor_test=True,
            max_epochs=150,
            embeddings_storage_mode="cpu",
            use_tensorboard=True,
            tensorboard_log_dir=tensorboard_path,
            tensorboard_comment=f"Flair UK: {config_name}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""That is the node worker to compute fasttext """
        """vectors using different params, store obtained vectors on gdrive and update google spreadsheet"""
    )

    parser.add_argument("--embeddings-dir", type=Path, help="Path base dir with embeddings", default=Path("/data/"))
    parser.add_argument("config")

    args = parser.parse_args()

    config = {
        "fb.fasttext": {
            "embeddings": FastTextEmbeddings(args.embeddings_dir / "fasttext/uk/cc.uk.300.bin"),
            "hidden_size": 256,
            "rnn_layers": 1,
        },
        "uk.flairembeddings": {
            "embeddings": StackedEmbeddings(
                [
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/backward/best-lm.pt"),
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/forward/best-lm.pt"),
                ]
            ),
            "hidden_size": 256,
            "rnn_layers": 1,
        },
        "uk.flairembeddings.large": {
            "embeddings": StackedEmbeddings(
                [
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/backward/best-lm.pt"),
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/forward/best-lm.pt"),
                ]
            ),
            "hidden_size": 512,
            "rnn_layers": 1,
        },
        "uk.flairembeddings.xlarge": {
            "embeddings": StackedEmbeddings(
                [
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/backward/best-lm.pt"),
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/forward/best-lm.pt"),
                ]
            ),
            "hidden_size": 1024,
            "rnn_layers": 1,
        },
    }

    choochoo(config_name=args.config, **config[args.config])
