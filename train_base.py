import argparse
from pathlib import Path

import flair
import torch

from flair.datasets import ColumnCorpus
from flair.data import Corpus
from flair.models import SequenceTagger
from flair.embeddings import (
    FastTextEmbeddings,
    StackedEmbeddings,
    FlairEmbeddings,
    TokenEmbeddings,
    CharacterEmbeddings,
)
from flair.trainers import ModelTrainer
from torch.optim.adam import Adam


def choochoo(
    hidden_size: int,
    rnn_layers: int,
    embeddings: TokenEmbeddings,
    config_name: str,
    optimize_lr: bool = False,
    learning_rate: float = 0.1,
    mini_batch_size: int = 32,
    dropout: float = 0.0,
) -> None:
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
        embeddings=embeddings(),
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True,
        dropout=dropout,
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
            base_path=results_path,
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
            checkpoint=True,
            train_with_dev=True,
            monitor_test=True,
            max_epochs=150,
            embeddings_storage_mode="gpu",
            use_tensorboard=True,
            tensorboard_log_dir=tensorboard_path,
            tensorboard_comment=f"Flair UK: {config_name}",
        )
    else:
        # 7. start training
        if optimize_lr:
            trainer.find_learning_rate(results_path, Adam)
        else:
            trainer.train(
                results_path,
                learning_rate=learning_rate,
                mini_batch_size=mini_batch_size,
                checkpoint=True,
                train_with_dev=True,
                monitor_test=True,
                max_epochs=150,
                embeddings_storage_mode="gpu",
                use_tensorboard=True,
                tensorboard_log_dir=tensorboard_path,
                tensorboard_comment=f"Flair UK: {config_name}",
            )


if __name__ == "__main__":
    flair.device = torch.device("gpu")

    parser = argparse.ArgumentParser(
        description="""That is the simple trainer that can accept a base dir
with embeddings and the name of the config to train the model"""
    )

    parser.add_argument("--embeddings-dir", type=Path, help="Path base dir with embeddings", default=Path("/data/"))
    parser.add_argument("config")

    args = parser.parse_args()

    config = {
        "fb.fasttext": {
            "embeddings": lambda: FastTextEmbeddings(args.embeddings_dir / "fasttext/uk/cc.uk.300.bin"),
            "hidden_size": 256,
            "rnn_layers": 1,
        },
        "uk.flairembeddings": {
            "embeddings": lambda: StackedEmbeddings(
                [
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/backward/best-lm.pt"),
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/forward/best-lm.pt"),
                ]
            ),
            "hidden_size": 256,
            "rnn_layers": 1,
        },
        "uk.flairembeddings.lr0.5": {
            "embeddings": lambda: StackedEmbeddings(
                [
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/backward/best-lm.pt"),
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/forward/best-lm.pt"),
                ]
            ),
            "hidden_size": 256,
            "rnn_layers": 1,
            "learning_rate": 0.5,
        },
        "uk.flairembeddings.find_lr": {
            "embeddings": lambda: StackedEmbeddings(
                [
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/backward/best-lm.pt"),
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/forward/best-lm.pt"),
                ]
            ),
            "hidden_size": 256,
            "rnn_layers": 1,
            "optimize_lr": True,
        },
        "uk.flairembeddings.fasttext": {
            "embeddings": lambda: StackedEmbeddings(
                [
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/backward/best-lm.pt"),
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/forward/best-lm.pt"),
                    FastTextEmbeddings(args.embeddings_dir / "fasttext/uk/cc.uk.300.bin"),
                ]
            ),
            "hidden_size": 256,
            "rnn_layers": 1,
        },
        "uk.flairembeddings.x2": {
            "embeddings": lambda: StackedEmbeddings(
                [
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/backward/best-lm.pt"),
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/forward/best-lm.pt"),
                ]
            ),
            "hidden_size": 256,
            "rnn_layers": 2,
        },
        "uk.flairembeddings.charembeddings": {
            "embeddings": lambda: StackedEmbeddings(
                [
                    CharacterEmbeddings(),
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/backward/best-lm.pt"),
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/forward/best-lm.pt"),
                ]
            ),
            "hidden_size": 256,
            "rnn_layers": 1,
        },
        "uk.flairembeddings.large": {
            "embeddings": lambda: StackedEmbeddings(
                [
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/backward/best-lm.pt"),
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/forward/best-lm.pt"),
                ]
            ),
            "hidden_size": 512,
            "rnn_layers": 1,
        },
        "uk.flairembeddings.xlarge": {
            "embeddings": lambda: StackedEmbeddings(
                [
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/backward/best-lm.pt"),
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/forward/best-lm.pt"),
                ]
            ),
            "hidden_size": 1024,
            "rnn_layers": 1,
        },
        "uk.flairembeddings.champ": {
            "embeddings": lambda: StackedEmbeddings(
                [
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/backward/best-lm.pt"),
                    FlairEmbeddings(args.embeddings_dir / "flair/uk/forward/best-lm.pt"),
                ]
            ),
            "hidden_size": 128,
            "rnn_layers": 2,
            "learning_rate": 0.1,
            "mini_batch_size": 32,
            "dropout": 0.3380078963015963,
        },
    }

    choochoo(config_name=args.config, **config[args.config])
