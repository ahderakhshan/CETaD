import argparse
import os
import pandas as pd
import logging
import pickle


class ArgumentManager:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument("--dataset_path", type=str, default="./check_covid_data",
                                 help="path to check covid data folder")
        self.parser.add_argument("--output_path", type=str, default="./created_corpus",
                                 help="path to corpus which contains contradictory pairs")

    def parse(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    arg_manager = ArgumentManager()
    args = arg_manager.parse()
    claim_path = os.path.join(args.dataset_path, "Check-COVID_all.jsonl")
    corpus_path = os.path.join(args.dataset_path, "corpus.jsonl")
    claim_data = pd.read_json(claim_path, lines=True)
    corpus_data = pd.read_json(corpus_path, lines=True)
    claim_data = claim_data[claim_data["label"] == "REFUTE"]
    claim_data = claim_data[claim_data["evidence_set"].map(len) == 1]

    all_sentences = []
    all_names = []
    contradictory_relations = {}

    for index, row in corpus_data.iterrows():
        for counter, sentence in enumerate(row["abstract"]):
            all_sentences.append(sentence)
            all_names.append(f"{row['cord_id']}_{counter}")

    for index, row in claim_data.iterrows():
        if row["label"] == "REFUTE" and len(row["evidence_set"]) == 1:
            all_sentences.append(row["claim"])
            all_names.append(f"claim_{row['id']}")
            contradictory_relations[all_names[-1]] = f"{row['cord_id']}_{row['evidence_set'][0]['sent_index']}"

    logging.info(f"len all sentences in corpus = {len(all_sentences)}")
    logging.info(f"number of contradiction pairs = {len(list(contradictory_relations.keys()))}")

    all_sentences_path = os.path.join(args.output_path, "all_sentences.pkl")
    all_names_path = os.path.join(args.output_path, "all_names.pkl")
    all_contradictory_relations_path = os.path.join(args.output_path, "all_contradictory_relations.pkl")
    with open(all_sentences_path, "wb") as f:
        pickle.dump(all_sentences, f)

    with open(all_names_path, "wb") as f:
        pickle.dump(all_names, f)

    with open(all_contradictory_relations_path, "wb") as f:
        pickle.dump(contradictory_relations, f)

    logging.info("complete!")
