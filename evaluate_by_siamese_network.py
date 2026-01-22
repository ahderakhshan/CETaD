import argparse
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArgumentManager:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument("--data", type=str, nargs='+', default=["SNLI", "MNLI"],
                                 help="datasets used in model training")
        self.parser.add_argument("--min_freq", type=int, default=10,
                                 help="minimum frequency of a term in dataset to consider")
        self.parser.add_argument("--output_path", type=str, default="./pmi",
                                 help="output path of pmi.pkl and terms.pkl to store")

    def parse(self):
        return self.parser.parse_args()