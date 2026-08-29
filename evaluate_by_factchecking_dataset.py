import logging
import argparse
import torch
from siamese_network.siames_model import SiameseMLMClassifier
from extract_contradiction.ContradictionExtractor import RepresentativePointSelector, ContradictionExtractor
import pickle
from Evaluators.Evaluators import EvaluateByFactCheckingCorpus


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArgumentManager:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument("--data_path", type=str, default="./created_corpus",
                                 help="path to created corpus files which must contains 3 .pkl files."
                                      " change utils.read_input_data function based on your data type")
        self.parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/checkpoint.pt",
                                 help="siamese language model that fine tuned to generate NLI specific embeddings")
        self.parser.add_argument("--tokenizer_max_length", type=int, default=128,
                                 help="max tokens that tokenizer processed")
        self.parser.add_argument("--MLP_no_layers", type=int, default=3,
                                 help="number of layers of the MLP part of siamese")
        self.parser.add_argument("--no_unfreeze_layer", type=int, default=6,
                                 help="number of last layers which are not freeze at MLM")
        self.parser.add_argument("--MLP_number_of_neurons", type=int, nargs="+", default=[128, 64],
                                 help="number of neurons for each layer of MLP. it must be a list with length equal to"
                                      " --MLP_no_layers - 1. last layer number of neurons is 3 because the problem is 3"
                                      " class classification")
        self.parser.add_argument("--contradiction_ppmi_path", type=str, default="./static/contradiction_ppmi.pkl",
                                 help="path to pkl file which contains ppmi of each word with contradiction class")
        self.parser.add_argument("--contradiction_terms_path", type=str, default="./static/temrs_contradiction.pkl",
                                 help="path pkl file which contains term index to find ppmi with contradiction")
        self.parser.add_argument("--entailment_ppmi_path", type=str, default="./static/entailment_ppmi.pkl",
                                 help="path to pkl file which contains ppmi of each word with entailment class")
        self.parser.add_argument("--entailment_terms_path", type=str, default="./static/temrs_entailment.pkl",
                                 help="path pkl file which contains term index to find ppmi with entailment")
        self.parser.add_argument("--e_th", type=float, default=0.7, help="e_th parameter of Method")
        self.parser.add_argument("--c_th", type=float, default=0.7, help="c_th parameter of Method")
        self.parser.add_argument("--rep_point_optimizer_iterations", type=int, default=200,
                                 help="number of optimizer iterations to find R_e and R_c")
        self.parser.add_argument("--rep_point_optimizer_learning_rate", type=float, default=0.005,
                                 help="optimizer learning rate at representative point selection")
        self.parser.add_argument("--alpha", type=int, default=1, hlep="alph in final representative point selection")
        self.parser.add_argument("--beta", type=int, default=1, help="beta in final representative point selection")
        self.parser.add_argument("--sort_by_similarity", type=bool, default=False)
        self.parser.add_argument("--similarity_sort_model", type=str, default="Jaccard")
        self.parser.add_argument("--sort_by_impc", type=bool, default=False)
        self.parser.add_argument("--sort_by_impe", type=bool, default=False)
        self.parser.add_argument("--top_n", type=int, default=100, help="evaluate on top_n retrieval after sorting")

    def parse(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    arg_manager = ArgumentManager()
    args = arg_manager.parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    siamese_model = SiameseMLMClassifier(
        no_unfreeze_layer=args.no_unfreeze_layer,
        mlp_number_of_neurons=args.MLP_number_of_neurons,
    )
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    siamese_model.load_state_dict(checkpoint['model_state_dict'])
    siamese_model.eval()
    siamese_model.to(device)
    representative_point_selector = RepresentativePointSelector(siamese_model=siamese_model,
                                                                optimizer_iterations=args.rep_point_optimizer_iterations,
                                                                optimizer_learning_rate=args.rep_point_optimizer_learning_rate,
                                                                alpha=args.alpha, beta=args.beta)
    with open(f"{args.data_path}/all_sentences.pkl", "rb") as f:
        sentences = pickle.load(f)
    with open(f"{args.data_path}/all_names.pkl", "rb") as f:
        names = pickle.load(f)
    with open(f"{args.data_path}/all_contradictory_relations.pkl", "rb") as f:
        contradictory_relation = pickle.load(f)
    contradiction_extractor = ContradictionExtractor(
        siamese_model=siamese_model,
        tokenizer_max_length=args.tokenizer_max_length,
        sentences=sentences,
        representative_point_selector=representative_point_selector,
        terms_contradiction_path=args.contradiction_terms_path,
        ppmi_contradiction_path=args.contradiction_ppmi_path,
        terms_entailment_path=args.entailment_terms_path,
        ppmi_entailment_path=args.entailment_ppmi_path,
        e_th=args.e_th,
        c_th=args.c_th,
        similarity_sort=args.sort_by_similarity,
        similarity_sort_model=args.similarity_sort_model,
        impc_sort=args.sort_by_impc,
        impe_sort=args.sort_by_impe
    )
    logger.info("start extracting contradictory sentences and probabilites")
    points_data_status = contradiction_extractor.construct_entailment_and_contradiction_sets()
    extracted_pairs = contradiction_extractor.extract_contradictory_pairs()
    all_extractions, true_extractions = EvaluateByFactCheckingCorpus(sentences, names, contradictory_relation, points_data_status, args.top_n).evaluate()
    logger.info(f"""Results:
    all extractions: {all_extractions}
    true extractions: {true_extractions}
    n: {args.top_n}""")

