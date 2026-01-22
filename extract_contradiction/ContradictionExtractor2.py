import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import tqdm
import logging
import torch.nn.functional as F


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RepresentativePointSelector:
    def __init__(self, siamese_model, optimizer_iterations, optimizer_learning_rate, alpha, beta, embeddings):
        self.siamese_model = siamese_model
        self.siamese_model.eval()
        self.optimizer_iterations = optimizer_iterations
        self.optimizer_learning_rate = optimizer_learning_rate
        self.alpha = alpha
        self.beta = beta
        self.embeddings = embeddings

    def compute_r(self, class_number):
        mean_vec = self.embeddings.mean(dim=0).detach().clone()
        mean_vec_clone = mean_vec.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([mean_vec_clone], lr=self.optimizer_learning_rate)
        for step in tqdm(range(self.optimizer_iterations)):
            optimizer.zero_grad()
            rep_point = mean_vec_clone
            repeated_point = rep_point.repeat(len(self.embeddings), 1)
            abs_diff = torch.abs(self.embeddings - repeated_point)
            elem_mult = self.embeddings * repeated_point
            combined = torch.cat([self.embeddings, repeated_point, abs_diff, elem_mult], dim=1)
            logits = self.siamese_model.classifier(combined)
            probs = torch.softmax(logits, dim=-1)
            loss = -(probs[:, class_number].mean())  # maximize class probability
            loss.backward()
            optimizer.step()
        return mean_vec_clone

    @property
    def r_e(self):
        return self.compute_r(0)

    @property
    def r_c(self):
        return self.compute_r(2)

    @property
    def rep_point(self):
        point = (self.alpha * self.r_c + self.beta * self.r_e) / (self.alpha + self.beta)
        return point.unsqueeze(0)


class ContradictionExtractor2:
    def __init__(self, siamese_model, embeddings, sentences, representative_point_selector: RepresentativePointSelector, terms_pickle_path,
                 pmi_pickle_path, output_path, e_th=0.5, c_th=0.5, similarity_sort=False,
                 similarity_sort_model="Jaccard", impc_sort=False, impe_sort=False):
        self.siamese_model = siamese_model
        self.embeddings = embeddings
        self.sentences = sentences
        assert self.embeddings.shape[0] == len(sentences), f"len sentences and embeddings must be equal but" \
                                                           f" {self.embeddings.shape[0]} != {len(sentences)}"
        self.representative_point_selector = representative_point_selector
        self.representative_point = self.representative_point_selector.rep_point
        logger.info("representative point computed successfully")
        self.terms = np.load(terms_pickle_path, allow_pickle=True)
        self.pmi = np.load(pmi_pickle_path, allow_pickle=True)
        self.output_path = output_path
        self.e_th = e_th
        self.c_th = c_th
        self.similarity_sort = similarity_sort
        if self.similarity_sort and similarity_sort_model != "Jaccard":
            self.similarity_sort_model = SentenceTransformer(similarity_sort_model)
            logger.info("semantic similarity model loaded successfully")
        self.impc_sort = impc_sort
        self.impe_sort = impe_sort
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def compute_probability_with_rep_point(self):
        all_probs = []
        for point in self.representative_point:
            repeated_point = point.repeat(len(self.embeddings), 1).to(self.device)
            abs_diff = torch.abs(self.embeddings - repeated_point)
            elem_mult = self.embeddings * repeated_point
            combined = torch.cat([self.embeddings, repeated_point, abs_diff, elem_mult], dim=1)
            logits = self.siamese_model.classifier(combined)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)
        return all_probs

    def construct_entailment_and_contradiction_sets(self):
        all_probs = self.compute_probability_with_rep_point()
        points_data_status = {}
        point_counter = 0
        for prob in all_probs:
            points_data_status[point_counter] = {"entailment": [], "contradiction": []}
            counter = 0
            for instance in prob:
                if instance[0] >= self.e_th:
                    points_data_status[point_counter]["entailment"].append(counter)
                if instance[2] >= self.c_th:
                    points_data_status[point_counter]["contradiction"].append(counter)
                counter += 1
            point_counter += 1
        no_extracted_pairs = sum([len(points_data_status[i]["entailment"])*len(points_data_status[i]["contradiction"])
                                  for i in range(point_counter)])
        logger.info(f"Number of extracted sentence pairs: {no_extracted_pairs}")
        return points_data_status

    def extract_contradictory_pairs(self):
        points_data_status = self.construct_entailment_and_contradiction_sets()
        pairs = set()
        for point, status in points_data_status.items():
            point_entailments = status["entailment"]
            point_contradictions = status["contradiction"]
            for e_number in point_entailments:
                for c_number in point_contradictions:
                    pairs.add((e_number, c_number))
        return pairs
