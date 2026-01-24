import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import tqdm
import logging
import torch.nn.functional as F
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


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
    def __init__(self, siamese_model, embeddings, sentences, representative_point_selector: RepresentativePointSelector,
                 terms_pickle_path, pmi_pickle_path, output_path, terms_contradiction_path, ppmi_contradiction_path,
                 terms_entailment_path, ppmi_entailment_path, e_th=0.5, c_th=0.5, similarity_sort=False,
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
        self.terms_contradiction = np.load(terms_contradiction_path, allow_pickle=True)
        self.ppmi_contradiction = np.load(ppmi_contradiction_path, allow_pickle=True)
        self.terms_entailment = np.load(terms_entailment_path, allow_pickle=True)
        self.ppmi_entailment = np.load(ppmi_entailment_path, allow_pickle=True)
        self.impc_sort = impc_sort
        self.impe_sort = impe_sort
        self.impc_cache = {}
        self.impe_cache = {}
        self.stemmr = PorterStemmer()
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

    @staticmethod
    def jaccard_sim(text1, text2):
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return float(intersection) / union

    def lemmatize_text(self, sentence):
        words = word_tokenize(sentence)
        stems = [self.stemmr.stem(w) for w in words]
        result = ""
        for stem in stems:
            result += stem + " "
        return result.strip()

    def compute_impc(self, sentence, c_index):
        try:
            return self.impc_cache[c_index]
        except:
            text = self.lemmatize_text(sentence)
            words = text.split(" ")
            sum_pmi = 0
            for word in words:
                try:
                    sum_pmi += self.ppmi_contradiction[np.where(self.terms_contradiction == word)[0]][0]
                except:
                    sum_pmi += 1
            self.impc_cache[c_index] = len(words) / sum_pmi
            return len(words) / sum_pmi

    def compute_impe(self, sentence, e_index):
        try:
            return self.impe_cache[e_index]
        except:
            text = self.lemmatize_text(sentence)
            words = text.split(" ")
            sum_pmi = 0
            for word in words:
                try:
                    sum_pmi += self.ppmi_entailment[np.where(self.terms_entailment == word)[0]][0]
                except:
                    sum_pmi += 1
            self.impc_cache[e_index] = len(words) / sum_pmi
            return len(words) / sum_pmi

    def _sort_pairs(self, pairs):
        if not (self.similarity_sort or self.impe_sort or self.impc_sort):
            return pairs
        e_indexes = [pair[0] for pair in pairs]
        c_indexes = [pair[1] for pair in pairs]
        if self.similarity_sort and self.similarity_sort_model != "Jaccard":
            sentence_embeddings = self.similarity_sort_model.encode(self.sentences, show_progress_bar=True, convert_to_tensor=True)
            i_idx = torch.tensor(e_indexes, device=self.device)
            j_idx = torch.tensor(c_indexes, device=self.device)
            emb_i = sentence_embeddings[i_idx]
            emb_j = sentence_embeddings[j_idx]
            sims = emb_i @ emb_j.T
        pairs_status = np.zeros((len(pairs), 5)) # entailment_index, contradiction_index, similarity, impc, impe
        pair_counter = 0
        for e_counter, e_index in enumerate(e_indexes):
            for c_counter, c_index in enumerate(c_indexes):
                similarity, impc, impe = 0, 0, 0
                if self.similarity_sort and self.similarity_sort_model != "Jaccard":
                    similarity = sims[e_counter, c_counter].item()
                elif self.similarity_sort_model:
                    similarity = self.jaccard_sim(self.sentences[e_index], self.sentences[c_index])

                if self.impc_sort:
                    impc = self.compute_impc(self.sentences[c_index])
                if self.impe_sort:
                    impe = self.compute_impe(self.sentences[e_index])
                pairs_status[pair_counter] = [e_index, c_index, similarity, impc, impe]

        return pairs_status

    def extract_contradictory_pairs(self):
        points_data_status = self.construct_entailment_and_contradiction_sets()
        pairs = set()
        for point, status in points_data_status.items():
            point_entailments = status["entailment"]
            point_contradictions = status["contradiction"]
            for e_number in point_entailments:
                for c_number in point_contradictions:
                    pairs.add((e_number, c_number))
        pairs_scores = self._sort_pairs(pairs)
        pairs = sorted(pairs_scores, key=lambda x: x[2] + x[3] + x[4], reverse=True)
        return pairs
