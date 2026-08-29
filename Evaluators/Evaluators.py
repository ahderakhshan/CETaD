import torch
import torch.nn.functional as F
from openai import OpenAI


class EvaluateBySiamese:
    def __init__(self, siamese_model, retrieved_pairs, embeddings):
        self.siamese_model = siamese_model
        self.retrieved_pairs = retrieved_pairs
        self.embeddings = embeddings

    def evaluate(self):
        sum_e, sum_c, sum_n = 0, 0, 0
        for pairs in self.retrieved_pairs:
            entailment_index, contradiction_index = pairs[0], pairs[1]
            entailment_embedding, contradiction_embedding = self.embeddings[entailment_index],\
                                                              self.embeddings[contradiction_index]
            abs_diff = torch.abs(entailment_embedding - contradiction_embedding)
            elem_mult = entailment_embedding * contradiction_embedding
            combined = torch.cat([entailment_embedding, contradiction_embedding, abs_diff, elem_mult], dim=0)
            logits = self.siamese_model.classifier(combined)
            probs = F.softmax(logits, dim=-1)
            sum_e += probs[0]
            sum_n += probs[1]
            sum_c += probs[2]
        all_pairs_no = self.retrieved_pairs.shape[0]

        return sum_e / all_pairs_no, sum_n / all_pairs_no, sum_c / all_pairs_no


class EvaluateByFactCheckingCorpus:
    def __init__(self, all_sentences, all_names, all_contradictory_relations, points_data_status, extracted_pairs, n):
        self.all_sentences = all_sentences
        self.all_names = all_names
        self.all_contradictory_relations = all_contradictory_relations
        self.points_data_status = points_data_status
        self.extracted_pairs = extracted_pairs[:n]

    def evaluate(self):
        true_positive_counter = 0
        all_counter = 0
        for i in self.points_data_status.keys():
            entailments = self.points_data_status[i]["entailment"]
            contradictions = self.points_data_status[i]["contradiction"]
            all_counter += len(entailments) * len(contradictions)
            for en in entailments:
                for co in contradictions:
                    if "claim" in self.all_names[co]:
                        if self.all_contradictory_relations.get(self.all_names[co]):
                            if self.all_names[en] == self.all_contradictory_relations[self.all_names[co]]:
                                if ((en, co)) in self.extracted_pairs:
                                    true_positive_counter += 1
                    elif "claim" in self.all_names[en]:
                        if self.all_contradictory_relations.get(self.all_names[en]):
                            if self.all_names[co] == self.all_contradictory_relations[self.all_names[en]]:
                                if ((en, co)) in self.extracted_pairs:
                                    true_positive_counter += 1
        return all_counter, true_positive_counter


class EvaluateByLLM:
    def __init__(self, all_sentences, extracted_pairs, model_name, api_key):
        self.all_sentences = all_sentences
        self.extracted_pairs = extracted_pairs
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = "Your task is to determine whether there is contradiction or inconsistency between following two sentences which are extracted from medical documents. answer only with yes or no. Do not provide any explanation at all."

    def evaluate(self):
        all_response = []
        for e, c in self.extracted_pairs:
            entailment_sentence = self.all_sentences[e]
            contradiction_sentence = self.all_sentences[c]
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Sentence1: {entailment_sentence} \n Sentence2: {contradiction_sentence}"}
            ]
            response = self.gpt_api_call(messages)
            all_response.append(response)
        return all_response

    def gpt_api_call(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False,
            temperature=0
        )
        return response.choices[0].message.content



