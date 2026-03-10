import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from typing import List, Tuple


class BM25TextRetriever:
    def __init__(self):
        self.bm25 = None
        self.file_paths = []
        self.captions = []
    
    @classmethod
    def load_documents(cls, documents: str | List[str]) -> 'BM25TextRetriever':
        retriever = cls()
        tokenized_docs = []
        if isinstance(documents, str):
            documents = [documents]

        for doc_path in documents:
            if doc_path.endswith('.tsv'):
                doc_data = pd.read_csv(doc_path, sep='\t', engine='pyarrow')
                for _, row in tqdm(doc_data.iterrows(), desc=f"Loading documents from {doc_path}"):
                    text = row['text']
                    doc_id = row['id']
                    retriever.file_paths.append(doc_id)
                    tokenized_docs.append(retriever._tokenize(text))

            elif os.path.isdir(doc_path):
                doc_list = [doc for doc in os.listdir(doc_path) if doc.endswith('.json')]
                for doc in tqdm(doc_list, desc=f"Loading documents from {doc_path}"):
                    file_path = os.path.join(doc_path, doc)
                    tokenized_doc, file_path, captions = retriever._process_json_file_30sec(file_path)
                    retriever.file_paths.extend(file_path)
                    retriever.captions.extend(captions)
                    tokenized_docs.extend(tokenized_doc)
            
            elif os.path.isfile(doc_path):
                tokenized_doc, file_path, captions = retriever._process_json_file(doc_path)
                retriever.file_paths.extend(file_path)
                retriever.captions.extend(captions)
                tokenized_docs.extend(tokenized_doc)

            else:
                raise ValueError("Only TSV file or directory of TXT files are supported for loading documents.")

        print("Building BM25 index...")
        retriever.bm25 = BM25Okapi(tokenized_docs)

        return retriever
    
    def _tokenize(self, text: str) -> List[str]:
        return re.sub(r'[^\w\s]', '', text.lower()).split(" ")
    
    def _process_file(self, file_path: str) -> Tuple[List[str], str]:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return self._tokenize(text), file_path

    def _process_json_file(self, file_path: str) -> Tuple[List[str], str]:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            keys, captions = [], []
            for elem in data:
                for i, cap in enumerate(elem["caption"]):
                    keys.append(f"{elem['day']}-{elem['start']}-{elem['end']}_{i}")
                    captions.append("DAY" + str(elem['day']) + " " + str(elem['start'])[:2] + ":" + str(elem['start'])[2:4] + ":00 - " + str(elem['end'])[:2] + ":" + str(elem['end'])[2:4] + ":00\n" + cap["action"] + "\n" + cap["detail"])

        return [self._tokenize(caption) for caption in captions], keys, captions

    def _process_json_file_30sec(self, file_path: str) -> Tuple[List[str], str]:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            keys = list(data.keys())
            formatted_keys = [f"{x.split('_')[0]} {x.split('_')[3][:2]}:{x.split('_')[3][2:4]}:{x.split('_')[3][4:6]} {x.split('_')[2].capitalize()}" for x in keys]
            captions = list(data.values())
        return [self._tokenize(key + ", " + caption) for key, caption in zip(formatted_keys, captions)], keys, captions

    def retrieve(self, query: str, top_k: int = 5, return_scores: bool = False) -> List[str] | List[Tuple[int, float, str]] | List[Tuple[int, float, str, str]]:
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        if return_scores:
            return [self.file_paths[idx] for idx in top_indices], [self.captions[idx] for idx in top_indices], [scores[idx] for idx in top_indices]
        return [self.file_paths[idx] for idx in top_indices], [self.captions[idx] for idx in top_indices]
    
    def save_vectorized_format(self, filepath: str):
        print(f"Saving BM25 index to {filepath}...")
        data = {
            'file_paths': self.file_paths,
            'captions': self.captions,
            'bm25': self.bm25
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"BM25 index saved successfully at {filepath}.")
    
    @classmethod
    def load_vectorized_format(cls, filepath: str) -> 'BM25TextRetriever':
        # print(f"Loading BM25TextRetriever from {filepath}...")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        retriever = cls()
        retriever.file_paths = data['file_paths']
        retriever.bm25 = data['bm25']
        retriever.captions = data['captions']
        
        return retriever


if __name__ == "__main__":

    ### index 10min shared memory

    documents = ["data/10min_shared_memory.json"]
    
    retriever = BM25TextRetriever.load_documents(documents)
    retriever.save_vectorized_format("data/10min_bm25.pkl")
    
    loaded_retriever = BM25TextRetriever.load_vectorized_format("data/10min_bm25.pkl")
    with open("data/MA-EgoQA.json", "r") as f:
        data = json.load(f)
    for elem in tqdm(data, desc="Retrieving captions"):
        question = elem['question']
        results, captions = loaded_retriever.retrieve(question, top_k=20)
        elem['bm25'] = [{'id': result, 'caption': caption} for result, caption in zip(results, captions)]
    with open("data/MA-EgoQA_bm25.json", "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    #### index 30sec agent captions

    documents = ["data/caption/30sec"]
    retriever = BM25TextRetriever.load_documents(documents)
    retriever.save_vectorized_format("data/30sec_bm25.pkl")
