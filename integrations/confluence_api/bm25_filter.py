import json
import logging
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO)

class BM25Filter:
    def __init__(self, documents):
        """
        Initiating BM25 filter.
        :param documents: list of documents where each document is a dictionary with a key "text""
        """
        self.documents = documents
        self.tokenized_docs = [self.tokenize(doc.get("text", "")) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        logging.info(f"BM25 The index is built for {len(self.documents)} Documents.")

    @staticmethod
    def tokenize(text):
        'Simplest text currentization (brokening by gaps and entry to lower register)).'
        return text.lower().split()

    def get_scores(self, query):
        """
        Computes the BM25 account for the request.
        :param query: row of request
        :return: BM25 array of accounts for each document
        """
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        return scores

    def filter_documents(self, query):
        """
        Computes BM25 account for request and sorts documents for the loss of this account.
        :param query: row of request
        :Return: sorted list of documents with added field 'bm25_score'
        """
        scores = self.get_scores(query)
        for idx, doc in enumerate(self.documents):
            doc["bm25_score"] = float(scores[idx])
        filtered = sorted(self.documents, key=lambda x: x["bm25_score"], reverse=True)
        return filtered

    def save_results(self, results, filename="filtered_results.json"):
        """
        Saves the filtering results in the JSON file.
        :param results: list of documents
        :param filename: filename to save
        """
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            logging.info(f"The filtering results are stored in {filename}.")
        except Exception as e:
            logging.error(f"Error of file storage {filename}: {e}")
