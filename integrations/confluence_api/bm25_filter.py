import json
import logging
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO)

class BM25Filter:
    def __init__(self, documents):
        """
        Инициализирует фильтр BM25.
        :param documents: список документов, где каждый документ — словарь с ключом "text"
        """
        self.documents = documents
        self.tokenized_docs = [self.tokenize(doc.get("text", "")) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        logging.info(f"BM25 индекс построен для {len(self.documents)} документов.")

    @staticmethod
    def tokenize(text):
        """Простейшая токенизация текста (разбивка по пробелам и приведение к нижнему регистру)."""
        return text.lower().split()

    def get_scores(self, query):
        """
        Вычисляет BM25-счёт для запроса.
        :param query: строка запроса
        :return: массив BM25-счётов для каждого документа
        """
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        return scores

    def filter_documents(self, query):
        """
        Вычисляет BM25-счёт для запроса и сортирует документы по убыванию этого счёта.
        :param query: строка запроса
        :return: отсортированный список документов с добавленным полем 'bm25_score'
        """
        scores = self.get_scores(query)
        for idx, doc in enumerate(self.documents):
            doc["bm25_score"] = float(scores[idx])
        filtered = sorted(self.documents, key=lambda x: x["bm25_score"], reverse=True)
        return filtered

    def save_results(self, results, filename="filtered_results.json"):
        """
        Сохраняет результаты фильтрации в JSON файл.
        :param results: список документов
        :param filename: имя файла для сохранения
        """
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            logging.info(f"Результаты фильтрации сохранены в {filename}.")
        except Exception as e:
            logging.error(f"Ошибка сохранения файла {filename}: {e}")
