import base64
import io
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bertopic import BERTopic
from loguru import logger
import nltk
from nltk.corpus import stopwords
import re
import requests


# Предобработка текста
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text


class Clustering:

    def __init__(self, lemma_df):
        nltk.download('stopwords')
        self.russian_stopwords = stopwords.words('russian')
        self.topic_model = None

        self.url = "https://api.deepinfra.com/v1/openai/chat/completions"

        # Фильтрация негативных комментариев
        self.df = lemma_df[lemma_df['label'] == 'negative']
        self.df = self.df.dropna(subset=['text_preprocessed'])

        self.df['cleaned'] = self.df['text_preprocessed'].apply(clean_text)

        # Векторизация текста
        vectorizer = TfidfVectorizer(stop_words=self.russian_stopwords, max_df=0.9, min_df=10)
        self.X_tfidf = vectorizer.fit_transform(self.df['cleaned'])

        self.topics = []

    def fit_bert_topic(self):
        # Обучение BERTopic
        texts = self.df['cleaned'].tolist()
        logger.info(f"Texts count - {len(texts)}. Clustering...")
        self.topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True)
        topics, probs = self.topic_model.fit_transform(texts)
        logger.info("Clustering done!")
        # Добавляем темы в датафрейм
        self.df['topic'] = topics

    def get_topics(self):

        # Показ тем
        logger.info("\n🧠 Топ-15 тем:")
        for idx, topic in self.topic_model.get_topics().items():
            if idx == -1:
                continue  # -1 означает "шум"
            logger.info(f"Тема {idx}: " + " | ".join([word for word, _ in topic[:10]]))
            self.topics.append(topic)
            if idx >= 45:
                break
        return self.topics

    def get_topics_names(self):
        result = []
        for idx, topic in self.topic_model.get_topics().items():
            if idx == -1:
                continue  # -1 означает "шум"
            name = str(self._get_topic_name_openrouter(topic, idx))
            self.df.loc[self.df.topic == idx, "topic_name"] = name
            result.append(name)
            time.sleep(0.8)
            if idx >= 45:
                break
        return result

    def _get_topic_name(self, words: list, number: int):
        headers = {
            "Authorization": "Bearer Hqml68jmyOwWpG85tKfE3bYSEhisuh7H",  # Регистрация: https://deepinfra.com/
            "Content-Type": "application/json"
        }
        data = {
            # "model": "mistralai/Mistral-7B-Instruct-v0.1",
            "model": "meta-llama/Meta-Llama-3-70B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": f"Используя облако слов, определи кратко общую тему на которую жалуются люди, напиши в короткой фразе по-русски в формате 'тема: (не более 4 слов по теме)': {words[:30]}"
                }
            ]
        }

        response = requests.post(self.url, json=data, headers=headers)
        try:
            logger.info(f"Topic {number}: {response.json()["choices"][0]["message"]["content"]}")
            return response.json()["choices"][0]["message"]["content"]
        except:
            logger.error(f"Incorrect resp: {response.json()}")
            raise

    def _get_topic_name_openrouter(self, words: list, number: int):
        # https://openrouter.ai/
        url = "https://openrouter.ai/api/v1/completions"

        payload = {
            "model": "google/gemma-3-27b-it:free",
            "prompt": f"Используя облако слов, определи кратко общую тему на которую жалуются люди, в одном коротком предложении: {words[:30]}"
        }
        headers = {
            "Authorization": "Bearer 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",
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)
        try:
            logger.info(f"Topic {number}: {response.json()["choices"][0]["text"]}")
            return response.json()["choices"][0]["text"]
        except:
            logger.error(f"Incorrect resp: {response.json()}")
            raise

    def _get_topic_name_hugging(self, words: list, number: int):
        headers = {
            "Authorization": "Bearer Hqml68jmyOwWpG85tKfE3bYSEhisuh7H",  # Регистрация: https://deepinfra.com/
            "Content-Type": "application/json"
        }
        data = {
            # "model": "mistralai/Mistral-7B-Instruct-v0.1",
            "model": "meta-llama/Meta-Llama-3-70B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": f"Используя облако слов, определи кратко общую тему на которую жалуются люди, напиши в короткой фразе по-русски в формате 'тема: (не более 4 слов по теме)': {words[:30]}"
                }
            ]
        }

        response = requests.post(self.url, json=data, headers=headers)
        try:
            logger.info(f"Topic {number}: {response.json()["choices"][0]["message"]["content"]}")
            return response.json()["choices"][0]["message"]["content"]
        except:
            logger.error(f"Incorrect resp: {response.json()}")
            raise

    def get_word_clouds(self):
        # Визуализация облаков слов по темам
        unique_topics = self.df['topic'].unique()
        for topic_id in sorted(unique_topics):
            if topic_id == -1:
                continue  # Пропускаем шум
            text = " ".join(self.df[self.df['topic'] == topic_id]['cleaned'])
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                  stopwords=self.russian_stopwords).generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Облако слов для темы {topic_id}')
            plt.show()

    def get_wordcloud_base64_by_topic(self, topic_id):
        logger.info(f"Generation WordCloud for: {topic_id}")
        print(self.df['topic'].unique())
        if topic_id not in self.df['topic'].unique():
            return None
        text = " ".join(self.df[self.df['topic'] == topic_id]['cleaned'])
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              stopwords=self.russian_stopwords).generate(text)
        buffer = io.BytesIO()
        wordcloud.to_image().save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"

    def get_comments_by_topic(self, topic_id, n=20):
        return self.df[self.df['topic'] == topic_id]['text'].head(n).tolist()


class WordCloudShower:
    def __init__(self, df):
        self.df = df.dropna(subset=['topic_name'])
        nltk.download('stopwords')
        self.russian_stopwords = stopwords.words('russian')

    def get_topic_names(self):
        return sorted(list(self.df['topic_name'].unique()))

    def get_wordcloud_base64_by_topic(self, topic_id):
        logger.info(f"Generation WordCloud for: {topic_id}")
        if topic_id not in self.df['topic_name'].unique():
            return None
        text = " ".join(self.df[self.df['topic_name'] == topic_id]['cleaned'])
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              stopwords=self.russian_stopwords).generate(text)
        buffer = io.BytesIO()
        wordcloud.to_image().save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"

    def get_comments_by_topic(self, topic_id, n=20):
        return self.df[self.df['topic_name'] == topic_id]['text'].head(n).tolist()
