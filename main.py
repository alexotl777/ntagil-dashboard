from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

from clustering import Clustering
from dashboard import app
from lemmatizer import Lemmatizer
from vk_parser import VkParser


def full_pipeline():
    logger.info("Start Job...")
    parser = VkParser(40)
    parsed_df = parser.run_parsing()
    lemma = Lemmatizer(parsed_df)

    lemma.df['text_preprocessed'] = lemma.df['text'].apply(lemma.remove_links)
    logger.info('ready_1')

    lemma.df['text_preprocessed'] = lemma.df['text_preprocessed'].apply(lemma.remove_emojis)
    logger.info('ready_2')

    lemma.df['text_preprocessed'] = lemma.df['text_preprocessed'].apply(lemma.clean_text)
    logger.info('ready_3')

    lemma.df['text_preprocessed'] = lemma.df['text_preprocessed'].apply(lemma.remove_stopwords)
    logger.info('ready_4')

    lemma.df['text_preprocessed'] = lemma.df['text_preprocessed'].apply(lemma.lemmatize_words)
    logger.info('ready_5')

    sentiment_df = lemma.analize_sentiment()
    sentiment_df.to_excel('tagil_comments_with_labels.xlsx')

    cluster = Clustering(sentiment_df)
    cluster.fit_bert_topic()

    topics = cluster.get_topics_names()
    logger.info(f"Topics: {topics}")
    cluster.df.to_excel("data_with_topic_names.xlsx")


if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        full_pipeline,
        'interval',
        hours=12,
        next_run_time=datetime.now()
    )
    scheduler.start()
    logger.info("Start App...")
    app.run()
    # full_pipeline()
