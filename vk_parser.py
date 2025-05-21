import csv
import os

from loguru import logger

import pandas as pd
import vk_api
import warnings
import traceback
from datetime import datetime, timedelta

headers = [
    'group_domain', 'post_id',
    'post_date', 'comment_id',
    'from_id', 'text', 'comment_date'
]


def format_timestamp(unix_time):
    return datetime.utcfromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')


def write_batch_to_file(data_batch, head, file_path, is_first_batch=False):
    """Записывает пачку данных в файл"""
    mode = 'w' if is_first_batch else 'a'
    with open(file_path, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if is_first_batch:
            writer.writerow(head)
        writer.writerows(data_batch)


class VkParser:
    def __init__(self, days: int = 45) -> None:
        self.days = days
        pd.set_option('display.max_columns', None)
        warnings.filterwarnings(action='ignore')
        # Авторизация через токен
        token = 'dd735edadd735edadd735edac8de555238ddd73dd735edaba246ac7ad99520f3b495346'
        vk_session = vk_api.VkApi(token=token)
        self.vk = vk_session.get_api()

        # Инпут- и аутпут-файлы
        self.groups_file = 'tagil_news.txt'
        self.output_file = 'scraped_comments.csv'
        self.temp_output_file = 'temp_scraped_comments.csv'  # Временный файл для того, чтобы не потерять файлы

        # Настройки
        self.LAST_DATE = datetime.today()  # сегодняшняя дата
        self.FIRST_DATE = self.LAST_DATE - timedelta(days=days)  # примерно 1.5 месяца назад
        self.BATCH_SIZE = 100  # Количество постов после которых происходит запись в файл

    def get_posts_and_comments(self, group_domain, head = None):
        """
        Возвращает список списков, где каждая вложенная запись —
        это либо пост, либо комментарий.
        """
        if head is None:
            head = headers
        data = []
        batch_counter = 0

        try:
            all_posts = []
            offset = 0
            reached_end = False
            reached_early_posts = False

            # Получаем посты с фильтрацией по дате
            while not reached_end and not reached_early_posts:
                response = self.vk.wall.get(domain=group_domain, count=100, offset=offset)
                posts = response.get('items', [])
                if not posts:
                    break

                for post in posts:
                    post_date = datetime.utcfromtimestamp(post['date'])

                    if post_date > self.LAST_DATE:
                        continue  # Пропускаем посты после LAST_DATE

                    if post_date < self.FIRST_DATE:
                        reached_early_posts = True
                        break  # Прекращаем, если достигли постов раньше FIRST_DATE

                    all_posts.append(post)
                    if len(all_posts) >= MAX_POSTS:
                        break  # Ограничение на максимальное количество постов

                offset += len(posts)

                if len(all_posts) >= MAX_POSTS:
                    break

                if reached_early_posts:
                    break

            total_comments = 0
            logger.info(f"{post_date}, {len(all_posts)}")

            # Для каждого поста:
            for post_idx, post in enumerate(all_posts, 1):
                if post_idx % 100 == 0:
                    logger.info(f"Processed {post_idx} posts")
                post_id = post['id']
                post_date_unix = post['date']
                post_date_str = format_timestamp(post_date_unix)
                post_text = post.get('text', '')
                from_id = post.get('from_id', '')

                # Сохраняем сам пост (comment_id='', comment_date='')
                data.append([
                    group_domain,  # group_domain
                    post_id,  # post_id
                    post_date_str,  # post_date
                    '',  # comment_id (пусто для поста)
                    from_id,  # from_id
                    post_text,  # text
                    ''  # comment_date
                ])

                # Теперь получаем комментарии к этому посту
                comments_offset = 0
                while True:
                    if total_comments >= MAX_COMMENTS:
                        break  # Достигли лимита по комментариям в группе

                    resp_comments = self.vk.wall.getComments(
                        owner_id=post['owner_id'],
                        post_id=post_id,
                        count=100,
                        offset=comments_offset
                    )

                    comment_items = resp_comments.get('items', [])
                    if not comment_items:
                        break

                    # Обрабатываем каждый комментарий
                    for comment in comment_items:
                        if total_comments >= MAX_COMMENTS:
                            break
                        text = comment.get('text', '')
                        if not text:
                            continue

                        from_id_c = comment.get('from_id', '')
                        comment_id = comment.get('id', '')
                        comment_date_unix = comment.get('date', 0)
                        comment_date_str = format_timestamp(comment_date_unix)

                        # Фильтрация комментариев по дате
                        comment_date = datetime.utcfromtimestamp(comment_date_unix)
                        if comment_date < self.FIRST_DATE or comment_date > self.LAST_DATE:
                            continue

                        data.append([
                            group_domain,  # group_domain
                            post_id,  # post_id
                            post_date_str,  # постовая дата для контекста
                            comment_id,  # comment_id
                            from_id_c,  # from_id
                            text,  # text (комментария)
                            comment_date_str  # comment_date
                        ])

                        total_comments += 1

                    comments_offset += 100

                # Записываем каждые BATCH_SIZE постов или в конце
                if post_idx % self.BATCH_SIZE == 0 or post_idx == len(all_posts):
                    batch_counter += 1
                    is_first_batch = (batch_counter == 1)
                    write_batch_to_file(data, headers, self.temp_output_file, is_first_batch)
                    data = []  # Очищаем текущий буфер данных

        except Exception as e:
            logger.error(f"Ошибка при обработке группы {group_domain}: {e}\n{traceback.format_exc()}")
            # Записываем оставшиеся данные перед выходом
            if data:
                write_batch_to_file(data, headers, self.temp_output_file)
            raise e

        return batch_counter  # Возвращаем количество записанных батчей


    def run_parsing(self) -> pd.DataFrame:
        with open('tagil_news.txt', 'r', encoding='utf-8') as f:
            group_domains = [line.strip() for line in f if line.strip()]
        logger.info(f"Parsing started, days: {self.days}, {self.FIRST_DATE} - {self.LAST_DATE}\nGroup count: {len(group_domains)}")

        # Шапка CSV
        headers = [
            'group_domain',
            'post_id',
            'post_date',
            'comment_id',
            'from_id',
            'text',
            'comment_date'
        ]

        # Создаем пустой файл с заголовками
        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)

        total_rows = 0

        for group_domain in group_domains:
            try:
                # Очищаем временный файл перед обработкой новой группы
                if os.path.exists(self.temp_output_file):
                    os.remove(self.temp_output_file)

                batch_count = self.get_posts_and_comments(group_domain, headers)

                # После обработки группы, добавляем данные из временного файла в основной
                if os.path.exists(self.temp_output_file):
                    with open(self.temp_output_file, 'r', encoding='utf-8') as temp_f:
                        reader = csv.reader(temp_f)
                        next(reader)  # Пропускаем заголовок
                        rows = list(reader)

                    with open(self.output_file, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows(rows)

                    num_rows = len(rows)
                    total_rows += num_rows
                    logger.info(f"\nСобрано записей (постов + комментариев) из группы {group_domain}: {num_rows}")
                    logger.info(f"Данные записаны в {batch_count} батчах")

                    # Удаляем временный файл
                    os.remove(self.temp_output_file)

            except Exception as e:
                logger.error(f"Ошибка при обработке группы {group_domain}: {e}")
                continue

        logger.info(f"\nИтоговое число собранных строк (посты + комментарии): {total_rows}")
        logger.info(f"Данные успешно сохранены в {self.output_file}")
        return pd.read_csv(self.output_file)


# Здесь должны быть определены MAX_POSTS и MAX_COMMENTS
MAX_POSTS = 1000000
MAX_COMMENTS = 1000000
