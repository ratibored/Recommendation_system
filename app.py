import os
import pandas as pd
from typing import List
from catboost import CatBoostClassifier
from fastapi import FastAPI
from schema import PostGet, Response
from datetime import datetime
from sqlalchemy import create_engine
from loguru import logger
import uvicorn
import hashlib

app = FastAPI()
SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"


def get_exp_group(user_id: int) -> str:
    temp_exp_group = int(int(hashlib.md5((str(user_id) + 'my_salt').encode()).hexdigest(), 16) % 100)
    if temp_exp_group <= 50:
        exp_group = "control"
    elif temp_exp_group > 50:
        exp_group = "test"
    return exp_group


def batch_load_sql(query: str):
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    conn = engine.connect().execution_options(
        stream_results=True
    )
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=200000):
        chunks.append(chunk_dataframe)
        logger.info(f'Got chunk: {len(chunk_dataframe)}')
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def get_model_path(path: str, name: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = fr'/workdir/user_input/{name}'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_features_control():
    # Уникальные записи post_id, user_id, где был совершен лайк

    logger.info("loading liked posts")
    liked_posts_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        where action='like'"""
    liked_posts = batch_load_sql(liked_posts_query)

    # Загружаем фичи по постам с эмбеддингами на основе tf-idf
    logger.info("loading posts features")
    posts_features = pd.read_sql(
        "SELECT * FROM public.posts_info_features",
        con=SQLALCHEMY_DATABASE_URL
    )

    # Загружаем фичи по пользователям
    logger.info("loading user features")
    user_features = pd.read_sql(
        "SELECT * FROM public.user_data",

        con=SQLALCHEMY_DATABASE_URL
    )
    return [liked_posts, posts_features, user_features]


def load_features_test():
    # Уникальные записи post_id, user_id, где был совершен лайк

    logger.info("loading liked posts")
    liked_posts_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        where action='like'"""
    liked_posts = batch_load_sql(liked_posts_query)

    # Загружаем фичи по постам с эмбеддингами на основе DistilBert
    logger.info("loading posts features")
    posts_features = pd.read_sql(
        "SELECT * FROM public.posts_info_features_dl",
        con=SQLALCHEMY_DATABASE_URL
    )

    # Загружаем фичи по пользователям
    logger.info("loading user features")
    user_features = pd.read_sql(
        "SELECT * FROM public.user_data",

        con=SQLALCHEMY_DATABASE_URL
    )

    # Загружаем посты без предобработки
    content = pd.read_sql(
        "SELECT * FROM public.post_text_df",
        con=SQLALCHEMY_DATABASE_URL
    )

    return [liked_posts, posts_features, user_features, content]


def load_models(name: str):
    # Загружаем CatBoost

    model_path = get_model_path(fr".\Models\{name}", name)
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    return loaded_model

# При поднятии сервиса положим модели и фичи в переменные control_model, test_model, feature_control, features_test

logger.info("loading model control (classic)")
control_model = load_models('model_classic')

logger.info("loading model test (dl)")
test_model = load_models('model_berted')

logger.info("loading control features")
features_control = load_features_control()

logger.info("loading test features")
features_test = load_features_test()

logger.info("service is up and running")


def get_recommended_feed_control(id: int, time: datetime, limit: int):
    # Загрузим фичи по пользователям
    logger.info(f"user_id: {id}")
    logger.info("reading features")
    user_features = features_control[2].loc[features_control[2].user_id == id]
    user_features = user_features.drop('user_id', axis=1)

    # Загрузим фичи по постам
    logger.info("dropping columns")
    posts_features = features_control[1].drop(['index', 'text'], axis=1)
    content = features_control[1][['post_id', 'text', 'topic']]

    # Объединим эти фичи
    logger.info("zipping everything")
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    logger.info("assigning everything")
    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index('post_id')

    # Добавим информацию о дате рекомендаций
    logger.info("add time info")
    user_posts_features["hour"] = time.hour
    user_posts_features["month"] = time.month

    # Сформируем предсказания вероятности лайкнуть пост для всех постов
    logger.info("predicting")
    predicts = control_model.predict_proba(user_posts_features)[:, 1]
    user_posts_features["predicts"] = predicts

    # Уберем записи, где пользователь ранее уже ставил лайк
    logger.info("deleting liked posts")
    liked_posts = features_control[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    # Рекомендуем топ-limit по вероятности постов
    recommended_posts = filtered_.sort_values('predicts')[-limit:].index

    return [
        PostGet(**{
            "id": i,
            "text": content[content.post_id == i].text.values[0],
            "topic": content[content.post_id == i].topic.values[0]
        }) for i in recommended_posts
    ]


def get_recommended_feed_test(id: int, time: datetime, limit: int):
    # Загрузим фичи по пользователям
    logger.info(f"user_id: {id}")
    logger.info("reading features")
    user_features = features_test[2].loc[features_test[2].user_id == id]
    user_features = user_features.drop('user_id', axis=1)

    # Загрузим фичи по постам
    logger.info("dropping columns")
    posts_features = features_test[1].drop(['index'], axis=1)
    content = features_test[3]

    # Объединим эти фичи
    logger.info("zipping everything")
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    logger.info("assigning everything")
    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index('post_id')

    # Добавим информацию о дате рекомендаций
    logger.info("add time info")
    user_posts_features["hour"] = time.hour
    user_posts_features["month"] = time.month

    # Сформируем предсказания вероятности лайкнуть пост для всех постов
    logger.info("predicting")
    columns = ['hour', 'month', 'gender', 'age', 'country', 'city', 'exp_group', 'os', 'source',
               'topic', 'TextCluster', 'DistanceToCluster_0', 'DistanceToCluster_1', 'DistanceToCluster_2',
               'DistanceToCluster_3', 'DistanceToCluster_4', 'DistanceToCluster_5', 'DistanceToCluster_6',
               'DistanceToCluster_7', 'DistanceToCluster_8', 'DistanceToCluster_9', 'DistanceToCluster_10',
               'DistanceToCluster_11', 'DistanceToCluster_12', 'DistanceToCluster_13', 'DistanceToCluster_14']
    user_posts_features = user_posts_features[columns]
    predicts = test_model.predict_proba(user_posts_features)[:, 1]
    user_posts_features["predicts"] = predicts

    # Уберем записи, где пользователь ранее уже ставил лайк
    logger.info("deleting liked posts")
    liked_posts = features_test[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    # Рекомендуем топ-limit по вероятности постов
    recommended_posts = filtered_.sort_values('predicts')[-limit:].index

    return [
        PostGet(**{
            "id": i,
            "text": content[content.post_id == i].text.values[0],
            "topic": content[content.post_id == i].topic.values[0]
        }) for i in recommended_posts
    ]


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int = 10) -> Response:
    exp_group = get_exp_group(id)
    if exp_group == "control":
        result = get_recommended_feed_control(id, time, limit)
    else:
        result = get_recommended_feed_test(id, time, limit)
    return {"exp_group": exp_group, "recommendations": result}
