from dotenv import load_dotenv
import os
from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions

from datetime import timedelta
from tqdm import tqdm
import uuid
import pandas as pd
from openai import OpenAI

MOVIES_DATASET = "imdb_top_1000.csv"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
REQUIRED_ENVIRONMENT_VARIABLES = (
    "OPENAI_API_KEY",
    "DB_CONN_STR",
    "DB_USERNAME",
    "DB_PASSWORD",
    "DB_BUCKET",
    "DB_SCOPE",
    "DB_COLLECTION",
)


def check_environment_variable(variable_name):
    """Check if environment variable is set"""
    if variable_name not in os.environ:
        raise ValueError(
            f"{variable_name} environment variable is not set. Please add it to the environment"
        )


def connect_to_couchbase(connection_string, db_username, db_password):
    """Connect to couchbase"""
    print("Connecting to couchbase...")
    auth = PasswordAuthenticator(db_username, db_password)
    options = ClusterOptions(auth)
    connect_string = connection_string
    cluster = Cluster(connect_string, options)

    # Wait until the cluster is ready for use.
    cluster.wait_until_ready(timedelta(seconds=5))

    return cluster


def generate_embeddings(client, input_data: str, model: str):
    """Generate OpenAI embeddings for the input data"""
    response = client.embeddings.create(
        input=input_data,
        model=model,
        extra_body={"input_type": "query", "truncate": "NONE"},
    )
    return response.data[0].embedding


def prepare_movie_records(data: pd.DataFrame) -> list[dict]:
    """Normalize the IMDB dataset into JSON-ready movie documents.

    The search app filters on these fields, and Couchbase cannot store NaN
    values, so this is the one place where the dataset is cleaned.
    """
    data = data.copy()

    # Convert columns to numeric types
    data["Gross"] = data["Gross"].str.replace(",", "").astype(float)

    # Fill empty values
    data["Gross"] = data["Gross"].fillna(0)
    data["Certificate"] = data["Certificate"].fillna("NA")
    data["Meta_score"] = data["Meta_score"].fillna(-1)

    return data.to_dict(orient="records")


def main():
    # Load environment variables
    load_dotenv(override=True)

    # Ensure that all environment variables are set
    for variable_name in REQUIRED_ENVIRONMENT_VARIABLES:
        check_environment_variable(variable_name)

    # Use text-embedding-3-small as the embedding model if not set
    embedding_model = os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    client = OpenAI()

    try:
        cluster = connect_to_couchbase(
            os.getenv("DB_CONN_STR"), os.getenv("DB_USERNAME"), os.getenv("DB_PASSWORD")
        )
        bucket = cluster.bucket(os.getenv("DB_BUCKET"))
        scope = bucket.scope(os.getenv("DB_SCOPE"))
        collection = scope.collection(os.getenv("DB_COLLECTION"))
        movies = prepare_movie_records(pd.read_csv(MOVIES_DATASET))

        print("Ingesting Data...")
        for row in tqdm(movies):
            try:
                row["Overview_embedding"] = generate_embeddings(
                    client, row["Overview"], embedding_model
                )
            except Exception as e:
                print(f"Error while generating embeddings for {row['Series_Title']}", e)
                continue
            doc_id = uuid.uuid4().hex
            collection.upsert(doc_id, row)

    except Exception as e:
        print("Error while ingesting data", e)


if __name__ == "__main__":
    main()
