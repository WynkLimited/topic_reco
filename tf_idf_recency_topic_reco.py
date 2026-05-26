from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import date, datetime, timedelta

from pymongo import MongoClient
from urllib.parse import quote_plus
import numpy as np
import gcsfs

def get_spark():
    return (
        SparkSession.builder
        .appName("TF-IDF Topic Reco")
        .getOrCreate()
    )

spark = get_spark()
spark.conf.set("spark.sql.parquet.enableVectorizedReader", "false")

def subtractDays(
    numberOfDays: int
) -> str:
    '''
    Subtract n days from todays date.

    Args:
        numberOfDays (int): Number of days to subtract.

    Returns:
        str: Date n days before.
    '''    
    
    yesterday = datetime.now() - timedelta(numberOfDays)
    return yesterday.strftime('%Y-%m-%d')

def load_spark_parquet(base_path, date):
    start_date = datetime.strptime(date, "%Y-%m-%d")

    valid_paths = []
    for i in range(90):
        path = f"{base_path}/day={(start_date - timedelta(days=i)).strftime('%Y-%m-%d')}/"
        try:
            spark.read.parquet(path).limit(1)  
            valid_paths.append(path)
        except AnalysisException:
            pass  

    df = spark.read.parquet(*valid_paths)
    return df


def load_spark_parquet_recent(base_path, date):
    start_date = datetime.strptime(date, "%Y-%m-%d")

    valid_paths = []
    for i in range(15):
        path = f"{base_path}/day={(start_date - timedelta(days=i)).strftime('%Y-%m-%d')}/"
        try:
            spark.read.parquet(path).limit(1)  
            valid_paths.append(path)
        except AnalysisException:
            pass  

    df = spark.read.parquet(*valid_paths)
    return df
    
d_date = subtractDays(2)

base_path = "gs://wynk-ml-workspace/projects/rails_reranking/user_watch_history_new/v1"
watch_df = load_spark_parquet(base_path, d_date).cache()  

watch_df_recent = load_spark_parquet_recent(base_path, d_date).cache()  

movie_df = spark.read.parquet(f'gs://wynk-ml-workspace/projects/xstream_nlu/catalog-db/{d_date}/enriched_movie.parquet')
tv_df = spark.read.parquet(f'gs://wynk-ml-workspace/projects/xstream_nlu/catalog-db/{d_date}/enriched_tv.parquet')

def extract_l2(df):
    return (
        df
        .select("ID", "Title", F.explode(F.col("GptMeta.L2Tags")).alias("l2"),
                F.array_union(
            F.slice(F.col("Actors"), 1, 5),   
            F.slice(F.col("Directors"), 1, 5)             
        ).alias("People"))
        .filter(F.col("l2.Relevance").isin("Extremely Relevant", "Highly Relevant", "Relevant"))
        .select(
            F.col("ID"),
            F.col("Title"),
            F.col("l2.Tag").alias("L2Tag"),
            F.explode("People").alias("People")
        )
    )

content_df = extract_l2(movie_df).unionByName(extract_l2(tv_df))
content_df.cache()

mongo_host = "10.161.216.23"
mongo_port = 27017
mongo_user = "admin"
mongo_password = "IAJNSR$:123xyzesqipy"
mongo_db = "metadata_db"

def get_mongo_string(mongo_host, mongo_port, mongo_user, mongo_password, mongo_db):
    return f"mongodb://{mongo_user}:{quote_plus(mongo_password)}@{mongo_host}:{mongo_port}/{mongo_db}?authSource=admin"
   
mongo_string = get_mongo_string(mongo_host, mongo_port, mongo_user, mongo_password, mongo_db)

client = MongoClient(mongo_string)  
db = client[mongo_db] 

# l2_coll = db['l2_tags_v2']

# cursor = l2_coll.find(
#     {
#         "publish_count_contents": {"$gt": 10},
#         "is_hero_tag": True  # Filters for hero tags only
#     }, 
#     {"_id": 0, "name": 1}
# )

# l2_tag_names = [doc["name"] for doc in cursor]

topics_coll = db['topics']

cursor = topics_coll.find(
    {"published_count": {"$gt": 10}},  
    {"_id": 0, "name": 1}  
)

topics_tag_names = [doc["name"] for doc in cursor]

# filtered_content_df = content_df.filter(F.col("L2Tag").isin(l2_tag_names))
filtered_content_df = content_df.filter(F.col("L2Tag").isin(topics_tag_names))
filtered_content_df = filtered_content_df.filter(F.col("People").isin(topics_tag_names))
# filtered_content_df.show(truncate=False)

fs = gcsfs.GCSFileSystem()
file_path = f"gs://wynk-ml-workspace/projects/rails_reranking/idf_vectors_new/v1/day={d_date}/idf_vector.npy"

with fs.open(file_path, 'rb') as f:
    idf_vector = np.load(f)

# Broadcast IDF
sc = spark.sparkContext
idf_broadcast = sc.broadcast(idf_vector)

# Create mapping L2Tag to index
l2_tag_to_index = {tag: idx for idx, tag in enumerate(topics_tag_names)}
l2_tag_index_broadcast = sc.broadcast(l2_tag_to_index)

UID_COLUMN = "userId"
ITEM_COLUMN = "IDs"

# FULL DATA
user_items_full_df = (
    watch_df
    .withColumn("ID", F.explode(F.col(ITEM_COLUMN)))
    .select(F.col(UID_COLUMN).alias("uid"), "ID")
    .distinct()
)

# LAST 15 DAYS
user_items_recent_df = (
    watch_df_recent
    .withColumn("ID", F.explode(F.col(ITEM_COLUMN)))
    .select(F.col(UID_COLUMN).alias("uid"), "ID")
    .distinct()
)

# Join with content
user_content_full_df = user_items_full_df.join(filtered_content_df, on="ID", how="inner")

user_content_recent_df = user_items_recent_df.join(filtered_content_df, on="ID", how="inner")

tf_full_df = (
    user_content_full_df
    .groupBy("uid", "L2Tag")
    .agg(F.count("*").alias("tf"))
)

tf_recent_df = (
    user_content_recent_df
    .groupBy("uid", "L2Tag")
    .agg(F.count("*").alias("tf"))
)

def compute_tfidf(l2tag, tf):
    try:
        idx = l2_tag_index_broadcast.value[l2tag]
        return float(tf * idf_broadcast.value[idx])
    except Exception:
        return 0.0

tfidf_udf = F.udf(compute_tfidf, "double")

tfidf_full_df = tf_full_df.withColumn("tfidf", tfidf_udf(F.col("L2Tag"), F.col("tf")))
tfidf_recent_df = tf_recent_df.withColumn("tfidf", tfidf_udf(F.col("L2Tag"), F.col("tf")))

window_spec = Window.partitionBy("uid").orderBy(F.desc("tfidf"))

rank_full_df = tfidf_full_df.withColumn("rank", F.row_number().over(window_spec))
rank_recent_df = tfidf_recent_df.withColumn("rank", F.row_number().over(window_spec))

# recent_top3 = (
#     rank_recent_df
#     .filter(F.col("rank") <= 3)
#     .groupBy("uid")
#     .agg(F.collect_list("L2Tag").alias("recent_tags"))
# )
recent_top3 = (
    rank_recent_df
    .filter(F.col("rank") <= 3)
    .groupBy("uid")
    .agg(
        F.collect_list(
            F.struct(
                F.col("L2Tag").alias("tag_name"),
                F.col("tfidf").alias("tag_score")
            )
        ).alias("recent_tags")
    )
)

# full_topN = (
#     rank_full_df
#     .filter(F.col("rank") <= 15)
#     .groupBy("uid")
#     .agg(F.collect_list("L2Tag").alias("full_tags"))
# )
full_topN = (
    rank_full_df
    .filter(F.col("rank") <= 15)
    .groupBy("uid")
    .agg(
        F.collect_list(
            F.struct(
                F.col("L2Tag").alias("tag_name"),
                F.col("tfidf").alias("tag_score")
            )
        ).alias("full_tags")
    )
)

tf_people_df = (
    user_content_full_df
    .groupBy("uid", "People")
    .agg(F.count("*").alias("tf"))
)

# Rank People
window_people = Window.partitionBy("uid").orderBy(F.desc("tf"))

rank_people_df = tf_people_df.withColumn("rank", F.row_number().over(window_people))

# top2_people = (
#     rank_people_df
#     .filter(F.col("rank") <= 2)
#     .groupBy("uid")
#     .agg(F.collect_list("People").alias("people_tags"))
# )
top2_people = (
    rank_people_df
    .filter(F.col("rank") <= 2)
    .groupBy("uid")
    .agg(
        F.collect_list(
            F.struct(
                F.col("People").alias("tag_name"),
                F.col("tf").alias("tag_score")
            )
        ).alias("people_tags")
    )
)


# combined_tags = (
#     full_topN
#     .join(recent_top3, on="uid", how="left")
#     .join(top2_people, on="uid", how="left")
    
#     # Remove overlap from full tags
#     .withColumn(
#         "filtered_full_tags",
#         F.expr("array_except(full_tags, recent_tags)")
#     )
    
#     # Take top 5 from remaining
#     .withColumn(
#         "top5_full",
#         F.expr("slice(filtered_full_tags, 1, 5)")
#     )
    
#     # Final 10 tags
#     .withColumn(
#         "final_tags",
#         F.expr("concat(recent_tags, top5_full, people_tags)")
#     )
    
#     .select("uid", "final_tags")
# )

combined_tags = (
    full_topN
    .join(recent_top3, on="uid", how="left")
    .join(top2_people, on="uid", how="left")

    # Extract only names for overlap logic
    .withColumn(
        "recent_tag_names",
        F.expr("transform(recent_tags, x -> x.tag_name)")
    )

    .withColumn(
        "filtered_full_tags",
        F.expr("""
            filter(
                full_tags,
                x -> NOT array_contains(recent_tag_names, x.tag_name)
            )
        """)
    )

    # Take top 5 remaining
    .withColumn(
        "top5_full",
        F.expr("slice(filtered_full_tags, 1, 5)")
    )

    # Final tags
    .withColumn(
        "final_tags",
        F.expr("""
            concat(
                recent_tags,
                top5_full,
                people_tags
            )
        """)
    )

    .select("uid", "final_tags")
)

# combined_tags = (
#     full_top5.join(recent_top2, on="uid", how="left")
#     .withColumn(
#         "remaining_tags",
#         F.expr("slice(array_except(full_tags, recent_tags),1,3)")
#     )
#     .withColumn(
#         "top5_topic",
#         F.expr("concat(recent_tags, remaining_tags)")
#     )
#     .select("uid", "top5_topic")
# )

user_meta_df = (
    user_content_full_df
    .groupBy("uid")
    .agg(
        F.collect_set("ID").alias("nf_ids"),
        F.collect_set("Title").alias("titles")
    )
)

final_df = user_meta_df.join(combined_tags, on="uid", how="left")

final_df = final_df.filter(F.size(F.col("nf_ids")) >= 2)

# output_path = f"gs://wynk-ml-workspace/ritika/uid_topic_output/{d_date}/"

# (
#     final_df
#     .repartition(200) 
#     .write
#     .mode("overwrite")
#     .parquet(output_path)
# )

# print(f" Parquet written to {output_path}")

tag_meta_coll = db["l2_tags"]   

tag_cursor = tag_meta_coll.find(
    {},
    {
        "_id": 1,
        "name": 1,
        "type": 1,
        "category": 1,
        "parent_id": 1
    }
)

tag_meta_data = list(tag_cursor)

# Create Spark DF
tag_meta_df = spark.createDataFrame(tag_meta_data)

# Rename columns properly
tag_meta_df = (
    tag_meta_df
    .withColumnRenamed("_id", "tag_id")
    .withColumnRenamed("name", "tag_name")
)

# EXPLODE USER TAGS

# user_tags_exploded = final_df.select(
#     F.col("uid").alias("user_id"),
#     F.explode("final_tags").alias("tag_name")
# )
user_tags_exploded = final_df.select(
    F.col("uid").alias("user_id"),
    F.explode("final_tags").alias("tag_struct")
).select(
    "user_id",
    F.col("tag_struct.tag_name").alias("tag_name"),
    F.col("tag_struct.tag_score").alias("tag_score")
)

# JOIN WITH TAG METADATA

final_output_df = (
    user_tags_exploded
    .join(tag_meta_df, on="tag_name", how="left")
    .withColumn(
        "type",
        F.when(F.col("tag_id").isNotNull(), F.lit("L2-tag"))
         .otherwise(F.lit("People"))
    )
    .withColumn("model", F.lit("Recency TF-IDF"))
    .withColumn("version", F.lit("v1"))
)

# FINAL COLUMN ORDER

final_output_df = final_output_df.select(
    "tag_id",
    "user_id",
    "tag_name",
    "tag_score",
    "type",
    "category",
    "parent_id",
    "model",
    "version"
)


final_output_path = f"gs://wynk-ml-workspace/ritika/uid_topic_recency_output/{d_date}/"

(
    final_output_df
    .repartition(200)
    .write
    .mode("overwrite")
    .parquet(final_output_path)
)