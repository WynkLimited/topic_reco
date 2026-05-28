from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import date, datetime, timedelta

from pymongo import MongoClient
from urllib.parse import quote_plus
import numpy as np
import gcsfs
from pyspark.ml.recommendation import ALS

def get_spark():
    return (
        SparkSession.builder
        .appName("ALS Topic Reco")
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
    # start_date = date

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

d_date = subtractDays(1)

base_path = "gs://wynk-ml-workspace/projects/rails_reranking/user_watch_history_new/v1"
watch_df = load_spark_parquet(base_path, d_date).cache()  

df_result = watch_df.groupBy("userId") \
              .agg(F.array_distinct(F.flatten(F.collect_list("IDs"))).alias("deduped_IDs"))

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

mongo_host = spark.conf.get("spark.mongo.host")
mongo_port = spark.conf.get("spark.mongo.port")
mongo_user = spark.conf.get("spark.mongo.user")
mongo_password = spark.conf.get("spark.mongo.password")
mongo_db = "metadata_db"

def get_mongo_string(mongo_host, mongo_port, mongo_user, mongo_password, mongo_db):
    return f"mongodb://{mongo_user}:{quote_plus(mongo_password)}@{mongo_host}:{mongo_port}/{mongo_db}?authSource=admin"
   
mongo_string = get_mongo_string(mongo_host, mongo_port, mongo_user, mongo_password, mongo_db)

client = MongoClient(mongo_string)  
db = client[mongo_db] 

topics_coll = db['topics']

cursor = topics_coll.find(
    {"published_count": {"$gt": 10}, "type" : "L2-tag"},  
    {"name": 1}  
)

cursor2 = topics_coll.find(
    {"type": {"$in": ["Actor", "Director"]}},
    {"_id": 0, "name": 1}
)

topics_tag_names = [doc["name"] for doc in cursor]
topics_people_names = [doc["name"] for doc in cursor2]

print(f"L2 tags with publish_count > 10: {len(topics_tag_names)}")
print(f"People with publish_count > 0: {len(topics_people_names)}")

filtered_content_df = content_df.filter(F.col("L2Tag").isin(topics_tag_names))
filtered_content_df = filtered_content_df.filter(F.col("People").isin(topics_people_names))
# filtered_content_df.show(truncate=False)

meta_df = filtered_content_df.groupBy("ID").agg(
    F.first("Title").alias("Title"),
    F.collect_set("L2Tag").alias("L2Tags"),
    F.collect_set("People").alias("People")  
)

df_user_content = df_result.select(
    "userId",
    F.explode("deduped_IDs").alias("content_id")
).dropDuplicates()

# Add interaction
df_user_content = df_user_content.withColumn("interaction", F.lit(1))

# Filter users with >=3 interactions
user_activity = df_user_content.groupBy("userId").count()
active_users = user_activity.filter("count >= 3").select("userId")
df_user_content_active = df_user_content.join(active_users, "userId")

# Hash IDs to numeric indices
df_indexed = df_user_content_active.withColumn(
    "user_index", F.hash(F.col("userId")).cast("long")
).withColumn(
    "content_index", F.hash(F.col("content_id")).cast("long")
).withColumn(
    "interaction", F.lit(1)
)

df_indexed = df_indexed.repartition(2000)
df_indexed.cache()

df_indexed = df_indexed \
    .withColumn("user_index", F.col("user_index").cast("int")) \
    .withColumn("tag_index", F.col("content_index").cast("int")) \
    .withColumn("watch_count", F.col("interaction").cast("float"))

als = ALS(
    userCol="user_index",
    itemCol="content_index",
    ratingCol="interaction",
    implicitPrefs=True,

    # Core model tuning
    rank=40,                 
    maxIter=10,              
    regParam=0.08,           
    alpha=20,      

    # Stability and performance
    coldStartStrategy="drop",
    nonnegative=True,

    # Distributed performance tuning
    numUserBlocks=200,       
    numItemBlocks=50,        
    checkpointInterval=2    
)

model = als.fit(df_indexed)

user_recs = model.recommendForAllUsers(100)

user_recs = user_recs.withColumn("rec", F.explode("recommendations"))
user_recs = user_recs.select(
    F.col("user_index"),
    F.col("rec.content_index"),
    F.col("rec.rating")
)

# Create mapping from content_id to content_index
content_mapping = df_indexed.select("content_index", "content_id").dropDuplicates()

user_recs = user_recs.join(content_mapping, "content_index", "left")

recs_with_meta = user_recs.join(meta_df, user_recs.content_id == meta_df.ID, "left")
recs_with_meta = recs_with_meta.drop("ID")


def get_top_tags(df, tag_col, top_n):
    df_exploded = df.withColumn("tag", F.explode(F.col(tag_col)))
    
    df_ranked = df_exploded.groupBy("user_index", "tag") \
        .agg(F.sum("rating").alias("tag_score"))
    
    window = Window.partitionBy("user_index").orderBy(F.desc("tag_score"))

    window_score = Window.partitionBy("user_index", "tag_score").orderBy(F.desc("tag_score"))
    df_unique = df_ranked.withColumn("row_num", F.row_number().over(window_score)) \
                         .filter(F.col("row_num") == 1) \
                         .drop("row_num")
    
    return df_unique.withColumn("rank", F.row_number().over(window)) \
                    .filter(f"rank <= {top_n}")

df_top_l2 = get_top_tags(recs_with_meta, "L2Tags", 8)
df_top_people = get_top_tags(recs_with_meta, "People", 2)

df_top_tags = df_top_l2.unionByName(df_top_people)

# Create mapping from user_index to userId
user_mapping = df_indexed.select("user_index", "userId").dropDuplicates()
df_top_tags = df_top_tags.join(user_mapping, "user_index", "left")

df_top_tags_randomized = df_top_tags.withColumn(
    "tag_score",
    F.col("tag_score") + (F.rand() * 0.09 + 0.01)
)

final_user_tags = df_top_tags.select(
    "userId", "tag", "tag_score"
).orderBy("userId", "tag_score", ascending=False)

# Rank tags per user by tag_score
window = Window.partitionBy("userId").orderBy(F.col("tag_score").desc())

df_ranked = df_top_tags.withColumn("rank", F.row_number().over(window))

# Keep only top 10 tags per user
df_top10 = df_ranked.filter(F.col("rank") <= 10)

# # Aggregate into list
# final_df = df_top10.groupBy("userId").agg(
#     F.collect_list("tag").alias("tags")
# )

# Aggregate tag and score together
final_df = df_top10.groupBy("userId").agg(
    F.collect_list(
        F.struct(
            F.col("tag"),
            # F.col("tag_score")
            F.round(F.col("tag_score"), 4).alias("tag_score")
        )
    ).alias("tags")
) 

tag_cursor = topics_coll.find(
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
    .withColumn("tag_id", F.col("tag_id").cast("string"))
    .withColumn("parent_id", F.col("parent_id").cast("string"))
)

# EXPLODE USER TAGS
user_tags_exploded = final_df.select(
    F.col("userId").alias("user_id"),
    F.explode("tags").alias("tag_data")
).select(
    "user_id",
    F.col("tag_data.tag").alias("tag_name"),
    F.col("tag_data.tag_score").alias("tag_score")
)

# JOIN WITH TAG METADATA

final_output_df = (
    user_tags_exploded
    .join(tag_meta_df, on="tag_name", how="left")
    .withColumn("model", F.lit("ALS"))
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


# WRITE OUTPUT

final_output_path = f"gs://wynk-ml-workspace/projects/neuralflix/user-topic-dump/day={d_date}/"

# (
#     final_output_df
#     .repartition(200)
#     .write
#     .mode("overwrite")
#     .parquet(final_output_path)
# )

(
    final_output_df
    .write
    .mode("append")
    .partitionBy("model")   
    .parquet(final_output_path)
)

