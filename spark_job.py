from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import DataFrame


# --- TRANSFORMATION FUNCTIONS (Testable) ---

def prepare_receipts_for_join(receipts_df: DataFrame, year: int) -> DataFrame:
    """Filter receipts by year and add rounded coordinates."""
    return receipts_df.filter(F.year(F.col("date_time")) == year) \
        .withColumn("date_only", F.to_date("date_time")) \
        .withColumn("lat_r", F.round("lat", 2)) \
        .withColumn("lng_r", F.round("lng", 2))


def prepare_weather_for_join(weather_df: DataFrame) -> DataFrame:
    """Prepare weather data with rounded coordinates."""
    return weather_df \
        .withColumn("lat_r", F.round("lat", 2)) \
        .withColumn("lng_r", F.round("lng", 2)) \
        .withColumnRenamed("wthr_date", "date_only") \
        .select("lat_r", "lng_r", "date_only", "avg_tmpr_c")


def enrich_receipts_with_weather(receipts_df: DataFrame, weather_df: DataFrame) -> DataFrame:
    """Join receipts with weather and calculate original cost."""
    return receipts_df.join(weather_df, ["lat_r", "lng_r", "date_only"]) \
        .filter(F.col("avg_tmpr_c") > 0) \
        .withColumn("original_total_cost", F.col("total_cost") + F.col("discount"))


def categorize_order_size(items_count_col: str) -> F.Column:
    """Return a Column expression that categorizes order size."""
    return F.when((F.col(items_count_col).isNull()) | (F.col(items_count_col) <= 0), "erroneous_data_cnt") \
        .when(F.col(items_count_col) <= 1, "tiny_cnt") \
        .when(F.col(items_count_col) <= 3, "small_cnt") \
        .when(F.col(items_count_col) <= 10, "medium_cnt") \
        .otherwise("large_cnt")


def compute_receipt_counts(enriched_df: DataFrame) -> DataFrame:
    """Group by receipt_id to count items per order."""
    return enriched_df.groupBy("restaurant_franchise_id", "receipt_id") \
        .agg(F.count("id").alias("items_count"))


def build_initial_state(receipt_counts_df: DataFrame) -> DataFrame:
    """Build the pivoted state per restaurant with most popular order type."""
    count_cols = ["erroneous_data_cnt", "tiny_cnt", "small_cnt", "medium_cnt", "large_cnt"]
    
    state_categorized = receipt_counts_df.withColumn("order_size_cat", categorize_order_size("items_count"))
    
    initial_state = state_categorized.groupBy("restaurant_franchise_id") \
        .pivot("order_size_cat", count_cols) \
        .count().fillna(0) \
        .withColumn("batch_timestamp", F.current_timestamp())
    
    cols_struct = F.array([F.struct(F.col(c).alias("val"), F.lit(c).alias("name")) for c in count_cols])
    return initial_state.withColumn("most_popular_order_type", 
        F.sort_array(cols_struct, False).getItem(0).getField("name"))


def apply_promo_logic(df: DataFrame) -> DataFrame:
    """Add cold drinks promo flag based on temperature."""
    return df.withColumn("promo_cold_drinks", F.col("avg_tmpr_c") > 25.0)


# --- MAIN EXECUTION ---

def main():
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("RestaurantPreferenceTracking") \
        .getOrCreate()

    # Reduce log noise to see results clearly
    spark.sparkContext.setLogLevel("WARN")

    # --- 1. LOAD DATA ---
    receipts_raw = spark.read.option("header", "True").option("inferSchema", "True").csv("/workspace/receipt_restaurants")
    weather_raw = spark.read.option("header", "True").option("inferSchema", "True").csv("/workspace/weather")

    # --- 2. PROCESS 2022 DATA (INITIAL STATE) ---
    receipts_2022 = prepare_receipts_for_join(receipts_raw, 2022)
    weather_enrichment = prepare_weather_for_join(weather_raw)
    enriched_2022 = enrich_receipts_with_weather(receipts_2022, weather_enrichment)
    receipt_counts_2022 = compute_receipt_counts(enriched_2022)
    initial_state = build_initial_state(receipt_counts_2022)

    # Save Initial State to Local Storage
    initial_state.write.mode("overwrite").parquet("/workspace/output/initial_state_2022")
    print("2022 Initial State saved to /workspace/output/initial_state_2022")

    # --- 3. STREAMING 2021 DATA ---
    static_state = F.broadcast(initial_state)

    streaming_df = spark.readStream \
        .schema(receipts_raw.schema) \
        .option("header", "True") \
        .option("maxFilesPerTrigger", 1) \
        .csv("/workspace/receipt_restaurants")

    final_stream = streaming_df.filter(F.year(F.col("date_time")) == 2021) \
        .withColumn("date_only", F.to_date("date_time")) \
        .withColumn("lat_r", F.round("lat", 2)) \
        .withColumn("lng_r", F.round("lng", 2)) \
        .join(weather_enrichment, ["lat_r", "lng_r", "date_only"]) \
        .join(static_state, "restaurant_franchise_id")
    
    final_stream = apply_promo_logic(final_stream) \
        .withColumnRenamed("restaurant_franchise_id", "restaurant") \
        .select(
            "restaurant", 
            "promo_cold_drinks", 
            "batch_timestamp", 
            "erroneous_data_cnt", 
            "tiny_cnt", 
            "small_cnt", 
            "medium_cnt", 
            "large_cnt", 
            "most_popular_order_type"
        )

    # --- 4. SINK STREAM TO LOCAL STORAGE ---
    query = final_stream.writeStream \
        .format("parquet") \
        .option("path", "/workspace/output/final_enriched_2021") \
        .option("checkpointLocation", "/workspace/output/checkpoints") \
        .outputMode("append") \
        .start()

    print("Streaming started. Monitoring /workspace/output/final_enriched_2021...")
    query.awaitTermination()


if __name__ == "__main__":
    main()
