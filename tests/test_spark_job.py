"""Unit tests for spark_job transformation functions."""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    TimestampType, IntegerType, DateType
)
from datetime import datetime, date

from spark_job import (
    prepare_receipts_for_join,
    prepare_weather_for_join,
    enrich_receipts_with_weather,
    categorize_order_size,
    compute_receipt_counts,
    build_initial_state,
    apply_promo_logic,
)


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    return SparkSession.builder \
        .master("local[2]") \
        .appName("test_spark_job") \
        .getOrCreate()


@pytest.fixture
def sample_receipts(spark):
    """Sample receipts DataFrame for testing."""
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("receipt_id", StringType(), True),
        StructField("restaurant_franchise_id", StringType(), True),
        StructField("date_time", TimestampType(), True),
        StructField("lat", DoubleType(), True),
        StructField("lng", DoubleType(), True),
        StructField("total_cost", DoubleType(), True),
        StructField("discount", DoubleType(), True),
    ])
    data = [
        (1, "R001", "FRAN_A", datetime(2022, 6, 15, 12, 0), 40.7128, -74.0060, 25.0, 5.0),
        (2, "R001", "FRAN_A", datetime(2022, 6, 15, 12, 0), 40.7128, -74.0060, 15.0, 2.0),
        (3, "R002", "FRAN_A", datetime(2022, 6, 15, 12, 0), 40.7128, -74.0060, 10.0, 1.0),
        (4, "R003", "FRAN_B", datetime(2021, 6, 15, 12, 0), 51.5074, -0.1278, 30.0, 3.0),
        (5, "R004", "FRAN_B", datetime(2022, 7, 20, 14, 0), 51.5074, -0.1278, 20.0, 2.0),
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_weather(spark):
    """Sample weather DataFrame for testing."""
    schema = StructType([
        StructField("lat", DoubleType(), True),
        StructField("lng", DoubleType(), True),
        StructField("wthr_date", DateType(), True),
        StructField("avg_tmpr_c", DoubleType(), True),
    ])
    data = [
        (40.71, -74.01, date(2022, 6, 15), 28.5),
        (51.51, -0.13, date(2022, 7, 20), 22.0),
        (51.51, -0.13, date(2021, 6, 15), 18.0),
    ]
    return spark.createDataFrame(data, schema)


class TestPrepareReceiptsForJoin:
    """Tests for prepare_receipts_for_join function."""

    def test_filters_by_year(self, spark, sample_receipts):
        result = prepare_receipts_for_join(sample_receipts, 2022)
        years = [row.date_only.year for row in result.collect()]
        assert all(y == 2022 for y in years)
        assert result.count() == 4  # 4 records from 2022

    def test_adds_date_only_column(self, spark, sample_receipts):
        result = prepare_receipts_for_join(sample_receipts, 2022)
        assert "date_only" in result.columns

    def test_adds_rounded_coordinates(self, spark, sample_receipts):
        result = prepare_receipts_for_join(sample_receipts, 2022)
        assert "lat_r" in result.columns
        assert "lng_r" in result.columns
        row = result.first()
        assert row.lat_r == 40.71
        assert row.lng_r == -74.01


class TestPrepareWeatherForJoin:
    """Tests for prepare_weather_for_join function."""

    def test_renames_date_column(self, spark, sample_weather):
        result = prepare_weather_for_join(sample_weather)
        assert "date_only" in result.columns
        assert "wthr_date" not in result.columns

    def test_rounds_coordinates(self, spark, sample_weather):
        result = prepare_weather_for_join(sample_weather)
        assert "lat_r" in result.columns
        assert "lng_r" in result.columns

    def test_selects_only_needed_columns(self, spark, sample_weather):
        result = prepare_weather_for_join(sample_weather)
        assert set(result.columns) == {"lat_r", "lng_r", "date_only", "avg_tmpr_c"}


class TestEnrichReceiptsWithWeather:
    """Tests for enrich_receipts_with_weather function."""

    def test_joins_on_location_and_date(self, spark, sample_receipts, sample_weather):
        receipts = prepare_receipts_for_join(sample_receipts, 2022)
        weather = prepare_weather_for_join(sample_weather)
        result = enrich_receipts_with_weather(receipts, weather)
        assert result.count() > 0
        assert "avg_tmpr_c" in result.columns

    def test_filters_positive_temperature(self, spark):
        """Ensure records with temperature <= 0 are filtered out."""
        receipts_schema = StructType([
            StructField("id", IntegerType()),
            StructField("date_time", TimestampType()),
            StructField("lat", DoubleType()),
            StructField("lng", DoubleType()),
            StructField("total_cost", DoubleType()),
            StructField("discount", DoubleType()),
        ])
        receipts = spark.createDataFrame([
            (1, datetime(2022, 1, 1, 12, 0), 10.0, 20.0, 100.0, 10.0),
        ], receipts_schema)
        receipts = prepare_receipts_for_join(receipts, 2022)

        weather_schema = StructType([
            StructField("lat", DoubleType()),
            StructField("lng", DoubleType()),
            StructField("wthr_date", DateType()),
            StructField("avg_tmpr_c", DoubleType()),
        ])
        weather = spark.createDataFrame([
            (10.0, 20.0, date(2022, 1, 1), -5.0),  # negative temp
        ], weather_schema)
        weather = prepare_weather_for_join(weather)

        result = enrich_receipts_with_weather(receipts, weather)
        assert result.count() == 0

    def test_calculates_original_total_cost(self, spark, sample_receipts, sample_weather):
        receipts = prepare_receipts_for_join(sample_receipts, 2022)
        weather = prepare_weather_for_join(sample_weather)
        result = enrich_receipts_with_weather(receipts, weather)
        assert "original_total_cost" in result.columns
        row = result.first()
        assert row.original_total_cost == row.total_cost + row.discount


class TestCategorizeOrderSize:
    """Tests for categorize_order_size function."""

    def test_erroneous_for_null(self, spark):
        df = spark.createDataFrame([(None,)], ["items_count"])
        result = df.withColumn("cat", categorize_order_size("items_count"))
        assert result.first().cat == "erroneous_data_cnt"

    def test_erroneous_for_zero(self, spark):
        df = spark.createDataFrame([(0,)], ["items_count"])
        result = df.withColumn("cat", categorize_order_size("items_count"))
        assert result.first().cat == "erroneous_data_cnt"

    def test_erroneous_for_negative(self, spark):
        df = spark.createDataFrame([(-1,)], ["items_count"])
        result = df.withColumn("cat", categorize_order_size("items_count"))
        assert result.first().cat == "erroneous_data_cnt"

    def test_tiny_for_one_item(self, spark):
        df = spark.createDataFrame([(1,)], ["items_count"])
        result = df.withColumn("cat", categorize_order_size("items_count"))
        assert result.first().cat == "tiny_cnt"

    def test_small_for_two_to_three_items(self, spark):
        df = spark.createDataFrame([(2,), (3,)], ["items_count"])
        result = df.withColumn("cat", categorize_order_size("items_count"))
        categories = [row.cat for row in result.collect()]
        assert all(c == "small_cnt" for c in categories)

    def test_medium_for_four_to_ten_items(self, spark):
        df = spark.createDataFrame([(4,), (10,)], ["items_count"])
        result = df.withColumn("cat", categorize_order_size("items_count"))
        categories = [row.cat for row in result.collect()]
        assert all(c == "medium_cnt" for c in categories)

    def test_large_for_more_than_ten_items(self, spark):
        df = spark.createDataFrame([(11,), (100,)], ["items_count"])
        result = df.withColumn("cat", categorize_order_size("items_count"))
        categories = [row.cat for row in result.collect()]
        assert all(c == "large_cnt" for c in categories)


class TestComputeReceiptCounts:
    """Tests for compute_receipt_counts function."""

    def test_groups_by_receipt_id(self, spark):
        schema = StructType([
            StructField("id", IntegerType()),
            StructField("receipt_id", StringType()),
            StructField("restaurant_franchise_id", StringType()),
        ])
        data = [
            (1, "R001", "FRAN_A"),
            (2, "R001", "FRAN_A"),
            (3, "R001", "FRAN_A"),
            (4, "R002", "FRAN_A"),
        ]
        df = spark.createDataFrame(data, schema)
        result = compute_receipt_counts(df)
        
        r001 = result.filter(F.col("receipt_id") == "R001").first()
        r002 = result.filter(F.col("receipt_id") == "R002").first()
        
        assert r001.items_count == 3
        assert r002.items_count == 1


class TestBuildInitialState:
    """Tests for build_initial_state function."""

    def test_creates_pivot_columns(self, spark):
        schema = StructType([
            StructField("restaurant_franchise_id", StringType()),
            StructField("receipt_id", StringType()),
            StructField("items_count", IntegerType()),
        ])
        data = [
            ("FRAN_A", "R001", 1),   # tiny
            ("FRAN_A", "R002", 5),   # medium
            ("FRAN_A", "R003", 15),  # large
        ]
        df = spark.createDataFrame(data, schema)
        result = build_initial_state(df)
        
        expected_cols = {"restaurant_franchise_id", "erroneous_data_cnt", "tiny_cnt", 
                        "small_cnt", "medium_cnt", "large_cnt", "batch_timestamp",
                        "most_popular_order_type"}
        assert expected_cols.issubset(set(result.columns))

    def test_calculates_most_popular_order_type(self, spark):
        schema = StructType([
            StructField("restaurant_franchise_id", StringType()),
            StructField("receipt_id", StringType()),
            StructField("items_count", IntegerType()),
        ])
        # 3 medium orders should make medium the most popular
        data = [
            ("FRAN_A", "R001", 5),   # medium
            ("FRAN_A", "R002", 6),   # medium
            ("FRAN_A", "R003", 7),   # medium
            ("FRAN_A", "R004", 1),   # tiny
        ]
        df = spark.createDataFrame(data, schema)
        result = build_initial_state(df)
        row = result.first()
        assert row.most_popular_order_type == "medium_cnt"


class TestApplyPromoLogic:
    """Tests for apply_promo_logic function."""

    def test_promo_true_above_25_degrees(self, spark):
        df = spark.createDataFrame([(30.0,)], ["avg_tmpr_c"])
        result = apply_promo_logic(df)
        assert result.first().promo_cold_drinks is True

    def test_promo_false_at_25_degrees(self, spark):
        df = spark.createDataFrame([(25.0,)], ["avg_tmpr_c"])
        result = apply_promo_logic(df)
        assert result.first().promo_cold_drinks is False

    def test_promo_false_below_25_degrees(self, spark):
        df = spark.createDataFrame([(20.0,)], ["avg_tmpr_c"])
        result = apply_promo_logic(df)
        assert result.first().promo_cold_drinks is False
