# Databricks notebook source
# DATAFRAME API COMPREHENSIVE PROJECT
# Dataset: Employee Sales Performance Data
# Using PySpark DataFrame API
# =============================================

# SECTION 1: SETUP AND DATA CREATION
# =============================================

# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Initialize Spark Session
# Note: In Databricks, spark session is already created
spark = SparkSession.builder.appName("DataFrameAPIProject").getOrCreate()

# Display Spark version and configuration
print("Spark Version:", spark.version)

# Get Data from Sample Data
# =============================================
# Create Employees DataFrame
employees_data = [
    (1, "John Smith", "Sales", 50000, "2020-01-15"),
    (2, "Jane Doe", "Sales", 55000, "2019-03-20"),
    (3, "Mike Johnson", "Marketing", 60000, "2018-06-10"),
    (4, "Sarah Wilson", "Sales", 52000, "2020-11-05"),
    (5, "David Brown", "IT", 70000, "2017-09-12"),
    (6, "Emily Davis", "Marketing", 58000, "2019-08-22"),
    (7, "Robert Lee", "Sales", 53000, "2021-02-28"),
    (8, "Lisa Garcia", "IT", 72000, "2016-12-01"),
    (9, "Tom Miller", "Sales", 51000, "2022-01-10"),
    (10, "Amy Clark", "Marketing", 59000, "2018-04-15")
]

employees_schema = StructType([
    StructField("employee_id", IntegerType(), True),
    StructField("employee_name", StringType(), True),
    StructField("department", StringType(), True),
    StructField("salary", IntegerType(), True),
    StructField("hire_date", StringType(), True)
])

employees_df = spark.createDataFrame(employees_data, employees_schema)

# Convert hire_date string to DateType
employees_df = employees_df.withColumn("hire_date", to_date(col("hire_date"), "yyyy-MM-dd"))

print("=== Employees DataFrame ===")
employees_df.show()
print("Schema:")
employees_df.printSchema()

# Create Sales DataFrame
sales_data = [
    (1, 1, "2024-01-15", 1500.00, "North", "Electronics"),
    (2, 2, "2024-01-16", 2000.00, "South", "Furniture"),
    (3, 1, "2024-01-17", 1200.00, "North", "Electronics"),
    (4, 3, "2024-01-18", 1800.00, "East", "Office Supplies"),
    (5, 4, "2024-01-19", 2200.00, "West", "Furniture"),
    (6, 2, "2024-01-20", 1900.00, "South", "Electronics"),
    (7, 1, "2024-01-21", 2100.00, "North", "Furniture"),
    (8, 5, "2024-01-22", 1700.00, "East", "Electronics"),
    (9, 4, "2024-01-23", 2400.00, "West", "Office Supplies"),
    (10, 3, "2024-01-24", 1600.00, "East", "Furniture"),
    (11, 2, "2024-02-01", 2300.00, "South", "Electronics"),
    (12, 1, "2024-02-02", 1900.00, "North", "Office Supplies"),
    (13, 4, "2024-02-03", 2100.00, "West", "Furniture"),
    (14, 3, "2024-02-04", 1700.00, "East", "Electronics"),
    (15, 2, "2024-02-05", 2500.00, "South", "Furniture")
]

sales_schema = StructType([
    StructField("sale_id", IntegerType(), True),
    StructField("employee_id", IntegerType(), True),
    StructField("sale_date", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("region", StringType(), True),
    StructField("product_category", StringType(), True)
])

sales_df = spark.createDataFrame(sales_data, sales_schema)

# Convert sale_date string to DateType
sales_df = sales_df.withColumn("sale_date", to_date(col("sale_date"), "yyyy-MM-dd"))

print("\n=== Sales DataFrame ===")
sales_df.show()
print("Schema:")
sales_df.printSchema()

# SECTION 3: BASIC DATAFRAME OPERATIONS
# =============================================

print("\n" + "="*50)
print("SECTION 3: BASIC DATAFRAME OPERATIONS")
print("="*50)

# Operation 1: Select and Filter
# Purpose: Demonstrate column selection and data filtering
print("\n--- 1. Select Specific Columns and Filter ---")
filtered_employees = employees_df.select(
    "employee_id", 
    "employee_name", 
    "department",
    "salary"
    ).filter(
        col("department") == "Sales"
    ).orderBy(
        col("salary").desc()
    )

print("Sales Department Employees (High to Low Salary):")
filtered_employees.show()

# Operation 2: Adding New Columns
# Purpose: Show how to create derived columns
print("\n--- 2. Adding Derived Columns ---")
employees_with_bonus = employees_df.withColumn(
    "annual_bonus", col("salary") * 0.1  # 10% bonus
).withColumn(
    "total_compensation", col("salary") + col("annual_bonus")
).withColumn(
    "years_of_service", year(current_date()) - year(col("hire_date"))
)

print("Employees with Bonus and Compensation Details:")
employees_with_bonus.select(
    "employee_name",
    "department",
    "salary",
    "annual_bonus",
    "total_compensation",
    "years_of_service"
).show()

# Operation 3: GroupBy and Aggregations
# Purpose: Demonstrate data aggregation
print("\n--- 3. Department-wise Aggregations ---")
department_stats = employees_df.groupBy("department").agg(
    count("employee_id").alias("employee_count"),
    avg("salary").alias("average_salary"),
    max("salary").alias("max_salary"),
    min("salary").alias("min_salary"),
    sum("salary").alias("total_salary_budget")
).orderBy(col("average_salary").desc())

print("Department-wise Statistics:")
department_stats.show()

# Operation 4: Joining DataFrames
# Purpose: Combine data from multiple DataFrames
print("\n--- 4. Joining Employees and Sales Data ---")
employee_sales = employees_df.alias("emp").join(
    sales_df.alias("sales"),
    col("emp.employee_id") == col("sales.employee_id"), 
    "inner"
).select(
    col("emp.employee_id"),
    col("emp.employee_name"),
    col("emp.department"),
    col("sales.sale_date"),
    col("sales.amount"),
    col("sales.region"),
    col("sales.product_category")
).orderBy(col("amount").desc())

print("Employees with Sales Data:")
employee_sales.show()

# SECTION 4: WINDOW FUNCTIONS - RANKING
# =============================================

print("\n" + "="*50)
print("SECTION 4: WINDOW FUNCTIONS - RANKING")
print("="*50)

# Window Specification: Define how to partition and order data
department_window = Window.partitionBy("department").orderBy(col("salary").desc())

# Operation 5: ROW_NUMBER()
# Purpose: Assign unique sequential numbers within each department
print("\n--- 5. ROW_NUMBER() - Unique Ranking ---")
employees_ranked = employees_df.withColumn(
    "row_number", row_number().over(department_window)
).withColumn(
    "salary_rank", rank().over(department_window)
).withColumn(
    "dense_rank", dense_rank().over(department_window)
)

print("Employees with Ranking Functions:")
employees_ranked.select(
    "employee_name", "department", "salary", 
    "row_number","salary_rank", "dense_rank"
).orderBy("department", col("salary").desc()).show()


# Operation 6: NTILE - Quartiles
# Purpose: Divide employees into salary quartiles within departments
print("\n--- 6. NTILE() - Salary Quartiles ---")
employees_with_quartiles = employees_df.withColumn(
    "salary_quartile", ntile(4).over(department_window)
)

print("Employees with Salary Quartiles:")
employees_with_quartiles.select(
    "employee_name", "department", "salary", "salary_quartile"
).orderBy("department", "salary_quartile").show()

# SECTION 5: WINDOW FUNCTIONS - ANALYTICAL
# =============================================

print("\n" + "="*50)
print("SECTION 5: WINDOW FUNCTIONS - ANALYTICAL")
print("="*50)

# Window for analytical functions (needs full window frame)
department_window_analytical = Window.partitionBy("department").orderBy("salary").rowsBetween(
    Window.unboundedPreceding, Window.unboundedFollowing
)

# Operation 7: LAG and LEAD
# Purpose: Compare with previous and next values
print("\n--- 7. LAG() and LEAD() - Compare Adjacent Values ---")
salary_comparision = employees_df.withColumn(
    "previous_salary", lag("salary", 1).over(Window.partitionBy("department").orderBy("salary"))
).withColumn(
    "next_salary", lead("salary", 1).over(Window.partitionBy("department").orderBy("salary"))
).withColumn(
    "salary_growth",
    when(col("previous_salary").isNull(), 0)
    .otherwise(col("salary") - col("previous_salary"))
).withColumn(
    "salary_decline",
    when(col("next_salary").isNull(), 0)
    .otherwise(col("next_salary") - col("salary"))
)

print("Salary Comparision with Previous/Next:")
salary_comparision.select(
    "employee_name", "department", "salary", "previous_salary", "next_salary", "salary_growth", "salary_decline"
).show()

# Operation 8: FIRST_VALUE and LAST_VALUE
# Purpose: Get extreme values in window
print("\n--- 8. FIRST_VALUE() and LAST_VALUE() ---")
salary_extremes = employees_df.withColumn(
    "department_min", first("salary").over(department_window_analytical)
).withColumn(
    "department_max", last("salary").over(department_window_analytical)
).withColumn(
    "diff_from_min", col("salary") - col("department_min")
).withColumn(
    "percent_of_max", (col("salary") / col("department_max")) * 100
)

print("Salary Extremes within Departments:")
salary_extremes.select(
    "employee_name", "department", "salary", "department_min", "department_max", "diff_from_min", "percent_of_max"
).show()

# SECTION 6: WINDOW FUNCTIONS - AGGREGATES
# =============================================

print("\n" + "="*50)
print("SECTION 6: WINDOW FUNCTIONS - AGGREGATES")
print("="*50)

# Operation 9: Running Totals and Moving Averages
# Purpose: Calculate cumulative and moving aggregates
print("\n--- 9. Running Totals and Moving Averages ---")

# Define window for running calculations
sales_window = Window.partitionBy("employee_id").orderBy("sale_date").rowsBetween(
    Window.unboundedPreceding, Window.currentRow
)

# Define window for moving average (3 rows)
moving_avg_window = Window.partitionBy("employee_id").orderBy("sale_date").rowsBetween(
    -2, Window.currentRow
)

sales_analysis = employee_sales.withColumn(
    "running_total", sum("amount").over(sales_window)
).withColumn(
    "moving_avg_3", avg("amount").over(moving_avg_window)
).withColumn(
    "total_employee_sales", sum("amount").over(Window.partitionBy("employee_name"))
).withColumn(
    "percent_of_total", (col("amount") / col("total_employee_sales")) * 100
)

print("Sales Analysis with Running Totals:")
sales_analysis.select(
    "employee_name", "sale_date", "amount",
    "running_total", "moving_avg_3",
    "percent_of_total"
).orderBy("employee_name", "sale_date").show()


# SECTION 7: COMPLEX WINDOW SCENARIOS
# =============================================

print("\n" + "="*50)
print("SECTION 7: COMPLEX WINDOW SCENARIOS")
print("="*50)

# Operation 10: Multiple Window Functions Combined
# Purpose: Comprehensive analysis using multiple windows
print("\n--- 10. Comprehensive Salary Analysis ---")

# Define multiple window specifications
dept_window_rank = Window.partitionBy("department").orderBy(col("salary").desc())
dept_window_agg = Window.partitionBy("department")

comprehensive_analysis = employees_df.withColumn(
    "dept_rank", rank().over(dept_window_rank)
).withColumn(
    "dept_avg_salary", avg("salary").over(dept_window_agg)
).withColumn(
    "dept_max_salary", max("salary").over(dept_window_agg)
).withColumn(
    "dept_min_salary", min("salary").over(dept_window_agg)
).withColumn(
    "diff_from_avg", col("salary") - col("dept_avg_salary")
).withColumn(
    "percent_of_max", (col("salary") / col("dept_max_salary")) * 100
).withColumn(
    "salary_ratio", col("salary") / col("dept_avg_salary")
)

print("Comprehensive Salary Analysis:")
comprehensive_analysis.select(
    "employee_name", "department", "salary", "dept_rank",
    "dept_avg_salary", "diff_from_avg", "percent_of_max", "salary_ratio"
).orderBy("department", "dept_rank").show()

# Operation 11: Sales Performance with Multiple Dimensions
print("\n--- 11. Multi-dimensional Sales Analysis ---")

# Define multiple windows for different dimensions
region_window = Window.partitionBy("region").orderBy(col("amount").desc())
category_window = Window.partitionBy("product_category").orderBy(col("amount").desc())
employee_region_window = Window.partitionBy("employee_name", "region")

sales_performance = employee_sales.withColumn(
    "regional_rank", rank().over(region_window)
).withColumn(
    "category_rank", rank().over(category_window)
).withColumn(
    "employee_region_avg", avg("amount").over(employee_region_window)
).withColumn(
    "regional_avg", avg("amount").over(Window.partitionBy("region"))
).withColumn(
    "performance_vs_region", col("amount") - col("regional_avg")
)

print("Sales Performance Analysis:")
sales_performance.select(
    "employee_name", "region", "product_category", "amount",
    "regional_rank", "category_rank", 
    "employee_region_avg", "regional_avg", "performance_vs_region"
).orderBy("region", "regional_rank").show()

# SECTION 8: PRACTICAL BUSINESS SCENARIOS
# =============================================

print("\n" + "="*50)
print("SECTION 8: PRACTICAL BUSINESS SCENARIOS")
print("="*50)

# Scenario 1: Employee Performance Bonus Calculation
print("\n--- Scenario 1: Performance Bonus Calculation ---")

bonus_calculation = comprehensive_analysis.withColumn(
    "bonus_tier",
    when(col("dept_rank") == 1, "Gold Bonus: 20%")
    .when(col("dept_rank") == 2, "Silver Bonus: 15%")
    .when(col("dept_rank") == 3, "Bronze Bonus: 10%")
    .otherwise("Standard Bonus: 5%")
).withColumn(
    "bonus_amount",
    when(col("dept_rank") == 1, col("salary") * 0.20)
    .when(col("dept_rank") == 2, col("salary") * 0.15)
    .when(col("dept_rank") == 3, col("salary") * 0.10)
    .otherwise(col("salary") * 0.05)
)

print("Employee Bonus Calculation:")
bonus_calculation.select(
    "employee_name", "department", "salary", "dept_rank",
    "bonus_tier", "bonus_amount"
).orderBy("department", "dept_rank").show()

# Scenario 2: Sales Trend Analysis
print("\n--- Scenario 2: Sales Trend Analysis ---")

sales_trend_window = Window.partitionBy("employee_name").orderBy("sale_date")

sales_trend_analysis = employee_sales.withColumn(
    "previous_sale", lag("amount", 1).over(sales_trend_window)
).withColumn(
    "sales_trend",
    when(lag("amount", 1).over(sales_trend_window).isNull(), "First Sale")
    .when(col("amount") > lag("amount", 1).over(sales_trend_window), "Growth")
    .when(col("amount") < lag("amount", 1).over(sales_trend_window), "Decline")
    .otherwise("Stable")
).withColumn(
    "growth_percentage",
    when(lag("amount", 1).over(sales_trend_window).isNull(), 0)
    .otherwise(((col("amount") - lag("amount", 1).over(sales_trend_window)) / 
                lag("amount", 1).over(sales_trend_window)) * 100)
).filter(
    col("sales_trend").isNotNull()  # Remove rows where trend cannot be calculated
)

print("Sales Trend Analysis:")
sales_trend_analysis.select(
    "employee_name", "sale_date", "amount", "previous_sale",
    "sales_trend", round("growth_percentage", 2).alias("growth_percentage")
).orderBy("employee_name", "sale_date").show()

# SECTION 9: DATAFRAME WRITING AND EXPORT
# =============================================

print("\n" + "="*50)
print("SECTION 9: EXPORTING RESULTS")
print("="*50)

# Save results to Delta tables (Databricks native format)
print("\n--- Saving Results to Delta Tables ---")

# Save employees with rankings
employees_ranked.write.mode("overwrite").saveAsTable("employee_rankings")

# Save sales analysis
sales_analysis.write.mode("overwrite").saveAsTable("sales_analysis")

# Save bonus calculation
bonus_calculation.write.mode("overwrite").saveAsTable("employee_bonuses")

print("Results saved to Delta tables:")
print("- employee_rankings")
print("- sales_analysis") 
print("- employee_bonuses")

# Display saved tables
print("\n--- Displaying Saved Tables ---")
spark.sql("SHOW TABLES").show()

# SECTION 10: PERFORMANCE OPTIMIZATION TIPS
# =============================================

print("\n" + "="*50)
print("SECTION 10: PERFORMANCE OPTIMIZATION TIPS")
print("="*50)

print("""
DataFrame API Performance Best Practices:

1. **Use Column Operations**: Always prefer column operations over UDFs when possible
2. **Partition Wisely**: Window functions perform better with appropriate partitioning
3. **Avoid Too Many Partitions**: Too many small partitions can degrade performance
4. **Use Built-in Functions**: Spark's built-in functions are optimized
5. **Cache Frequently Used DataFrames**: Use df.cache() for DataFrames used multiple times
6. **Select Only Needed Columns**: Reduce data movement by selecting necessary columns

Example of caching:
employees_df.cache()  # Cache for repeated use
""")

# Cache example
employees_df.cache()
print(f"Employees DataFrame cached: {employees_df.is_cached}")

# Clean up cache (optional)
# employees_df.unpersist()

print("\n" + "="*50)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*50)
 