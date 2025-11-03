# PySpark DataFrame API Project - Databricks Edition

## Project Overview
This project demonstrates comprehensive PySpark DataFrame API operations with emphasis on Window Functions using Databricks Free Edition.

## Features
- Basic DataFrame Operations (select, filter, groupBy, agg, join)
- Window Functions (row_number, rank, dense_rank, ntile)
- Analytical Functions (lag, lead, first_value, last_value)
- Aggregate Window Functions (running totals, moving averages)
- Practical Business Scenarios

## Setup Instructions

### 1. Databricks Setup
- Sign up for [Databricks Community Edition](https://community.cloud.databricks.com/)
- Create a cluster with Databricks Runtime 10.4+
- Create a new Python notebook

### 2. Run the Project
- Copy and paste the Python code from `notebooks/dataframe_api_project.py`
- Execute the notebook

## Key Learning Outcomes

### Basic DataFrame Operations
- Column selection and filtering
- Adding derived columns
- Aggregations and groupBy
- DataFrame joins

### Window Functions
- **Ranking Functions**: row_number(), rank(), dense_rank(), ntile()
- **Analytical Functions**: lag(), lead(), first_value(), last_value()
- **Aggregate Window Functions**: Running totals, moving averages

### Practical Applications
- Employee performance analysis
- Sales trend identification
- Business intelligence reporting

## Code Examples

### Basic DataFrame Operations
```python
# Select and filter
filtered_df = df.select("col1", "col2").filter(col("salary") > 50000)

# GroupBy aggregations
stats_df = df.groupBy("department").agg(avg("salary"), max("salary"))