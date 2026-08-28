-- Spark SQL analytics queries, run against temp views registered by
-- src/transformation/sql_runner.py:
--
--   transactions  -- the cleansed transaction fact table (transactions_cleaned)
--   region_dim    -- small static region -> sales_territory lookup (5 rows)
--
-- sql_runner.py also opportunistically registers the 5 tables written by
-- src/transformation/transform.py (daily_revenue, monthly_revenue,
-- customer_rfm, top_products, payment_distribution) when they're present on
-- disk, but the queries below are written against `transactions` directly
-- so this file works standalone against just the cleansed data.
--
-- Each query is delimited by a `-- @name:` / `-- @description:` header pair
-- that sql_runner.py parses with a regex -- keep that two-line header
-- format exactly (one query per header, ending at the next `-- @name:` or
-- end of file).


-- =====================================================================
-- SECTION 1: Revenue trends over time
-- =====================================================================

-- @name: revenue_trend_daily
-- @description: Daily revenue with a 7-day trailing moving average to smooth day-to-day noise.
SELECT
    purchase_date,
    daily_revenue,
    ROUND(
        AVG(daily_revenue) OVER (
            ORDER BY purchase_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
        2
    ) AS revenue_7day_moving_avg
FROM (
    SELECT purchase_date, ROUND(SUM(total_amount), 2) AS daily_revenue
    FROM transactions
    WHERE purchase_date IS NOT NULL
    GROUP BY purchase_date
) daily
ORDER BY purchase_date;


-- @name: revenue_trend_monthly_growth
-- @description: Monthly revenue with month-over-month growth percentage via LAG().
SELECT
    purchase_year_month,
    monthly_revenue,
    LAG(monthly_revenue) OVER (ORDER BY purchase_year_month) AS prior_month_revenue,
    ROUND(
        (monthly_revenue - LAG(monthly_revenue) OVER (ORDER BY purchase_year_month))
        / LAG(monthly_revenue) OVER (ORDER BY purchase_year_month) * 100,
        2
    ) AS mom_growth_pct
FROM (
    SELECT
        date_format(purchase_date, 'yyyy-MM') AS purchase_year_month,
        ROUND(SUM(total_amount), 2) AS monthly_revenue
    FROM transactions
    WHERE purchase_date IS NOT NULL
    GROUP BY date_format(purchase_date, 'yyyy-MM')
) monthly
ORDER BY purchase_year_month;


-- @name: revenue_trend_category_share
-- @description: Monthly revenue by category alongside each category's share of that month's total.
SELECT
    purchase_year_month,
    category,
    category_revenue,
    ROUND(
        category_revenue / SUM(category_revenue) OVER (PARTITION BY purchase_year_month) * 100,
        2
    ) AS pct_of_month_revenue
FROM (
    SELECT
        date_format(purchase_date, 'yyyy-MM') AS purchase_year_month,
        category,
        ROUND(SUM(total_amount), 2) AS category_revenue
    FROM transactions
    WHERE purchase_date IS NOT NULL
    GROUP BY date_format(purchase_date, 'yyyy-MM'), category
) monthly_category
ORDER BY purchase_year_month, category_revenue DESC;


-- =====================================================================
-- SECTION 2: Customer lifetime value ranking
-- =====================================================================

-- @name: customer_ltv_ranking
-- @description: Every known customer's lifetime value, order count, tenure, rank, and LTV quartile.
SELECT
    customer_id,
    ROUND(SUM(total_amount), 2) AS lifetime_value,
    COUNT(*) AS total_orders,
    MIN(purchase_date) AS first_purchase_date,
    MAX(purchase_date) AS last_purchase_date,
    DATEDIFF(MAX(purchase_date), MIN(purchase_date)) AS customer_tenure_days,
    RANK() OVER (ORDER BY SUM(total_amount) DESC) AS ltv_rank,
    NTILE(4) OVER (ORDER BY SUM(total_amount) DESC) AS ltv_quartile
FROM transactions
WHERE customer_id != 'UNKNOWN_CUSTOMER'
GROUP BY customer_id
ORDER BY lifetime_value DESC;


-- @name: customer_ltv_by_segment
-- @description: Average and median customer LTV per customer_segment (New/Regular/Premium/VIP).
SELECT
    customer_segment,
    COUNT(DISTINCT customer_id) AS customer_count,
    ROUND(SUM(customer_total), 2) AS segment_revenue,
    ROUND(AVG(customer_total), 2) AS avg_customer_ltv,
    ROUND(percentile_approx(customer_total, 0.5), 2) AS median_customer_ltv
FROM (
    SELECT customer_id, customer_segment, SUM(total_amount) AS customer_total
    FROM transactions
    WHERE customer_id != 'UNKNOWN_CUSTOMER'
    GROUP BY customer_id, customer_segment
) per_customer
GROUP BY customer_segment
ORDER BY avg_customer_ltv DESC;


-- @name: customer_ltv_top_per_region
-- @description: Top 5 customers by lifetime value within each region.
SELECT region, customer_id, lifetime_value, region_ltv_rank
FROM (
    SELECT
        region,
        customer_id,
        ROUND(SUM(total_amount), 2) AS lifetime_value,
        RANK() OVER (PARTITION BY region ORDER BY SUM(total_amount) DESC) AS region_ltv_rank
    FROM transactions
    WHERE customer_id != 'UNKNOWN_CUSTOMER'
    GROUP BY region, customer_id
) ranked
WHERE region_ltv_rank <= 5
ORDER BY region, region_ltv_rank;


-- =====================================================================
-- SECTION 3: Cohort retention analysis
-- =====================================================================

-- @name: cohort_retention_rate
-- @description: Month-by-month retention rate per acquisition cohort (cohort_month = first purchase month).
WITH valid_transactions AS (
    -- Excluding the UNKNOWN_CUSTOMER sentinel and null purchase dates here,
    -- shared by both branches below, matters for correctness as much as
    -- performance -- see sql/query_optimization_notes.md, comparison #3.
    SELECT customer_id, purchase_date
    FROM transactions
    WHERE customer_id != 'UNKNOWN_CUSTOMER' AND purchase_date IS NOT NULL
),
first_purchase AS (
    SELECT customer_id, date_trunc('month', MIN(purchase_date)) AS cohort_month
    FROM valid_transactions
    GROUP BY customer_id
),
monthly_activity AS (
    SELECT DISTINCT customer_id, date_trunc('month', purchase_date) AS activity_month
    FROM valid_transactions
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(*) AS cohort_size
    FROM first_purchase
    GROUP BY cohort_month
),
cohort_activity AS (
    SELECT
        f.cohort_month,
        (YEAR(a.activity_month) - YEAR(f.cohort_month)) * 12
            + (MONTH(a.activity_month) - MONTH(f.cohort_month)) AS month_number,
        COUNT(DISTINCT a.customer_id) AS active_customers
    FROM first_purchase f
    JOIN monthly_activity a ON f.customer_id = a.customer_id
    GROUP BY f.cohort_month, month_number
)
SELECT
    ca.cohort_month,
    ca.month_number,
    cs.cohort_size,
    ca.active_customers,
    ROUND(ca.active_customers / cs.cohort_size * 100, 2) AS retention_rate_pct
FROM cohort_activity ca
JOIN cohort_sizes cs ON ca.cohort_month = cs.cohort_month
ORDER BY ca.cohort_month, ca.month_number;


-- @name: cohort_retention_summary
-- @description: Cohort sizes with month 1 / 3 / 6 retention rates pivoted into columns.
WITH valid_transactions AS (
    SELECT customer_id, purchase_date
    FROM transactions
    WHERE customer_id != 'UNKNOWN_CUSTOMER' AND purchase_date IS NOT NULL
),
first_purchase AS (
    SELECT customer_id, date_trunc('month', MIN(purchase_date)) AS cohort_month
    FROM valid_transactions
    GROUP BY customer_id
),
monthly_activity AS (
    SELECT DISTINCT customer_id, date_trunc('month', purchase_date) AS activity_month
    FROM valid_transactions
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(*) AS cohort_size
    FROM first_purchase
    GROUP BY cohort_month
),
cohort_activity AS (
    SELECT
        f.cohort_month,
        (YEAR(a.activity_month) - YEAR(f.cohort_month)) * 12
            + (MONTH(a.activity_month) - MONTH(f.cohort_month)) AS month_number,
        COUNT(DISTINCT a.customer_id) AS active_customers
    FROM first_purchase f
    JOIN monthly_activity a ON f.customer_id = a.customer_id
    GROUP BY f.cohort_month, month_number
)
SELECT
    cs.cohort_month,
    cs.cohort_size,
    ROUND(SUM(CASE WHEN ca.month_number = 1 THEN ca.active_customers END) / cs.cohort_size * 100, 2)
        AS month_1_retention_pct,
    ROUND(SUM(CASE WHEN ca.month_number = 3 THEN ca.active_customers END) / cs.cohort_size * 100, 2)
        AS month_3_retention_pct,
    ROUND(SUM(CASE WHEN ca.month_number = 6 THEN ca.active_customers END) / cs.cohort_size * 100, 2)
        AS month_6_retention_pct
FROM cohort_sizes cs
LEFT JOIN cohort_activity ca ON cs.cohort_month = ca.cohort_month
GROUP BY cs.cohort_month, cs.cohort_size
ORDER BY cs.cohort_month;


-- =====================================================================
-- SECTION 4: Regional performance comparison
-- =====================================================================

-- @name: regional_performance_summary
-- @description: Revenue, orders, unique customers, and AOV per region, joined to a sales-territory dimension.
SELECT /*+ BROADCAST(d) */
    t.region,
    d.sales_territory,
    ROUND(SUM(t.total_amount), 2) AS revenue,
    COUNT(*) AS total_orders,
    COUNT(DISTINCT t.customer_id) AS unique_customers,
    ROUND(AVG(t.total_amount), 2) AS avg_order_value,
    RANK() OVER (ORDER BY SUM(t.total_amount) DESC) AS revenue_rank
FROM transactions t
JOIN region_dim d ON t.region = d.region
GROUP BY t.region, d.sales_territory
ORDER BY revenue DESC;


-- @name: regional_category_breakdown
-- @description: Revenue by region x category, with each category's share of its region's total revenue.
SELECT
    region,
    category,
    ROUND(SUM(total_amount), 2) AS revenue,
    ROUND(SUM(total_amount) / SUM(SUM(total_amount)) OVER (PARTITION BY region) * 100, 2)
        AS pct_of_region_revenue
FROM transactions
GROUP BY region, category
ORDER BY region, revenue DESC;


-- @name: regional_yoy_comparison
-- @description: Yearly revenue per region with year-over-year growth percentage.
SELECT
    region,
    purchase_year,
    yearly_revenue,
    LAG(yearly_revenue) OVER (PARTITION BY region ORDER BY purchase_year) AS prior_year_revenue,
    ROUND(
        (yearly_revenue - LAG(yearly_revenue) OVER (PARTITION BY region ORDER BY purchase_year))
        / LAG(yearly_revenue) OVER (PARTITION BY region ORDER BY purchase_year) * 100,
        2
    ) AS yoy_growth_pct
FROM (
    SELECT
        region,
        YEAR(purchase_date) AS purchase_year,
        ROUND(SUM(total_amount), 2) AS yearly_revenue
    FROM transactions
    WHERE purchase_date IS NOT NULL
    GROUP BY region, YEAR(purchase_date)
) yearly
ORDER BY region, purchase_year;
