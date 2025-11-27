# Abu Dhabi Real Estate Peer Analysis

A comprehensive analytics tool for comparing real estate project performance in Abu Dhabi. This application helps you analyze transaction trends, benchmark projects against their peers, and make data-driven insights about the real estate market.

## Table of Contents
- [Data Cleaning Process](#data-cleaning-process)
- [How to Use](#how-to-use)
- [Understanding the Analysis](#understanding-the-analysis)
- [Interpreting the Results](#interpreting-the-results)

---

## Data Cleaning Process

The application automatically cleans and validates your data through a multi-step process to ensure accurate analysis:

### Step 1: Remove Invalid Records
- **Missing Registration Dates**: Transactions without valid dates are removed
- **Missing Price or Sold Area**: Records lacking price or area information are excluded (required for rate calculation)

### Step 2: Filter by Share Ownership
- Only keeps transactions where **Share = 1** (100% ownership)
- Partial ownership transactions are excluded to avoid double-counting

### Step 3: Remove Non-Positive Values
- Filters out records with zero or negative values in:
  - Sold Area / GFA (sqm)
  - Plot Area (sqm)
  - Rate (AED/sqm)
  - Price (AED)

### Step 4: Exclude Private Projects
- Removes transactions from projects containing "Private" in the name
- Focuses analysis on publicly available market data

### Step 5: Rate Validation & Calculation
- Automatically calculates Rate (AED/sqm) = Price (AED) / Sold Area (sqm)
- Validates existing rate data against calculated values (allows 1% tolerance)
- Flags mismatches for data quality review

### Step 6: Outlier Removal
- Uses **IQR (Interquartile Range) method** with k=3.0
- Removes extreme outliers within each project/community/district
- Groups data by Project > Community > District (in priority order)
- Applies separately to Rate and Price to catch both types of anomalies

### Cleaning Summary
After cleaning, you'll see statistics like:
- **Original rows**: Total records in source file
- **Deleted rows**: Records removed during cleaning (with percentage)
- **Kept rows**: Clean records used for analysis (with percentage)

---

## How to Use

### Peer Analysis Page

#### 1. Configure Your Analysis

**Data Overview**
- View total cleaned rows, date range, and update frequency (Biweekly)

**Peer Group Dimension**
- Choose how to group projects:
  - **Community**: Compare projects within the same community
  - **Project**: Compare individual projects
  - **District**: Compare broader district-level trends

**Metric Selection**
- **Rate (AED/sqft)**: Price per square foot (default, more intuitive)
- **Rate (AED/sqm)**: Price per square meter (metric system)

**Aggregation Method**
- **Median** (recommended): Reduces impact of luxury outliers
- **Mean**: Simple average, more sensitive to extreme values

**Time Frequency**
- **Monthly**: Month-by-month granularity
- **Quarterly**: Smoothed quarterly view

#### 2. Select Time Window

Choose from preset ranges:
- **1M, 3M, 6M**: Short-term trends (1, 3, or 6 months)
- **1Y**: One year view (default)
- **3Y, 5Y**: Multi-year trends
- **Max**: All available data

**Custom Date Range**
- Use the date pickers to select any custom start and end dates
- Custom dates override the preset time window selections

#### 3. Select Projects to Compare

- Browse the alphabetically sorted list of available projects
- Multiple selection allowed
- Default projects are pre-selected based on common interest
- Only projects with transactions in your selected time window appear

#### 4. View Analysis Results

**Relative Performance Metrics**
- **Top Group**: Best performing project (% growth over period)
- **Weakest Group**: Lowest performing project (% growth over period)

**Normalized Trend Chart**
- All projects start at baseline = 1.0
- Final value of 1.25 = 25% increase over the period
- Final value of 0.90 = 10% decrease over the period
- Easy visual comparison of relative performance

**Individual vs Peer Average**
- For each selected project, see two charts:
  1. **Project vs Peer Average**: Direct comparison line chart
  2. **Delta (Minus Peer Average)**: Shows outperformance (positive) or underperformance (negative)

---

### Raw Data Page

#### 1. View Data Cleaning Statistics
- See metrics for original, deleted, and cleaned rows
- Understand data quality at a glance

#### 2. Filter Data

**By Registration Date**
- Select start and end dates to filter transactions

**By Dimensions** (all optional, can combine multiple)
- **Select Project(s)**: Choose specific projects
- **Select Community(ies)**: Filter by community
- **Select District(s)**: Filter by district

*Leave filters empty to show all records*

#### 3. Customize Display

**Select Columns**
- Choose which columns to display in the table
- Default columns: Registration, Project, Community, District, Rate, Price, Sold Area
- Can add/remove any available columns

#### 4. Download Data

- **Download as CSV**: For Excel, data analysis tools
- **Download as Excel**: Native Excel format with preserved formatting
- Filenames include current date for easy organization

#### 5. Summary Statistics (Optional)
- Check "Show Summary Statistics" to see:
  - Count, mean, std, min, 25%, 50%, 75%, max
  - For all numeric columns in your filtered view

---

## Understanding the Analysis

### What is Normalization?

**Normalization (base=1)** transforms all projects to start at the same baseline:
- **Purpose**: Makes projects comparable regardless of their absolute price levels
- **Calculation**: Each project's first value in the time window = 1.0
- **Subsequent values**: Ratio compared to first value

**Example**:
- Project A: Starts at 2,000 AED/sqft -> normalized to 1.0
- After 6 months: 2,200 AED/sqft -> normalized to 1.10 (10% increase)
- Project B: Starts at 1,500 AED/sqft -> normalized to 1.0
- After 6 months: 1,575 AED/sqft -> normalized to 1.05 (5% increase)

**Result**: You can easily see Project A (+10%) outperformed Project B (+5%), even though their absolute prices differ.

### Understanding Rate Metrics

**Rate (AED/sqft) vs Rate (AED/sqm)**
- Both measure the same thing in different units
- **sqft (square foot)**: Common in UAE real estate (1 sqm = 10.764 sqft)
- **sqm (square meter)**: International standard
- Conversion: Rate(AED/sqft) = Rate(AED/sqm) / 10.764

**Rate vs Price**
- **Rate**: Price per unit area (normalizes for size differences)
- **Price**: Total transaction price
- **Use Rate** when comparing properties of different sizes
- **Use Price** for total investment amounts

### Peer Comparison Methodology

**Who are "Peers"?**
- Peers = all other selected projects in your comparison
- Peer Average = median or mean of all peers (excludes the current project)

**Why Compare to Peers?**
- Isolates project-specific performance from market-wide trends
- Shows if a project is outperforming or lagging its competitive set
- Helps identify relative value opportunities

---

## Interpreting the Results

### Reading the Normalized Trend Chart

**Strong Upward Trend** (line slopes up sharply)
- Project experiencing price appreciation
- Demand may be increasing
- Could indicate successful development/amenities

**Flat Trend** (horizontal line around 1.0)
- Prices stable over the period
- Mature/established market
- Supply-demand equilibrium

**Downward Trend** (line slopes down)
- Price depreciation occurring
- May indicate oversupply or changing preferences
- Could present value-buying opportunity

### Reading the Delta Charts

**Positive Delta (above zero)**
- Project outperforming peers
- Stronger demand relative to competition
- May justify premium pricing

**Delta Near Zero**
- Project performing in line with market
- Tracking peer average closely

**Negative Delta (below zero)**
- Project underperforming peers
- Weaker relative demand
- May indicate value opportunity or underlying issues

### Key Performance Indicators

**Top Group % Growth**
- Best-performing project in your selection
- Positive % = appreciation, negative % = depreciation
- Compare against market expectations and inflation

**Spread Between Top and Weakest**
- Large spread (>20%): High dispersion, market differentiation
- Small spread (<10%): Homogeneous market performance

### Common Analysis Scenarios

**Scenario 1: New Development Analysis**
```
Question: How is a new project performing vs established projects?
Action: Select new project + 3-5 established peers
Look for:
- Is new project gaining market share? (rising delta)
- Is pricing premium justified? (compare rate levels)
```

**Scenario 2: Investment Timing**
```
Question: Is now a good time to buy in Community X?
Action: Select 5+ projects in Community X, view 3Y trend
Look for:
- Recent downtrend = potential entry point
- Consistent uptrend = momentum but less upside
- Check if trend is slowing (flattening line)
```

**Scenario 3: Portfolio Diversification**
```
Question: Which districts show different performance patterns?
Action: Compare projects across multiple districts
Look for:
- Low correlation between districts = good diversification
- One district outperforming = concentration opportunity
```

### Tips for Accurate Analysis

1. **Use Median for Robustness**
   - Median is less affected by luxury penthouses or bulk sales
   - Gives more typical market pricing

2. **Check Transaction Volume**
   - Low transactions = less reliable trends
   - Look at raw data to verify sufficient sample size

3. **Consider Time Frequency**
   - Monthly: Better for short-term trends, more volatile
   - Quarterly: Smooths seasonality, better for long-term view

4. **Combine Multiple Time Windows**
   - Check both 6M (recent momentum) and 3Y (long-term trend)
   - Divergence may indicate trend reversal

5. **Validate with Raw Data**
   - Use Raw Data page to verify transaction counts
   - Check for data gaps or anomalies
   - Confirm price ranges make sense

---

## Data Quality Notes

- **Update Frequency**: Data is refreshed biweekly
- **Rate Validation**: Automatically checks rate calculation accuracy
- **Outlier Detection**: Conservative k=3.0 threshold (removes only extreme outliers)
- **Missing Data**: Transactions with missing critical fields are excluded
- **Data Period**: Check Data Overview for exact date range in your dataset

---

## Need Help?

If you encounter any issues or have questions about interpreting your results:

1. Check the tooltips throughout the interface
2. Verify your data has sufficient transactions in your selected time window
3. Try adjusting time frequency or aggregation method
4. Use the Raw Data page to inspect underlying transactions

---

*Last Updated: 2025-11-27*
