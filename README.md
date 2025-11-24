# 📈 Retail Vision: AI-Powered Sales Forecasting Platform

<div align="center">
  <img src="./application-view.png" alt="Retail Vision Dashboard" width="850"/>
  <p><i>Interactive forecasting dashboard with real-time AI predictions</i></p>
</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Prophet](https://img.shields.io/badge/Facebook-Prophet-0081FB?style=for-the-badge&logo=meta&logoColor=white)](https://facebook.github.io/prophet/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

**Enterprise-grade retail sales forecasting powered by Facebook Prophet**

[🚀 Live Demo](#-deployment) • [📖 Documentation](#-table-of-contents) • [💡 Features](#-key-features) • [🎯 Quick Start](#-quick-start)

</div>

---

## 📑 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [📊 Usage Guide](#-usage-guide)
- [🔬 Technical Details](#-technical-details)
- [🌐 Deployment](#-deployment)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🎯 Overview

**Retail Vision** is a production-ready sales forecasting platform that transforms retail data into predictive insights using Facebook Prophet. Built for data analysts, business intelligence teams, and retail managers, it provides an intuitive interface for time series forecasting with comprehensive analytics.

### 💼 Business Impact

<table>
<tr>
<td width="50%">

**📦 Inventory Optimization**
- Predictive inventory planning
- Demand pattern recognition
- Automated stock recommendations

**💰 Revenue Insights**
- Multi-period forecasting (7-365 days)
- Confidence interval predictions
- Revenue projection capabilities

</td>
<td width="50%">

**⚡ Operational Excellence**
- Interactive data visualization
- Real-time forecast updates
- Multi-store analytics support

**🎯 Strategic Planning**
- Peak demand identification
- Seasonal trend analysis
- Actionable business insights

</td>
</tr>
</table>

### 🎪 Use Cases

| Industry Segment | Application | Features |
|------------------|-------------|----------|
| 🛒 **Retail Chains** | Multi-store demand forecasting | Store performance analytics |
| 🛍️ **E-commerce** | SKU-level sales prediction | Product insights dashboard |
| 👗 **Fashion & Apparel** | Seasonal trend forecasting | Featured product analysis |
| 🍔 **Food & Beverage** | Perishable inventory planning | Daily/weekly breakdowns |
| 🏪 **Convenience Stores** | Daily sales optimization | Peak day detection |

---

## ✨ Key Features

### 🚀 Core Capabilities

<table>
<tr>
<td width="33%" align="center">

### 🔮 Advanced Forecasting
- Facebook Prophet algorithm
- 7-365 day forecast horizon
- Multiple model variants
- Automated seasonality detection
- Holiday effects modeling
- Confidence intervals (80-99%)

</td>
<td width="33%" align="center">

### 📊 Interactive Dashboard
- Real-time data visualization
- Custom date range selection
- Multi-store comparisons
- Export to CSV/TXT
- Responsive design
- Multiple chart themes

</td>
<td width="33%" align="center">

### 💡 Business Intelligence
- Automated insights generation
- Peak demand identification
- Inventory recommendations
- Business alert system
- Data quality reporting
- Executive KPI dashboard

</td>
</tr>
</table>

### 🎨 Dashboard Components

```
📈 RETAIL VISION Dashboard
├─ 📊 Key Metrics
│  ├─ Total Products Sold
│  ├─ Average Daily Sales
│  ├─ Business Direction
│  └─ Active Stores
│
├─ 🔮 Forecast Chart
│  ├─ Historical Sales Line
│  ├─ Prediction Line
│  └─ 95% Confidence Bands
│
├─ 💡 Business Insights
│  ├─ Daily Breakdown (7 days)
│  ├─ Weekly Performance (4+ weeks)
│  ├─ Monthly Forecast (2+ months)
│  └─ Quarterly Analysis (6+ months)
│
├─ 🎯 Executive Dashboard
│  ├─ Revenue Projections
│  ├─ Peak Inventory Needs
│  ├─ Growth Metrics
│  └─ Store Performance
│
├─ 🚨 Business Alerts
│  ├─ Sales Drop Warnings
│  ├─ High Demand Alerts
│  ├─ Volatility Notifications
│  └─ Weekend Performance
│
└─ 📋 Data Explorer
   ├─ Daily Sales Data
   ├─ Original Dataset
   └─ Quick Analysis
```

---

## 🏗️ Architecture

### 📐 System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Streamlit  │  │    Plotly    │  │   Controls   │     │
│  │   Dashboard  │  │  Charts      │  │   & Widgets  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                   BUSINESS LOGIC LAYER                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Prophet    │  │  Forecasting │  │   Insights   │     │
│  │   Model      │  │  Engine      │  │   Generator  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  CSV Upload  │  │  Data        │  │  Validation  │     │
│  │  & Sample    │  │  Processing  │  │  & Caching   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 🧩 Component Overview

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit 1.28+ | Interactive web interface |
| **ML Engine** | Facebook Prophet | Time series forecasting |
| **Visualization** | Plotly 5.14+ | Dynamic charts & graphs |
| **Data Processing** | Pandas 2.0+ | Data transformation |
| **Metrics** | Scikit-learn | Model evaluation (RMSE, MAE, MAPE) |
| **Caching** | Streamlit Cache | Performance optimization |

---

## 🚀 Quick Start

### 📋 Prerequisites

```bash
# System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 500MB free disk space
- Modern web browser (Chrome, Firefox, Edge)
```

### 🔧 Installation

#### **Option 1: Quick Setup (Recommended)**

```bash
# Clone repository
git clone https://github.com/iamrohitkandpal/RetailVision.git
cd "RetailVision"

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run interactive_dashboard.py
```

#### **Option 2: Virtual Environment**

```bash
# Create virtual environment
python -m venv retail_env

# Activate environment
# Windows:
retail_env\Scripts\activate
# Mac/Linux:
source retail_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run interactive_dashboard.py
```

#### **Option 3: Conda Environment (Recommended for Prophet)**

```bash
# Create conda environment
conda create -n retail-forecasting python=3.9

# Activate environment
conda activate retail-forecasting

# Install Prophet (conda recommended)
conda install -c conda-forge prophet

# Install other dependencies
pip install streamlit plotly pandas scikit-learn

# Launch dashboard
streamlit run interactive_dashboard.py
```

### 🎯 First Run

1. **Dashboard opens automatically** at `http://localhost:8501`
2. **Upload your CSV** or generate sample data
3. **Configure forecast settings** in the sidebar
4. **Generate predictions** and explore insights!

---

## 📊 Usage Guide

### 📂 Data Requirements

#### **Required CSV Format**

```csv
week,store_id,sku_id,units_sold,total_price
01-01-2024,STORE_001,SKU_12345,150,4500.00
01-01-2024,STORE_002,SKU_12345,200,6000.00
02-01-2024,STORE_001,SKU_12345,175,5250.00
```

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `week` | Date | Sales date (DD-MM-YYYY) | `15-03-2024` |
| `store_id` | String | Unique store identifier | `STORE_001` |
| `sku_id` | String | Product SKU code | `SKU_12345` |
| `units_sold` | Integer | Quantity sold | `150` |
| `total_price` | Float | Total revenue (optional) | `4500.00` |

#### **Data Quality Guidelines**

✅ **Best Practices:**
- Minimum **30 days** of historical data
- Consistent date format (DD-MM-YYYY)
- No missing values in required columns
- Clean data without extreme outliers

❌ **Common Issues:**
- Inconsistent date formats
- Negative sales values
- Missing store or SKU identifiers
- Large gaps in time series

### 🎮 Dashboard Controls

#### **Sidebar Settings**

| Setting | Options | Description |
|---------|---------|-------------|
| **📂 Data Upload** | CSV file | Upload your sales data |
| **🎲 Sample Data** | Generate | Create test data (30-365 days) |
| **📅 Forecast Method** | Days/Range/Business Days | Prediction period |
| **🔬 Model Type** | Default/Holidays/Enhanced | Algorithm variant |
| **👁️ Display Options** | Toggles | Show/hide components |
| **🎨 Visual Settings** | Themes/Height | Chart appearance |

#### **Advanced Features**

```python
# Forecast Configuration
├─ 📊 Days from Last Date (7-365 days)
├─ 📅 Specific Date Range
└─ 🎯 Next N Business Days

# Model Options
├─ Prophet (Default)
├─ Prophet with Holidays
└─ Prophet Enhanced (flexible trends)

# Seasonality Settings
├─ Auto Detection
├─ Weekly Patterns
├─ Monthly Patterns
└─ Quarterly Patterns

# Display Features
├─ Confidence Bands (80-99%)
├─ Data Quality Report
├─ Business Alerts
├─ Model Performance Metrics
└─ Raw Data Tables
```

### 📈 Interpreting Results

#### **Forecast Chart Components**

- **📊 Historical Sales (Blue Line)**: Actual sales data from your dataset
- **🔮 Forecast Line (Orange Dashed)**: Predicted sales for future dates
- **🎯 Confidence Band (Shaded Area)**: 95% confidence interval (customizable to 80-99%)

#### **Business Insights Tabs**

1. **📅 Daily Insights** (Always shown)
   - Next 7 days detailed breakdown
   - Peak and low sales days
   - Recommended actions

2. **📊 Weekly Performance** (28+ days forecast)
   - Week-by-week breakdown
   - Growth trends
   - Strategic recommendations

3. **📅 Monthly Forecast** (60+ days forecast)
   - Monthly sales projections
   - Confidence ranges
   - Recovery strategies

4. **🗓️ Quarterly Analysis** (180+ days forecast)
   - Quarterly performance
   - YoY analysis
   - Long-term trends

---

## 🔬 Technical Details

### 🧠 Prophet Algorithm

Facebook Prophet is designed for business time series forecasting with strong seasonal effects and multiple seasons of historical data.

#### **Model Components**

```python
y(t) = g(t) + s(t) + h(t) + ε(t)

Where:
g(t) = Trend component (piecewise linear or logistic growth)
s(t) = Seasonality (weekly, monthly, yearly patterns)
h(t) = Holiday effects (optional)
ε(t) = Error term (random variation)
```

#### **Implementation Features**

| Feature | Implementation | Benefit |
|---------|---------------|---------|
| **Automatic Seasonality** | Fourier series | Detects weekly/yearly patterns |
| **Trend Changepoints** | Automatic detection | Adapts to growth shifts |
| **Holiday Effects** | Country-specific | Accounts for special events |
| **Missing Data Handling** | Built-in imputation | Handles gaps gracefully |
| **Confidence Intervals** | Configurable (80-99%) | Quantifies uncertainty |

### 📊 Model Configuration

```python
# Available Model Types

# 1. Prophet (Default)
model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False,  # Optimized for cloud
    interval_width=0.95,
    changepoint_prior_scale=0.05
)

# 2. Prophet with Holidays
model = Prophet(...)
model.add_country_holidays(country_name='IN')  # or 'US', 'UK', 'AU'

# 3. Prophet Enhanced
model = Prophet(
    changepoint_prior_scale=0.1,  # More flexible trends
    seasonality_prior_scale=15.0
)
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
```

### 🎯 Data Processing Pipeline

```
Raw CSV Upload
     ↓
Data Validation (columns, dates, types)
     ↓
Date Parsing (DD-MM-YYYY format)
     ↓
Data Cleaning (remove negatives, outliers)
     ↓
Daily Aggregation (group by date)
     ↓
Prophet Format (ds, y columns)
     ↓
Model Training (cached)
     ↓
Forecast Generation
     ↓
Insights & Visualization
```

### ⚡ Performance Optimizations

| Optimization | Implementation | Benefit |
|-------------|---------------|---------|
| **@st.cache_data** | Data loading function | 50-70% faster reruns |
| **@st.cache_resource** | Model training function | Reuses trained model |
| **Data Sampling** | Large file handling (>50MB) | Processes big datasets |
| **Disabled yearly seasonality** | Cloud optimization | Faster training |
| **Streamlit config** | Custom settings | Optimized resources |

---

## 🌐 Deployment

### ☁️ Streamlit Cloud (Recommended)

Quick deployment for public access.

#### **Deployment Steps**

```bash
# 1. Prepare repository
git add .
git commit -m "Optimize for Streamlit Cloud deployment"
git push origin main

# 2. Deploy on Streamlit Cloud
# Visit: https://share.streamlit.io
# - Click "New app"
# - Select repository
# - Set main file: interactive_dashboard.py
# - Click "Deploy"
```

**Estimated deployment time**: 2-3 minutes

See [STREAMLIT_CLOUD_DEPLOYMENT.md](STREAMLIT_CLOUD_DEPLOYMENT.md) for detailed instructions.

### 🐳 Docker Deployment

```bash
# Build image
docker build -t retail-vision:latest .

# Run container
docker run -p 8501:8501 retail-vision:latest

# Or use Docker Compose
docker-compose up -d
```

### ☁️ Cloud Platforms

- **AWS Elastic Beanstalk**: See deployment guide in docs
- **Azure App Service**: Supported
- **Google Cloud Run**: Compatible

---

## 📦 Project Structure

```
Retail Vision/
│
├── 📊 Data Files
│   ├── train_data.csv              # Sample dataset
│   ├── adult_retail_data.csv       # Generated data
│   ├── teen_retail_data.csv        # Generated data
│   ├── child_retail_data.csv       # Generated data
│   └── daddy_retail_data.csv       # Generated data
│
├── 🎯 Main Application
│   ├── interactive_dashboard.py    # Main dashboard (1300+ lines)
│   ├── main.ipynb                  # Analysis notebook
│   └── generate_retail_data.py     # Data generator
│
├── 📋 Configuration
│   ├── requirements.txt            # Python dependencies
│   ├── .streamlit/config.toml      # Streamlit settings
│   ├── .gitignore                  # Git ignore patterns
│   ├── .dockerignore               # Docker ignore patterns
│   └── Dockerfile                  # Container config
│
└── 📖 Documentation
    ├── README.md                   # This file
    ├── STREAMLIT_CLOUD_DEPLOYMENT.md
    └── CLOUD_DEPLOYMENT_CHECKLIST.md
```

---

## 🛠️ Development

### 🔧 Setting Up

```bash
# Clone and setup
git clone <repository-url>
cd "Data Forecasting for Retail"
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run application
streamlit run interactive_dashboard.py
```

### 🧪 Testing

The application includes built-in data quality checks and model performance metrics. Generate sample data to test all features.

---

## 🆘 Troubleshooting

### ❌ Common Issues

**Prophet Installation Fails**
```bash
# Use conda (recommended)
conda install -c conda-forge prophet
```

**Dashboard Won't Start**
```bash
# Clear cache and restart
streamlit cache clear
streamlit run interactive_dashboard.py
```

**Large File Issues**
- Enable "Data Sampling" option in sidebar
- Process 25-50% of data for faster performance

**Memory Errors**
- Use data sampling for files >50MB
- Reduce forecast period
- Close other applications

---

## 📚 Resources

- **[Facebook Prophet Docs](https://facebook.github.io/prophet/)** - Official documentation
- **[Streamlit Docs](https://docs.streamlit.io/)** - Framework guide
- **[Plotly Python](https://plotly.com/python/)** - Visualization reference

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- 🌍 **Internationalization** - Multiple languages
- 📱 **Mobile Optimization** - Enhanced responsive design
- 🔌 **Data Connectors** - Database integrations
- 🤖 **ML Models** - Additional forecasting algorithms
- 📊 **Visualizations** - New chart types
- 🧪 **Tests** - Automated testing suite

**Process:**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **[Facebook Prophet Team](https://github.com/facebook/prophet)** - Forecasting algorithm
- **[Streamlit Team](https://streamlit.io/)** - Web framework
- **[Plotly Team](https://plotly.com/)** - Visualization library

---

## 📞 Contact

**Rohit Kandpal**
- 🌐 GitHub: [@iamrohitkandpal](https://github.com/iamrohitkandpal)
- 💼 Project: [Something-About-Data](https://github.com/iamrohitkandpal/Something-About-Data)

---

<div align="center">

**Built with ❤️ using Streamlit, Prophet, and Python**

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Powered by Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-red?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Uses Prophet](https://img.shields.io/badge/Uses-Prophet-0081FB?style=for-the-badge&logo=meta)](https://facebook.github.io/prophet/)

</div>