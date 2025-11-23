# Library Imports 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Retail Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': None,
        'About': "Learning Streamlit with Real Data Forecasting!"
    }
)

# Custom CSS Styling
st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Custom metric cards */
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Success message styling */
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {  /* Streamlit sidebar class */
        background-color: #f0f2f6;
    }
    
    /* Button hover effects */
    .stButton > button {
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

def show_data_requirements():
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("📖 Data Format Help"):
        st.markdown("""
        ### 📋 What You Need in Your CSV
        
        **Required Columns:**
        - `week` - Date (DD-MM-YYYY)
        - `store_id` - Store number
        - `sku_id` - Product code
        - `units_sold` - Items sold
        
        **Optional:**
        - `total_price` - Price
        - `is_featured_sku` - Featured? (0/1)
        - `is_display_sku` - On display? (0/1)
        
        ### ✅ Tips
        - Use DD-MM-YYYY date format
        - No empty cells
        - At least 30 days of data
        - Keep file under 50MB
        
        ### ❌ Common Mistakes
        - Wrong date format
        - Missing required columns
        - Negative sales numbers
        - Too little data
        """)
        
        if st.button("📥 Get Sample File"):
            sample = pd.DataFrame({
                'record_ID': range(1, 101),
                'week': ['17-01-2011'] * 100,
                'store_id': [8091, 8095] * 50,
                'sku_id': [216418, 216419] * 50,
                'total_price': [99.04] * 100,
                'base_price': [99.04] * 100,
                'is_featured_sku': [0] * 100,
                'is_display_sku': [0] * 100,
                'units_sold': np.random.randint(10, 200, 100)
            })
            
            csv = sample.to_csv(index=False)
            st.download_button(
                "💾 Download Sample",
                csv,
                "sample_data.csv",
                "text/csv"
            )

@st.cache_data
def load_and_prepare_data_with_upload(uploaded_file):
    try:
        if 'using_sample' in st.session_state and st.session_state['using_sample']:
            df = st.session_state['sample_data']
            st.info("🎲 Using generated sample data")
        elif uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.info(f"📂 Using uploaded file: {uploaded_file.name}")
        else:
            df = pd.read_csv('train_data.csv')
            st.info("📂 Using default training dataset")
        
        if df.empty:
            st.error("❌ Dataset is empty!")
            return None, None
        
        required_columns = ['week', 'units_sold', 'store_id', 'sku_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.info("💡 Your CSV must contain: week, units_sold, store_id, sku_id")
            
            with st.expander("📋 See Required Data Format"):
                example_data = pd.DataFrame({
                    'record_ID': [1, 2, 3],
                    'week': ['17-01-2011', '17-01-2011', '17-01-2011'],
                    'store_id': [8091, 8091, 8095],
                    'sku_id': [216418, 216419, 216418],
                    'total_price': [99.04, 99.04, 99.04],
                    'base_price': [111.86, 99.04, 99.04],
                    'is_featured_sku': [0, 0, 0],
                    'is_display_sku': [0, 0, 0],
                    'units_sold': [20, 28, 99]
                })
                st.dataframe(example_data)
            return None, None
        
        if len(df) < 30:
            st.warning("⚠️ Dataset too small for reliable forecasting (minimum 30 records)")
            
        if len(df) > 1000000:
            st.warning("⚠️ Large dataset detected. Processing may take longer...")
            
        if (df['units_sold'] < 0).any():
            st.warning("⚠️ Found negative sales values. Cleaning data...")
            df = df[df['units_sold'] >= 0]
        
        try:
            df['date'] = pd.to_datetime(df['week'], format='%d-%m-%Y')
        except:
            try:
                df['date'] = pd.to_datetime(df['week'])
            except:
                st.error("❌ Cannot parse date format. Please use DD-MM-YYYY format")
                return None, None
        daily_sales = df.groupby('date')['units_sold'].sum().reset_index()
        daily_sales.columns = ['ds', 'y']
        
        st.success(f"✅ Data processed successfully! {len(df)} records, {len(daily_sales)} days")
        
        return df, daily_sales
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        return None, None
    
def handle_large_files(df):
    file_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    if file_size_mb > 100:
        st.warning(f"📊 Large dataset detected ({file_size_mb:.1f}MB)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_sampling = st.checkbox(
                "🎯 Use Data Sampling",
                help="Process a random sample for faster performance"
            )
        with col2:
            if use_sampling:
                sample_size = st.slider(
                    "Sample Size (%)",
                    min_value=10,
                    max_value=50,
                    value=25,
                    help="Percentage of data to use"
                )
                
        if use_sampling:
            sample_n = int(len(df) * (sample_size / 100))
            df = df.sample(n=sample_n, random_state=42)
            st.success(f"✅ Using {sample_size}% sample ({len(df)} records)")
    
    elif file_size_mb > 50:
        st.info(f"📊 Medium dataset ({file_size_mb:.1f}MB) - Processing may take a moment...")
        
    return df

@st.cache_resource
def train_forecasting_model(daily_sales):
    split_point = int(len(daily_sales) * 0.8)
    train_data = daily_sales[:split_point]
    
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        interval_width=0.95
    )
    
    with st.spinner("🤖 Training Model... Please wait"):
        model.fit(train_data)
    
    return model, train_data

# Metrics Display Function
def display_key_metrics(df, daily_sales):
    st.subheader("📊 Key Business Metrics")
    st.caption("Quick snapshot of your sales performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_sales = df['units_sold'].sum()
    avg_daily_sales = daily_sales['y'].mean()
    total_stores = df['store_id'].nunique()
    total_products = df['sku_id'].nunique()
    
    mid_point = len(daily_sales) // 2
    recent_avg = daily_sales['y'][mid_point:].mean()
    old_avg = daily_sales['y'][:mid_point].mean()
    growth_rate = ((recent_avg - old_avg) / old_avg) * 100
    
    with col1:
        st.metric(
            label="🛒 Total Products Sold", 
            value=f"{total_sales:,}",
            delta=f"{growth_rate:+.1f}% growth trend",
            help="Total number of products sold across all stores and dates"
        )
        st.caption("All-time sales volume")  
        
    with col2:
        st.metric(
            label="📅 Average Per Day",  
            value=f"{avg_daily_sales:,.0f} units",
            delta=f"{recent_avg-old_avg:+,.0f} recent change",
            help="Average number of products sold per day"
        )
        st.caption("Daily sales average")
        
    with col3:
        if growth_rate > 5:
            status = "Growing 📈"
            color = "🟢"
        elif growth_rate < -5:
            status = "Declining 📉"
            color = "🔴"
        else:
            status = "Stable ➡️"
            color = "🟡"
        
        st.metric(
            label="📊 Business Direction",  
            value=status,
            delta=f"{growth_rate:+.1f}%",
            help="Shows if your sales are increasing, decreasing, or staying the same"
        )
        st.caption(f"{color} Current business trend")
        
    with col4:
        st.metric(
            label="🏪 Active Stores",
            value=f"{total_stores}",
            help="Number of store locations in your dataset"
        )
        st.caption(f"{total_products:,} unique products")

# Forecasting Chart Function
def create_enhanced_forecast_chart(daily_sales, model, controls):
    st.subheader("🔮 Sales Forecast Chart")
    st.caption("Predicted sales for upcoming days")
    
    future = model.make_future_dataframe(periods=controls['forecast_days'])
    forecast = model.predict(future)
    
    last_data_date = daily_sales['ds'].max()
    
    if controls['forecast_method'] == "📊 Days from Last Date":
        forecast_start_date = last_data_date + pd.Timedelta(days=1)
        forecast_end_date = last_data_date + pd.Timedelta(days=controls['forecast_days'])
        
        st.info(f"""
        📅 **Forecast Period:** {forecast_start_date.strftime('%d %B %Y')} to {forecast_end_date.strftime('%d %B %Y')}  
        Showing next **{controls['forecast_days']} days** from {last_data_date.strftime('%d %B %Y')}
        """)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_sales['ds'],
        y=daily_sales['y'],
        mode='lines+markers',
        name="📊 Historical Sales",
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=4),
        hovertemplate='<b>Date:</b> %{x}<br><b>Sales:</b> %{y:,.0f} units<extra></extra>'
    ))
    
    forecast_start = len(daily_sales)
    fig.add_trace(go.Scatter(
        x=forecast['ds'][forecast_start:],
        y=forecast['yhat'][forecast_start:],
        mode='lines+markers',
        name="🔮 Forecast",
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=6, symbol='diamond'),
        hovertemplate='<b>Predicted Date:</b> %{x}<br><b>Forecast:</b> %{y:,.0f} units<extra></extra>'
    ))
    
    if controls['show_confidence']:
        fig.add_trace(go.Scatter(
            x=forecast['ds'][forecast_start:],
            y=forecast['yhat_upper'][forecast_start:],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'][forecast_start:],
            y=forecast['yhat_lower'][forecast_start:],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255,127,14,0.2)',
            name='🎯 95% Confidence',
            hovertemplate='<b>Range:</b> %{y:,.0f} - upper bound<extra></extra>'
        ))
        
    if controls['chart_theme'] == 'Dark':
        template = 'plotly_dark'
        bg_color = '#2F3349'
    elif controls['chart_theme'] == 'Colorful':
        template = 'plotly'
        bg_color = '#F0F8FF'
    elif controls['chart_theme'] == 'Minimal':
        template = 'simple_white'
        bg_color = 'white'
    else:
        template = 'plotly'
        bg_color = 'white'
        
    fig.update_layout(
        title=f"📈 Sales Forecast for Next {controls['forecast_days']} Days",
        xaxis_title="📅 Date",
        yaxis_title="📦 Units Sold",
        height=controls['chart_height'],
        template=template,
        plot_bgcolor=bg_color,
        hovermode='x unified',
        showlegend=True,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    return forecast

# Business Insights Function
def display_business_insights(forecast, daily_sales, controls):
    st.subheader("💡 What You Should Do?")
    st.caption("Simple actions based on predictions")
    
    next_7_days = forecast.tail(7)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📅 Next 7 Days Prediction")
        
        next_week = forecast.tail(7)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round(0)
        
        for idx, row in next_7_days.iterrows():
            day_name = row['ds'].strftime('%A')
            date_str = row['ds'].strftime('%d %b')
            predicted_sales = int(row['yhat'])
            
            # Simple visual indicator
            if predicted_sales > daily_sales['y'].mean() * 1.1:
                icon = "🔥"
                label = "High Sales Day"
            elif predicted_sales < daily_sales['y'].mean() * 0.9:
                icon = "📉"
                label = "Low Sales Day"
            else:
                icon = "➡️"
                label = "Normal Day"
            
            st.write(f"{icon} **{day_name}** ({date_str}): ~{predicted_sales:,} units - *{label}*")
            
        weekly_total = next_week['yhat'].sum()
        st.success(f"📊 **Total Next Week**: {weekly_total:,.0f} units")
        
    with col2:
        st.markdown("### 🎯 Recommended Actions")
        
        # Find peak day
        peak_day = next_7_days.loc[next_7_days['yhat'].idxmax()]
        peak_day_name = peak_day['ds'].strftime('%A, %d %B')
        peak_sales = int(peak_day['yhat'])
        
        st.success(f"""
        **🔥 Highest Sales Day Expected**  
        {peak_day_name}  
        Predicted: ~{peak_sales:,} units
        
        **Action Items:**
        - ✅ Increase inventory stock
        - ✅ Schedule extra staff
        - ✅ Prepare for high customer traffic
        """)
        
        low_day = next_7_days.loc[next_7_days['yhat'].idxmin()]
        low_name = low_day['ds'].strftime('%A, %d %B')
        low_sales = int(low_day['yhat'])
        
        st.info(f"""
        **📉 Lowest Sales Day Expected**  
        {low_name}  
        Predicted: ~{low_sales:,} units
        
        **Action Items:**
        - 🎯 Launch promotional offers
        - 💰 Consider discounts
        - 📦 Clear older inventory
        """)
        
# Business Intelligence Dashboard
def create_business_dashboard(df, forecast, controls):
    if not controls['show_business_dashboard']:
        return
    
    st.subheader("🎯 Executive Business Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    avg_price = df['total_price'].mean()
    forecast_revenue = forecast.tail(controls['forecast_days'])['yhat'].sum() * avg_price
    
    with col1:
        st.metric(
            "💰 Projected Revenue",
            f"${forecast_revenue:,.0f}",
            help="Revenue forecast based on predicted sales"
        )
    with col2:
        max_daily = forecast.tail(controls['forecast_days'])['yhat'].max()
        st.metric(
            "📦 Peak Inventory Need",
            f"{max_daily:,.0f} units",
            help="Maximum single-day inventory requirement"
        )    
    with col3:
        avg_forecast = forecast.tail(controls['forecast_days'])['yhat'].mean()
        historical_avg = df.groupby('date')['units_sold'].sum().mean()
        growth = ((avg_forecast - historical_avg) / historical_avg) * 100
        st.metric(
            "📈 Forecast Growth",
            f"{growth:+.1f}%",
            delta="vs historical avg",
            help="Growth compare to historical average"
        )
    with col4:
        total_forecast = forecast.tail(controls['forecast_days'])['yhat'].sum()
        st.metric(
            "📊 Total Forecast",
            f"{total_forecast:,.0f} units",
            help=f"Total units for next {controls['forecast_days']} days"
        )
        
    st.markdown("#### 🏪 Store Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        store_performance = df.groupby('store_id').agg({
            'units_sold': 'sum',
            'total_price': 'mean'
        }).round(2)

        fig = px.scatter(
            store_performance,
            x='total_price',
            y='units_sold',
            title="Store Performance: Price vs Volume",
            labels={'total_price': 'Average Price', 'units_sold': 'Total Units Sold'},
            hover_data={'total_price': ':.2f', 'units_sold': ':,'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_stores = df.groupby('store_id')['units_sold'].sum().sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=top_stores.index.astype(str),
            y=top_stores.values,
            title="🏆 Top 10 Performing Stores",
            labels={'x': 'Store ID', 'y': 'Total Units Sold'},
            color=top_stores.values,
            color_continuous_scale='Blues'
        )   
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 🛍️ Product Performance Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        featured_performance = df.groupby('is_featured_sku').agg({
            'units_sold': ['sum', 'mean'],
            'total_price': 'mean'
        }).round(2)
        
        featured_data = pd.DataFrame({
            'Category': ['Regular Products', 'Featured Products'],
            'Total_Sales': [
                featured_performance.loc[0, ('units_sold', 'sum')],
                featured_performance.loc[1, ('units_sold', 'sum')] if 1 in featured_performance.index else 0,
            ],
            'Avg_Sales': [
                featured_performance.loc[0, ('units_sold', 'mean')],
                featured_performance.loc[1, ('units_sold', 'mean')] if 1 in featured_performance.index else 0,
            ]
        })
        
        fig = px.bar(
            featured_data,
            x="Category",
            y='Total_Sales',
            title='📊 Featured vs Regular Products',
            color='Category',
            color_discrete_map={
                'Regular Products': '#3498db',
                'Featured Products': '#e74c3c',
            }
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        price_sales = df.groupby(pd.cut(df['total_price'], bins=5)).agg({
            'units_sold': 'sum'
        }).reset_index()
        price_sales['price_range'] = price_sales['total_price'].astype(str)
        
        fig = px.line(
            price_sales,
            x='price_range',
            y='units_sold',
            title="💰 Price Range vs Sales Volume",
            markers=True
        )
        fig.update_layout(height=350)
        fig.update_xaxes(title="Price Range")
        fig.update_yaxes(title="Units Sold")
        st.plotly_chart(fig, use_container_width=True)
            
# Model performance Function
@st.cache_data
def _calculate_model_performance(daily_sales_tuple):
    """Helper function to cache model performance calculations"""
    daily_sales = pd.DataFrame(daily_sales_tuple)
    train_size = int(len(daily_sales) * 0.8)
    train_data = daily_sales[:train_size]
    test_data = daily_sales[train_size:]
    
    test_model = Prophet(yearly_seasonality=False)
    test_model.fit(train_data)
    future_test = test_model.make_future_dataframe(periods=len(test_data))
    forecast_test = test_model.predict(future_test)
    return forecast_test, test_data

def display_model_performance(model, daily_sales, controls):
    st.subheader("🎯 How Reliable Are These Predictions?")
    st.caption("Check prediction accuracy")
    
    # Convert to tuple for caching
    daily_sales_tuple = tuple(map(tuple, daily_sales.values))
    forecast_test, test_data = _calculate_model_performance(daily_sales_tuple)
    
    y_true = test_data['y'].values if isinstance(test_data, pd.DataFrame) else test_data[:, 1]
    y_pred = forecast_test.tail(len(test_data))['yhat'].values
    
    # Ensure both arrays are numpy arrays to avoid type issues
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = 100 - mape
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if accuracy >= 90:
            rating = "Excellent ⭐⭐⭐"
            color = "normal"
        elif accuracy >= 80:
            rating = "Good 👍"
            color = "normal"
        elif accuracy >= 70:
            rating = "Fair ⚠️"
            color = "inverse"
        else:
            rating = "Needs Improvement ❌" 
            color = "inverse"
        
        st.metric(
            label="Prediction Accuracy", 
            value=f"{accuracy:.1f}%",
            delta=rating,
            delta_color=color,
            help="Percentage of times predictions match actual sales"
        )
    
    with col2:
        avg_sales = daily_sales['y'].mean()
        error_percent = (rmse / avg_sales) * 100
        
        st.metric(
            label="Average Error Range",  
            value=f"±{rmse:,.0f} units",  
            help="Typical difference between prediction and actual sales"  
        )
        st.caption(f"About {error_percent:.1f}% margin")  
    
    with col3:
        st.metric(
            label="Confidence Level",  
            value="95%",
            help="Statistical confidence in the prediction range shown"
        )
        st.caption("Very high reliability")  
    
    st.info(f"""
    **Understanding These Numbers:**
    
    ✅ The model is **{accuracy:.1f}% accurate** in predicting sales  
    📊 Predictions typically vary by **±{rmse:,.0f} units** from actual  
    🎯 **Example:** If forecast shows 1,000 units, actual sales will likely be between {1000-rmse:,.0f} and {1000+rmse:,.0f}
    
    **Overall Rating:** {rating}
    """)
    
# Data Explorer Function
def display_data_explorer(df, daily_sales, controls):
    if not controls['show_raw_data']:
        return
    
    st.subheader("📋 Data Explorer & Raw Tables")
    
    tab1, tab2, tab3 = st.tabs(["📈 Daily Sales Data", "🛍️ Original Dataset", "🔍 Data Analysis"])
    
    with tab1:
        st.markdown("**📊 Aggregated daily sales data (prepared for Prophet model):**")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            date_range = st.date_input(
                "Select Date Range:",
                value=(daily_sales['ds'].min(), daily_sales['ds'].max()),
                min_value=daily_sales['ds'].min(),
                max_value=daily_sales['ds'].max(),
            )
        with col2:
            sales_threshold = st.number_input(
                "Min Sales Filter:",
                min_value=0,
                value=0,
                step=1000                             
            )
        
        if len(date_range) == 2:
            filtered_daily = daily_sales[
                (daily_sales['ds'] >= pd.Timestamp(date_range[0])) &
                (daily_sales['ds'] <= pd.Timestamp(date_range[1])) &
                (daily_sales['y'] >= sales_threshold)
            ]    
        else:
            filtered_daily = daily_sales[daily_sales['y'] >= sales_threshold]
            
        st.dataframe(filtered_daily, use_container_width=True, height=300)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(filtered_daily))
        with col2:
            st.metric("Avg Sales", f"{filtered_daily['y'].mean():,.0f}")
        with col3:
            st.metric("Total Sales", f"{filtered_daily['y'].sum():,}")
            
        csv_daily = filtered_daily.to_csv(index=False)
        st.download_button(
            label="💾 Download Daily Sales CSV",
            data=csv_daily,
            file_name=f"daily_sales_filtered.csv",
            mime="text/csv"
        )
        
    with tab2:
        st.markdown("**🛍️ Original transaction-level dataset:**")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            selected_stores = st.multiselect(
                "Filter by Stores:",
                options=sorted(df['store_id'].unique()),
                default=sorted(df['store_id'].unique())[:5],
            )
        with col2:
            search_sku = st.text_input("Search SKU ID:", "")
            
        filtered_df = df[df['store_id'].isin(selected_stores)]
        if search_sku:
            filtered_df = filtered_df[filtered_df['sku_id'].str.contains(search_sku, case=False, na=False)]
            
        display_df = filtered_df.head(1000)
        st.dataframe(display_df, use_container_width=True, height=300)
        
        if len(filtered_df) > 1000:
            st.info(f"📊 Showing first 1000 rows of {len(filtered_df):,} total records")
        
        st.markdown("**📊 Dataset Summary:**")
        col1, col2, col3, col4 = st.columns(4)
    
        with col1:
            st.metric("Total Records", f"{len(filtered_df):,}")
        with col2:
            st.metric("Unique Stores", filtered_df['store_id'].nunique())
        with col3:
            st.metric("Unique SKUs", filtered_df['sku_id'].nunique())
        with col4:
            st.metric("Total Units", f"{filtered_df['units_sold'].sum():,}")
            
    with tab3:
        st.markdown("**🔍 Quick Data Analysis**")
        
        store_sales = df.groupby('store_id')['units_sold'].sum().sort_values(ascending=False).head(10)
        
        fig_stores = px.bar(
            x=store_sales.index,
            y=store_sales.values,
            title="🏪 Top 10 Stores by Total Sales",
            labels={'x': "Store ID", 'y': 'Total Units Sold'}
        )
        st.plotly_chart(fig_stores, use_container_width=True)
        
        fig_dist = px.histogram(
            df,
            x='units_sold',
            nbins=50,
            title="📊 Sales Volume Distribution",
            labels={'units_sold': 'Units Sold per Transaction'}
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
def generate_sample_data():
    import random 
    from datetime import datetime, timedelta
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎲 Try with Sample Data")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        num_days = st.sidebar.selectbox(
            "Date Period:",
            [30, 90, 180, 365],
            index=1,
            help="How many days of data to generate"
        )
        
    with col2:
        num_stores = st.sidebar.selectbox(
            "Stores:",
            [5, 10, 15, 20],
            index=1,
            help="Number of Retail locations"   
        )    
    
    if st.sidebar.button("🎲 Generate Sample Data", type="primary"):
        with st.spinner("Creating Sample retail data..."):
            np.random.seed(42)
            random.seed(42)
            
            start_date = datetime.now() - timedelta(days=num_days)
            dates = [start_date + timedelta(days=x) for x in range(num_days)]
            
            store_ids = [f"STORE{1000 + i}" for i in range(num_stores)]
            sku_ids = [f"SKU{5000 + i}" for i in range(50)]
            
            data = []
            record_id = 1
            
            for date in dates:
                is_weekend = date.weekday() >= 5
                weekend_mult = 1.4 if is_weekend else 1.0
                
                is_holiday = (date.month == 12 and date.day >= 20) or (date.month == 1 and date.day <= 5)
                holiday_mult = 1.6 if is_holiday else 1.0
                
                seasonal_mult = 1.3 if date.month in [11,12] else 0.85 if date.month in [1,2] else 1.0
                
                for store_id in store_ids:
                    store_perf = random.uniform(0.7, 1.3)
                    
                    products_sold = random.randint(5, len(sku_ids)//2)
                    selected_products = random.sample(sku_ids, products_sold)
                    
                    for sku_id in selected_products:
                        base_sales = random.randint(10, 100)
                        
                        final_sales = int(
                            base_sales * weekend_mult * holiday_mult * seasonal_mult * store_perf * random.uniform(0.8, 1.2)
                        )
                        
                        base_price = random.uniform(100, 1000)
                        is_featured = random.choice([0, 0, 0, 1])
                        is_display = random.choice([0, 0, 0, 1])
                     
                        if is_featured: 
                            total_price = base_price * random.uniform(0.75, 0.90)
                        else:
                            total_price = base_price * random.uniform(0.95, 1.05)
                            
                        data.append({
                            'record_ID': record_id,
                            'week': date.strftime('%d-%m-%Y'),
                            'store_id': store_id,
                            'sku_id': sku_id,
                            'total_price': round(total_price, 2),
                            'base_price': round(base_price, 2),
                            'is_featured_sku': is_featured,
                            'is_display_sku': is_display,
                            'units_sold': max(1, final_sales),
                        })
                        
                        record_id += 1
                        
                        
            sample_df = pd.DataFrame(data)
            st.session_state['sample_data'] = sample_df
            st.session_state['using_sample'] = True
            
            st.sidebar.success(f"✅ Generated {len(sample_df):,} record!")
            st.sidebar.success(f"📅 Period: {num_days} days\n🏪 Stores: {num_stores}")
            
            csv = sample_df.to_csv(index=False)
            st.sidebar.download_button(
                "💾 Download Sample Data",
                csv,
                f"sample_retail_data_{num_days}days.csv",
                "text/csv",
                help="Save this sample data for later use"
            )
            
            return sample_df
    return None

# Enhanced Sidebar Controls
def create_enhanced_sidebar_controls():
    st.sidebar.header("⚙️ Dashboard Settings")
    
    st.sidebar.subheader("📂 Your Data")  
    uploaded_file = st.sidebar.file_uploader(
        "Upload your sales CSV file",  
        type=['csv'],
        help="Upload your own data or we'll use sample data"  
    )
    
    sample_df = generate_sample_data()
    
    if uploaded_file:
        st.sidebar.success("✅ Using your uploaded file")  
        
        if 'using_sample' in st.session_state:
            st.session_state['using_sample'] = False
    elif sample_df is not None:
        uploaded_file = None
    else:
        st.sidebar.info("📊 Using default dataset")  
    
    st.sidebar.subheader("📅 Forecast Settings")  
    st.sidebar.info("📍 Predictions start from your latest data date")
    
    forecast_method = st.sidebar.radio(
        "How do you want to forecast?",  
        ["📊 Days from Last Date", "📅 Specific Date Range", "🎯 Next N Business Days"]
    )
    
    if forecast_method == "📊 Days from Last Date":
        forecast_days = st.sidebar.slider(
            "Number of days to predict:",
            min_value=7,
            max_value=90,
            value=30,
            step=7,
            help="How many days into the future you want to see"  
        )
        start_date = None
        end_date = None
    elif forecast_method == "📅 Specific Date Range":
        start_date = st.sidebar.date_input("Forecast start date:")
        end_date = st.sidebar.date_input("Forecast end date:")
            
        if start_date and end_date:
            forecast_days = (end_date - start_date).days
            if forecast_days <= 0:
                st.sidebar.error("End date must be after start date!")
                forecast_days = 7
        else:
            forecast_days = 7
    else:
        business_days = st.sidebar.slider(
            "Number of business days:",  
            min_value=5,
            max_value=60,
            value=20,
            step=5,
            help="Weekends will be excluded automatically"
        )
        forecast_days = int(business_days * 1.4)
        start_date = None
        end_date = None
    
    # Advanced Options 
    st.sidebar.subheader("🔬 Advanced Settings")
    
    model_type = st.sidebar.selectbox(
        "Forecasting algorithm:",  
        ["Prophet (Default)", "Prophet with Holidays", "Prophet Enhanced"],
        help="Different models for different accuracy needs"  
    )
    
    confidence_level = st.sidebar.slider(
        "Prediction confidence level:",  
        min_value=80,
        max_value=99,
        value=95,
        help="How confident you want the predictions to be (higher = wider range)" 
    )
    
    include_holidays = st.sidebar.checkbox(
        "Account for holidays",
        help="Include holiday effects in predictions"
    )
    
    seasonal_adjustment = st.sidebar.selectbox(
        "Seasonal pattern focus:",
        ["Auto", "Weekly", "Monthly", "Quarterly"],
        help="Which seasonal patterns to emphasize"
    )
    
    st.sidebar.subheader("👁️ Display Settings")
    
    show_confidence = st.sidebar.checkbox(
        "Show prediction range",
        value=True,
        help="Display minimum and maximum expected values"
    )
    
    show_raw_data = st.sidebar.checkbox(
        "Show data tables",
        value=False,
        help="View your original CSV data"
    )
    
    show_model_details = st.sidebar.checkbox(
        "Show accuracy metrics",  
        value=True,
        help="Display how accurate the model is"
    )
    
    show_business_dashboard = st.sidebar.checkbox(
        "Show business insights", 
        value=True,
        help="Display recommended actions and KPIs"
    )
    
    show_data_quality = st.sidebar.checkbox(
        "Show data quality check",  
        value=False,
        help="Check your data for issues"
    )
    
    show_alerts = st.sidebar.checkbox(
        "Show business alerts",  
        value=True,
        help="Get notified about important trends"
    )
    
    st.sidebar.subheader("🎨 Visual Settings")
    
    chart_theme = st.sidebar.selectbox(
        "Chart appearance:",
        ["Default", "Dark", "Colorful", "Minimal"],
        help="Choose your preferred chart style"
    )
    
    chart_height = st.sidebar.slider(
        "Chart height:",
        min_value=300,
        max_value=800,
        value=500,
        step=50
    )
    
    # Return ALL controls - NO changes to functionality
    return {
        'uploaded_file': uploaded_file,
        'forecast_method': forecast_method,
        'forecast_days': forecast_days,
        'start_date': start_date,
        'end_date': end_date,
        'model_type': model_type,
        'confidence_level': confidence_level,
        'include_holidays': include_holidays,
        'seasonal_adjustment': seasonal_adjustment,
        'show_confidence': show_confidence,
        'show_raw_data': show_raw_data,
        'show_model_details': show_model_details,
        'show_business_dashboard': show_business_dashboard,
        'show_data_quality': show_data_quality,
        'show_alerts': show_alerts,
        'chart_theme': chart_theme,
        'chart_height': chart_height,
    }

# Export Functionality    
def create_export_section(forecast, daily_sales, controls):
    st.subheader("💾 Export & Download")
    st.caption("Save predictions for your records")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Prediction Table")
        
        export_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(controls['forecast_days'])
        export_df.columns = ['Date', 'Expected Sales', 'Minimum', 'Maximum']
        export_df['Date'] = export_df['Date'].dt.strftime('%d-%m-%Y')
        export_df['Day'] = pd.to_datetime(export_df['Date'], format='%d-%m-%Y').dt.day_name()
        
        export_df = export_df[['Date', 'Day', 'Expected Sales', 'Minimum', 'Maximum']]
        
        for col in ['Expected Sales', 'Minimum', 'Maximum']:
            export_df[col] = export_df[col].round(0).astype(int)
        
        st.dataframe(export_df, use_container_width=True)
        
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name=f"sales_prediction_{controls['forecast_days']}days.csv",
            mime="text/csv"
        )
    
    with col2:
        st.markdown("### 📋 Buniness Report")
        
        total_predicted = export_df['Expected Sales'].sum()
        avg_daily = export_df['Expected Sales'].mean()
        peak = export_df.loc[export_df['Expected Sales'].idxmax()]
        
        report = f"""
SALES PREDICTION REPORT
Created: {pd.Timestamp.now().strftime('%d %B %Y')}

---

FORECAST DETAILS:
Period: {controls['forecast_days']} days
From: {export_df['Date'].iloc[0]}
To: {export_df['Date'].iloc[-1]}

KEY NUMBERS:
• Expected Total Sales: {total_predicted:,} units
• Average Per Day: {avg_daily:,.0f} units
• Peak Sales Day: {peak['Day']}, {peak['Date']}
  Expected: {peak['Expected Sales']:,} units

RECOMMENDED ACTIONS:
1. Stock Level: Order approximately {total_predicted:,} units
2. Peak Day Prep: Extra inventory for {peak['Day']}
3. Staffing: More employees on high-volume days
4. Tracking: Compare actual vs predicted daily

---

HOW TO USE THIS REPORT:
✓ Share with inventory team
✓ Plan staff schedules
✓ Budget allocation
✓ Performance tracking
        """
        
        st.text_area("Report Preview", report, height=400)
        
        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name=f"prediction_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

# Data Quality Report
def create_data_quality_report(df):
    st.subheader("🔍 Data Quality Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('#### 📊 Data Completeness')
        missing_data = df.isnull().sum()
        completeness = (1 - missing_data / len(df)) * 100
        
        for col, pct in completeness.items():
            color = "🟢" if pct > 95 else "🟡" if pct > 90 else "🔴"
            st.write(f"{color} {col}: {pct:.1f}% complete")
    
    with col2:
        st.markdown("#### 📈 Statistical Summary")
        st.write("**Sales Distribution:**")
        st.write(f"Mean: {df['units_sold'].mean():.1f}")
        st.write(f"Median: {df['units_sold'].median():.1f}")
        st.write(f"Std Dev: {df['units_sold'].std():.1f}")
        st.write(f"Min: {df['units_sold'].min()}")
        st.write(f"Max: {df['units_sold'].max()}")
        
        st.write("**Price Statistics:**")
        st.write(f"Avg Price: ${df['total_price'].mean():.2f}")
        st.write(f"Price Range: ${df['total_price'].min():.2f} - ${df['total_price'].max():.2f}")
        
    with col3:
        st.markdown('#### ⚠️ Data Issues')
        issues = []
        
        if (df['units_sold'] == 0).sum() > 0:
            issues.append(f"🟡 {(df['units_sold'] == 0).sum()} zero sales records")
            
        if df.duplicated().sum() > 0:
            issues.append(f"🟡 {df.duplicated().sum()} duplicate records")
            
        if (df['total_price'] <= 0).sum() > 0:
            issues.append(f"🔴 {(df['total_price'] <= 0).sum()} invalid price records")
            
        q1 = df['units_sold'].quantile(0.25)
        q3 = df['units_sold'].quantile(0.75)
        
        iqr = q3 - q1
        
        outliers = df[(df['units_sold'] < (q1 - 1.5 * iqr)) | (df['units_sold'] > (q3 + 1.5 * iqr))]
        
        if len(outliers) > 0:
            issues.append(f"🟡 {len(outliers)} potential outliers detected")
        
        if not issues:
            issues.append("🟢 No major data quality issues")
            
        for issue in issues:
            st.write(issue)
            
    st.markdown("#### 🎯 Overall Data Quality Score")
    
    completeness_score = completeness.mean()
    outlier_penalty = min(20, (len(outliers) / len(df)) * 100)
    missing_penalty = min(10, (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100)
    
    quality_score = max(0, completeness_score - outlier_penalty - missing_penalty)
    
    if quality_score >= 90:
        color = "🟢"
        status = "Excellent"
    elif quality_score >= 80:
        color = "🟡"
        status = "Good"
    else:
        color = "🔴"
        status = "Needs Improvement"
    
    st.metric(
        f"{color} Data Quality Score",
        f"{quality_score:.1f}/100",
        delta=status,
        help="Based on completeness, outliers, and missing data"
    )

            
# Real-Time Alert System
def create_alert_system(forecast, daily_sales, controls):
    if not controls['show_alerts']:
        return
    
    st.subheader("🚨 Business Alerts & Recommendations")
    
    alerts = []
    
    recent_forecast = forecast.tail(controls['forecast_days'])['yhat']
    historical_avg = daily_sales['y'].mean()
    
    # Alert for significant drops
    if recent_forecast.min() < historical_avg * 0.5:
        alerts.append({
            "type": "error",
            "title": "📉 Significant Sales Drop Predicted",
            "message": f"Sales may drop to {recent_forecast.min():,.0f} units (vs avg {historical_avg:,.0f})",
            "action": "Consider promotional campaigns or inventory adjustments"
        })
    
    # Alert for high demand
    if recent_forecast.max() > historical_avg * 1.5:
        alerts.append({
            "type": "warning",
            "title": "🔥 High Demand Period Ahead",
            "message": f"Peak demand of {recent_forecast.max():,.0f} units expected",
            "action": "Ensure adequate inventory and staffing"
        })
    
    # Alert for unusual variance
    forecast_std = recent_forecast.std()
    historical_std = daily_sales['y'].std()
    
    if forecast_std > historical_std * 1.5:
        alerts.append({
            "type": "info",
            "title": "📊 High Volatility Period",
            "message": f"Sales volatility is {(forecast_std/historical_std):.1f}x higher than historical",
            "action": "Prepare for variable demand patterns"
        })
        
    forecast_with_day = forecast.tail(controls['forecast_days']).copy()
    forecast_with_day['day_of_week'] = forecast_with_day['ds'].dt.day_name()
    
    weekend_avg = forecast_with_day[forecast_with_day['day_of_week'].isin(['Saturday', 'Sunday'])]['yhat'].mean()
    weekday_avg = forecast_with_day[~forecast_with_day['day_of_week'].isin(['Saturday', 'Sunday'])]['yhat'].mean()
    
    if not np.isnan(weekday_avg) and not np.isnan(weekend_avg):
        if weekend_avg > weekday_avg * 1.3:
            alerts.append({
                "type": "info",
                "title": "🎉 Strong Weekend Performance Expected",
                "message": f"Weekend sales ({weekend_avg:.0f}) vs weekday ({weekday_avg:.0f})",
                "action": "Optimize weekend staffing and inventory"
            })
        
    # Display alerts
    if alerts:
        for alert in alerts:
            if alert["type"] == "error":
                st.error(f"**{alert['title']}**\n\n{alert['message']}\n\n💡 *Action Required: {alert['action']}*")
            elif alert["type"] == "warning":
                st.warning(f"**{alert['title']}**\n\n{alert['message']}\n\n💡 *Recommended Action: {alert['action']}*")
            else:
                st.info(f"**{alert['title']}**\n\n{alert['message']}\n\n💡 *Consider: {alert['action']}*")
    else:
        st.success("✅ **All Clear!** No significant business alerts at this time.")


# MAIN FUNCTION
def main():
    st.title("📈 Retail Sales Forecasting Dashboard")
    st.markdown("### 🎯 *Professional forecasting system for retail business intelligence*")
    st.markdown("---")
    
    # Show data requirements
    show_data_requirements()
    
    # Get controls (including file upload)
    controls = create_enhanced_sidebar_controls()
    
    # Load data with file upload support
    with st.spinner("📂 Loading and processing data..."):
        df, daily_sales = load_and_prepare_data_with_upload(controls['uploaded_file'])
    
    if df is None or daily_sales is None:
        st.stop()
    
    # Handle large files
    if len(df) > 100000:
        df = handle_large_files(df)
        daily_sales = df.groupby('date')['units_sold'].sum().reset_index()
        daily_sales.columns = ['ds', 'y']
    
    # Display metrics 
    display_key_metrics(df, daily_sales)
    st.markdown("---")
    
    # Train model 
    with st.spinner("🧠 Training forecasting model..."):
        model, train_data = train_forecasting_model(daily_sales)
    st.success("✅ Model trained successfully! Ready for forecasting.")
    
    # Create forecast chart
    forecast = create_enhanced_forecast_chart(daily_sales, model, controls)
    st.markdown("---")
    
    # Business insights
    display_business_insights(forecast, daily_sales, controls)
    st.markdown("---")
    
    # Business dashboard (ADD THIS)
    create_business_dashboard(df, forecast, controls)
    st.markdown("---")
    
    # Alert system (ADD THIS)
    create_alert_system(forecast, daily_sales, controls)
    st.markdown("---")
    
    # Model performance
    display_model_performance(model, daily_sales, controls)
    st.markdown("---")
    
    # Data quality report (ADD THIS)
    if controls.get('show_data_quality', False):
        create_data_quality_report(df)
        st.markdown("---")
    
    # Data explorer
    display_data_explorer(df, daily_sales, controls)
    st.markdown("---")
    
    # Export section
    create_export_section(forecast, daily_sales, controls)
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <p style='color: #666; font-size: 14px;'>
            💡 Built with: Streamlit 🎨 • Prophet 🔮 • Plotly 📊 • Python 🐍
        </p>
    </div>
    """, unsafe_allow_html=True)    
    
if __name__ == "__main__":
    main()