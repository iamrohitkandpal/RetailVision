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
    page_title="RetailVision",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': None,
        'About': "RetailVision - Data-Powered Retail Sales Forecasting"
    }
)

# Custom CSS Styling
st.markdown("""
<style>
    /* Modern card design */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #F0F4F8 !important;
        background: none !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
        letter-spacing: 0.5px;
    }
    
    /* Enhanced button styling - better visibility */
    .stButton > button {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 8px;
        background-color: #667eea !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: 2px solid #667eea !important;
        padding: 10px 20px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4) !important;
        background-color: #5568d3 !important;
        border-color: #5568d3 !important;
    }
    
    /* Primary button (type=primary) */
    .stButton > button[kind="primary"] {
        background-color: #667eea !important;
        color: #ffffff !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #5568d3 !important;
    }
    
    /* Modern tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        color: #333333 !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #667eea !important;
        color: #ffffff !important;
    }
    
    /* Icon styling - ensure visibility */
    .stIcon {
        color: #333333 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f0f2f6 !important;
        color: #333333 !important;
        font-weight: 600 !important;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background-color: #27ae60 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .stDownloadButton > button:hover {
        background-color: #229954 !important;
    }
    
    /* Hide scrollbar but keep functionality */
    .main::-webkit-scrollbar {
        width: 8px;
    }
    
    .main::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    .main::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
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
        st.info("""
💡      **Troubleshooting:**
        - Check file format (must be CSV)
        - Verify column names match requirements
        - Ensure no special characters in data
        """)
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

# @st.cache_resource(ttl=3600)
def train_forecasting_model(daily_sales, model_type='Prophet (Default)', confidence_level=95, include_holidays=False, seasonal_adjustment='Auto', holiday_country='IN'):
    split_point = int(len(daily_sales) * 0.8)
    train_data = daily_sales[:split_point]
    
    if seasonal_adjustment == 'Weekly':
        daily_season = True
        weekly_season = True
        yearly_season = False
    elif seasonal_adjustment == 'Monthly':
        daily_season = False
        weekly_season = True
        yearly_season = True
    elif seasonal_adjustment == 'Quarterly':
        daily_season = False
        weekly_season = False
        yearly_season = True
    else:
        daily_season = True
        weekly_season = True
        yearly_season = False
    
    if model_type == "Prophet with Holidays":
        model = Prophet(
            daily_seasonality=daily_season,
            weekly_seasonality=weekly_season,
            yearly_seasonality=True,
            interval_width=confidence_level / 100,
            changepoint_prior_scale=0.05
        )
        model.add_country_holidays(country_name=holiday_country)
    elif model_type == "Prophet Enhanced":
        model = Prophet(
            daily_seasonality=daily_season,
            weekly_seasonality=weekly_season,
            yearly_seasonality=yearly_season,
            interval_width=confidence_level / 100,
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=15.0
        )
        model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        # st.info("✅ Enabled model with flexible trend detection")
        
    else:
        model = Prophet(
            daily_seasonality=daily_season,
            weekly_seasonality=weekly_season,
            yearly_seasonality=yearly_season,
            interval_width=confidence_level / 100,
            changepoint_prior_scale=0.05,
        )
        # st.info("✅ Standard Prophet Model")
        
    if include_holidays and model_type == "Prophet (Default)":
        model.add_country_holidays(country_name=holiday_country)
        # st.info("🎉 Indian holidays included in forecast")
        
    seasonality_msg = f"📊 Seasonality: "
    if daily_season:
        seasonality_msg += "Daily ✓ "
    if weekly_season:
        seasonality_msg += "Weekly ✓ "
    if yearly_season:
        seasonality_msg += "Yearly ✓ "
    st.info(seasonality_msg)
            
    with st.spinner(f"🤖 Training {model_type} (Confidence: {confidence_level}%)..."):
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

    if controls.get('auto_zoom_forecast', True) and controls['forecast_days'] > 30:

        forecast_start = forecast['ds'][len(daily_sales):].min()
        forecast_end = forecast['ds'].max()

        buffer_days = (forecast_end - forecast_start).days * 0.1
        forecast_start = forecast_start - pd.Timedelta(days=buffer_days)

        fig.update_xaxes (
            range=[forecast_start, forecast_end],
            rangeselector=dict(
                buttons=list([
                    dict(step="all", label="All"),
                ]),
                bgcolor="black",
                bordercolor="#667eea",
                borderwidth=1
            )
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    return forecast

# Business Insights Function
def display_business_insights(forecast, daily_sales, controls):
    st.subheader("💡 Business Insights & Recommendations")
    st.caption("Simple actions based on predictions")
    
    total_forecast_length = len(forecast)
    historical_length = len(daily_sales)
    
    future_forecast = forecast.tail(controls['forecast_days']).copy().reset_index(drop=True)
    
    if len(future_forecast) == 0:
        st.error("❌ No forecast data generated!")
        st.info("""
        **Possible causes:**
        - Forecast period is 0 days
        - Model training failed
        - Data processing error
        
        **Try:**
        - Refresh the page
        - Check your forecast settings
        - Upload data again
        """)
        return
    
    tabs_to_show = ["📅 Daily (7d)"]
    
    if controls['forecast_days'] >= 28:
        tabs_to_show.append("📊 Weekly")
        
    if controls['forecast_days'] >= 60:
        tabs_to_show.append("📅 Monthly")
    
    if controls['forecast_days'] >= 180:
        tabs_to_show.append("🗓️ Quarterly")
        
    tabs = st.tabs(tabs_to_show)
    
    # TAB 1: DAILY (existing code)
    with tabs[0]:
        st.markdown("### 📅 Next 7 Days Detailed Forecast")
        
        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown(f"#### 📊 Day-by-Day Breakdown")

            insight_days = min(controls['forecast_days'], 7)
            next_days = future_forecast.head(insight_days)
            
            if len(next_days) == 0:
                st.warning("⚠️ No daily data available")
            else:
                for idx, row in next_days.iterrows():
                    day_name = row['ds'].strftime('%A')
                    date_str = row['ds'].strftime('%d %b %Y')
                    predicted_sales = int(row['yhat'])

                    # Simple visual indicator
                    if predicted_sales > daily_sales['y'].mean() * 1.1:
                        icon = "🔥"
                        label = "High Sales Day"
                        color = "🟢"
                    elif predicted_sales < daily_sales['y'].mean() * 0.9:
                        icon = "📉"
                        label = "Low Sales Day"
                        color = "🔴"
                    else:
                        icon = "➡️"
                        label = "Normal Day"
                        color = "🟡"

                    st.write(f"""{icon} **{day_name}** ({date_str}): ~**{predicted_sales:,}** units - *{label}*""")

                if controls['forecast_days'] <= 7:
                    total_forecast = future_forecast['yhat'].sum()
                    st.success(f"📊 **Total Next {controls['forecast_days']} Days**: {total_forecast:,.0f} units")
                else:
                    weekly_total = next_days['yhat'].sum()
                    full_total = future_forecast['yhat'].sum()

                    st.success(f"📊 **Next {insight_days} Days**: {weekly_total:,.0f} units")
                    st.info(f"📊 **Full {controls['forecast_days']}-Day Total**: {full_total:,.0f} units")

        with col2:
            st.markdown("#### 🎯 Recommended Actions")

            if len(next_days) > 0:
                peak_day_7 = next_days.loc[next_days['yhat'].idxmax()]
                low_day_7 = next_days.loc[next_days['yhat'].idxmin()]

                st.success(f"""
                **🔥 Peak Day This Week**  
                {peak_day_7['ds'].strftime('%A, %d %B %Y')}  
                Expected: ~{int(peak_day_7['yhat']):,} units
                
                **Actions:**
                - ✅ Stock extra {int(peak_day_7['yhat'] * 0.2):,} units
                - ✅ Schedule +2 staff members
                - ✅ Prepare checkout stations
                """)
                
                st.info(f"""
                **📉 Slowest Day This Week**  
                {low_day_7['ds'].strftime('%A, %d %B %Y')}  
                Expected: ~{int(low_day_7['yhat']):,} units
                
                **Actions:**
                - 🎯 Run flash sale (15-20% off)
                - 💰 Bundle offers on slow items
                - 📣 Social media promotions
                """)
                
    # TAB 2: WEEKLY - FIXED VERSION
    if controls['forecast_days'] >= 28:
        with tabs[1]:  # ✅ IMPORTANT: All weekly content MUST be inside this block
            st.markdown("### 📊 Weekly Performance Forecast")
            
            future_forecast['week_start'] = future_forecast['ds'] - pd.to_timedelta(
                future_forecast['ds'].dt.dayofweek, unit='D'
            )
            
            weekly_forecast = future_forecast.groupby('week_start').agg({
                'yhat': 'sum',
                'yhat_lower': 'sum',
                'yhat_upper': 'sum'
            }).reset_index()
            
            num_weeks = len(weekly_forecast)
            
            # ✅ IMPROVED GRID LAYOUT (Less Suffocated)
            if num_weeks > 8:
                st.markdown(f"#### 📈 {num_weeks}-Week Overview")
                
                # Summary at top
                col1, col2, col3, col4 = st.columns(4)
                
                total_forecast = weekly_forecast['yhat'].sum()
                avg_weekly = weekly_forecast['yhat'].mean()
                best_week = weekly_forecast.loc[weekly_forecast['yhat'].idxmax()]
                worst_week = weekly_forecast.loc[weekly_forecast['yhat'].idxmin()]
                
                with col1:
                    st.metric("📊 Total", f"{total_forecast:,.0f}")
                with col2:
                    st.metric("📈 Avg/Week", f"{avg_weekly:,.0f}")
                with col3:
                    st.metric("🔥 Peak", f"{int(best_week['yhat']):,}")
                with col4:
                    st.metric("📉 Low", f"{int(worst_week['yhat']):,}")
                
                st.markdown("---")
                st.markdown("#### 📅 Week-by-Week Breakdown")
                
                # ✅ LESS CRAMPED: 2 columns instead of 3
                cols_per_row = 2
                for i in range(0, num_weeks, cols_per_row):
                    cols = st.columns(cols_per_row)
                    
                    for j in range(cols_per_row):
                        idx = i + j
                        if idx >= num_weeks:
                            break
                        
                        row = weekly_forecast.iloc[idx]
                        week_start = row['week_start'].strftime('%d %b')
                        week_end = (row['week_start'] + pd.Timedelta(days=6)).strftime('%d %b')
                        weekly_sales = int(row['yhat'])
                        
                        if idx > 0:
                            prev = int(weekly_forecast.iloc[idx-1]['yhat'])
                            growth = ((weekly_sales - prev) / prev) * 100
                            delta = f"{growth:+.1f}%"
                            delta_color = "normal" if growth > 0 else "inverse"
                        else:
                            delta = "Baseline"
                            delta_color = "off"
                        
                        with cols[j]:
                            st.metric(
                                label=f"**Week {idx+1}**",
                                value=f"{weekly_sales:,} units",
                                delta=delta,
                                delta_color=delta_color
                            )
                            st.caption(f"📅 {week_start} - {week_end}")
                            
                            # ✅ ADD BREATHING ROOM
                            st.markdown("<div style='margin-bottom: 16px;'></div>", 
                                      unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Strategic recommendations
                col1, col2 = st.columns(2)
                
                with col1:
                    best_week_num = weekly_forecast[
                        weekly_forecast['yhat'] == best_week['yhat']
                    ].index[0] + 1
                    
                    st.success(f"""
                    **🏆 Peak Week #{best_week_num}**  
                    {best_week['week_start'].strftime('%d %b')} - {
                        (best_week['week_start'] + pd.Timedelta(days=6)).strftime('%d %b')
                    }  
                    Expected: **{int(best_week['yhat']):,}** units
                    
                    **Actions:**
                    - 📦 +{int(best_week['yhat'] * 0.2):,} safety stock
                    - 👥 Full team roster
                    - 🚚 Extra delivery slots
                    """)
                
                with col2:
                    worst_week_num = weekly_forecast[
                        weekly_forecast['yhat'] == worst_week['yhat']
                    ].index[0] + 1
                    
                    st.warning(f"""
                    **📊 Recovery Week #{worst_week_num}**  
                    {worst_week['week_start'].strftime('%d %b')} - {
                        (worst_week['week_start'] + pd.Timedelta(days=6)).strftime('%d %b')
                    }  
                    Expected: **{int(worst_week['yhat']):,}** units
                    
                    **Boost:**
                    - 💰 Week-long promo
                    - 🎁 Loyalty 2x points
                    - 📧 Email blast
                    """)
            
            else:  # ≤8 weeks - traditional layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"#### 📈 Weekly Breakdown ({num_weeks} weeks)")
                    
                    for idx, row in weekly_forecast.iterrows():
                        week_start = row['week_start'].strftime('%d %b')
                        week_end = (row['week_start'] + pd.Timedelta(days=6)).strftime('%d %b %Y')
                        weekly_sales = int(row['yhat'])
                        
                        if idx > 0:
                            prev_week = int(weekly_forecast.iloc[idx-1]['yhat'])
                            growth = ((weekly_sales - prev_week) / (prev_week + 1e-10)) * 100
                            
                            if growth > 5:
                                trend = f"📈 +{growth:.1f}%"
                            elif growth < -5:
                                trend = f"📉 {growth:.1f}%"
                            else:
                                trend = f"➡️ {growth:+.1f}%"
                        else:
                            trend = "🔵 Baseline"
                            
                        st.markdown(f"""
                        **Week {idx+1}**: {week_start} - {week_end}  
                        💰 **{weekly_sales:,}** units | {trend}
                        """)
                    
                    total_forecast = weekly_forecast['yhat'].sum()
                    st.success(f"📊 **Total {num_weeks}-Week Forecast**: {total_forecast:,.0f} units")
                    
                with col2:
                    st.markdown("#### 🎯 Weekly Strategy")
                    
                    best_week = weekly_forecast.loc[weekly_forecast['yhat'].idxmax()]
                    worst_week = weekly_forecast.loc[weekly_forecast['yhat'].idxmin()]
                                
                    best_week_num = weekly_forecast[weekly_forecast['yhat'] == best_week['yhat']].index[0] + 1
                    worst_week_num = weekly_forecast[weekly_forecast['yhat'] == worst_week['yhat']].index[0] + 1
                    
                    st.success(f"""
                    **🏆 Best Week: #{best_week_num}**  
                    {best_week['week_start'].strftime('%d %b')} - {(best_week['week_start'] + pd.Timedelta(days=6)).strftime('%d %b')}  
                    Expected: ~{int(best_week['yhat']):,} units
                    
                    **Prepare:**
                    - 📦 Order {int(best_week['yhat'] * 1.15):,} units
                    - 👥 Full staff roster
                    - 🚚 Extra delivery
                    """)
                    
                    st.warning(f"""
                    **⚠️ Slowest Week: #{worst_week_num}**  
                    {worst_week['week_start'].strftime('%d %b')} - {(worst_week['week_start'] + pd.Timedelta(days=6)).strftime('%d %b')}  
                    Expected: ~{int(worst_week['yhat']):,} units
                    
                    **Actions:**
                    - 💰 Weekly promotion
                    - 🎁 Loyalty boost
                    - 📧 Email campaign
                    """)
    
    # TAB 3: MONTHLY (existing code)
    if controls['forecast_days'] >= 60:
        with tabs[2]:  # ✅ Make sure it's tabs[2], not tabs[1]
            st.markdown("### 📅 Monthly Performance Forecast")
            
            future_forecast['month'] = future_forecast['ds'].dt.to_period('M')
            
            monthly_forecast = future_forecast.groupby('month').agg({
                'yhat': 'sum',
                'yhat_lower': 'sum',
                'yhat_upper': 'sum'
            }).reset_index()
            
            monthly_forecast['month_name'] = monthly_forecast['month'].dt.strftime('%B %Y')
            
            num_months = len(monthly_forecast)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown(f"#### 📈 Monthly Breakdown ({num_months} months)")
                
                for idx, row in monthly_forecast.iterrows():
                    month_name = row['month_name']
                    monthly_sales = int(row['yhat'])
                    lower = int(row['yhat_lower'])
                    upper = int(row['yhat_upper'])
                    
                    if idx > 0:
                        prev = int(monthly_forecast.iloc[idx-1]['yhat'])
                        growth = ((monthly_sales - prev) / prev) * 100
                        
                        if growth > 10:
                            icon = "🚀"
                            status = "Strong Growth"
                        elif growth > 0:
                            icon = "📈"
                            status = "Growing"
                        else:
                            icon = "📉"
                            status = "Declining"
                        
                        trend = f"{icon} {growth:+.1f}% - {status}"
                    else:
                        trend = "🔵 Baseline"
                        
                    st.markdown(f"""
                    **{month_name}**
                    💰 **{monthly_sales:,}** units (Range: {lower:,} - {upper:,})
                    {trend}
                    """)
                    st.markdown("---")
                
                total = monthly_forecast['yhat'].sum()
                avg = monthly_forecast['yhat'].mean()
                
                st.success(f"""
                📊 **Total {num_months}-Month**: {total:,.0f} units  
                📈 **Avg Per Month**: {avg:,.0f} units
                """)
                
            with col2:
                st.markdown("#### 🎯 Monthly Strategy")
                
                best = monthly_forecast.loc[monthly_forecast['yhat'].idxmax()]
                worst = monthly_forecast.loc[monthly_forecast['yhat'].idxmin()]
                
                st.success(f"""
                **🏆 Best Month:**
                {best['month_name']}
                Expected: ~{int(best['yhat']):,} units
                
                **Strategic Actions:**
                - 📦 Secure {int(best['yhat'] * 1.2):,} units
                - 💼 Bulk supplier deals
                - 👥 Hire seasonal staff
                - 📈 Max marketing budget
                """)
                
                st.info(f"""
                **📊 Lowest Month**  
                {worst['month_name']}  
                Expected: ~{int(worst['yhat']):,} units
                
                **Recovery Plan:**
                - 💰 Month-long sale
                - 🎁 Retention programs
                - 📧 Re-engagement
                - 🔄 Inventory clearance
                """)
                
    # TAB 4: QUARTERLY (existing code)
    if controls['forecast_days'] >= 180:
        with tabs[3]:  # ✅ Make sure it's tabs[3]
            st.markdown("### 📅 Quarterly Performance Forecast")
            
            future_forecast['quarter'] = future_forecast['ds'].dt.to_period('Q')
            
            quarterly_forecast = future_forecast.groupby('quarter').agg({
                'yhat': 'sum',
                'yhat_lower': 'sum',
                'yhat_upper': 'sum',
            }).reset_index()
            
            quarterly_forecast['quarter_name'] = quarterly_forecast['quarter'].astype(str)
            
            num_quarters = len(quarterly_forecast)
            
            st.markdown(f"#### 📊 Quarterly Performance ({num_quarters} quarters)")
            
            cols = st.columns(min(num_quarters, 3))
            
            for idx, row in quarterly_forecast.iterrows():
                qtr_name = row['quarter_name']
                qtr_sales = int(row['yhat'])
                lower = int(row['yhat_lower'])
                upper = int(row['yhat_upper'])
            
                parts = qtr_name.split('Q')
                year = parts[0]
                qtr = f"Q{parts[1]}"
                
                if idx > 0:
                    prev = int(quarterly_forecast.iloc[idx-1]['yhat'])
                    growth = ((qtr_sales - prev) / prev) * 100
                    delta = f"{growth:+.1f}%"
                    delta_color = "normal" if growth > 0 else "inverse"
                else:
                    delta = "Baseline"
                    delta_color = "off"
                    
                with cols[idx % 3]:
                    st.metric(
                        label=f"{qtr} {year}",
                        value=f"{qtr_sales:,} units",
                        delta=delta,
                        delta_color=delta_color
                    )
                    st.caption(f"Range: {lower:,} - {upper:,}")
                    
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Quarterly Plan")
                
                best = quarterly_forecast.loc[quarterly_forecast['yhat'].idxmax()]
                worst = quarterly_forecast.loc[quarterly_forecast['yhat'].idxmin()]
                
                st.success(f"""
                **🏆 Strongest: {best['quarter_name']}**
                Expected: ~{int(best['yhat']):,} units
                
                **Strategic Initiatives:**
                - 💼 Negotiate annual supplier contracts   
                - 📦 Reserve warehouse capacity         
                - 👥 Seasonal hiring (3mo ahead)         
                - 💰 Max marketing
                - 🚚 Logistics partnerships
                """)
                
                st.warning(f"""
                **📊 Weakest: {worst['quarter_name']}**  
                Expected: ~{int(worst['yhat']):,} units
                
                **Recovery:**
                - 💰 Quarterly clearance
                - 🎁 Loyalty overhaul
                - 📧 Multi-channel campaign
                - 🔄 Product mix optimization
                - 💼 Cost reduction
                """)

            with col2:
                st.markdown("#### 📈 YoY Analysis")
                
                total = quarterly_forecast['yhat'].sum()
                avg = quarterly_forecast['yhat'].mean()
                
                std = quarterly_forecast['yhat'].std()
                volatility = (std / avg) * 100
                
                st.info(f"""
                **Forecast Summary:**
                
                📊 **Total**: {total:,.0f} units  
                📈 **Avg/Quarter**: {avg:,.0f} units  
                📉 **Volatility**: {volatility:.1f}%
                
                **Interpretation:**
                """)
                
                if volatility < 10:
                    st.success("✅ **Stable** - Consistent demand")
                elif volatility < 20:
                    st.info("📊 **Moderate** - Some seasonality")
                else:
                    st.warning("⚠️ **High Volatility** - Strong seasonal effects")
                    
                if num_quarters >= 2:
                    q1 = quarterly_forecast.iloc[0]['yhat']
                    q_last = quarterly_forecast.iloc[-1]['yhat']
                    trend = ((q_last - q1) / q1) * 100
                    
                    st.markdown("#### 🎯 Long-Term Trend")

                    if trend > 10:
                        st.success(f"📈 **Growing Market** (+{trend:.1f}%)")
                        st.write("→ Expand capacity")
                    elif trend > 0:
                        st.info(f"➡️ **Stable Growth** (+{trend:.1f}%)")
                        st.write("→ Maintain strategy")
                    else:
                        st.warning(f"📉 **Declining** ({trend:.1f}%)")
                        st.write("→ Reassess fit")       
        
                
# Business Intelligence Dashboard
def create_business_dashboard(df, forecast, controls):
    if not controls['show_business_dashboard']:
        return
    
    st.subheader("🎯 Executive Business Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate revenue per unit correctly
    # Handle missing or zero total_price column
    revenue_per_unit = 0
    
    # Try total_price first
    if 'total_price' in df.columns:
        valid_prices = df['total_price'].dropna()
        if len(valid_prices) > 0 and valid_prices.sum() > 0:
            total_revenue = valid_prices.sum()
            total_units = df['units_sold'].sum()
            revenue_per_unit = total_revenue / total_units if total_units > 0 else 0
    
    # Fallback to base_price if total_price didn't work
    if revenue_per_unit == 0 and 'base_price' in df.columns:
        valid_base = df['base_price'].dropna()
        if len(valid_base) > 0 and valid_base.sum() > 0:
            revenue_per_unit = valid_base.mean()
    
    # Last resort: default estimate
    if revenue_per_unit == 0:
        revenue_per_unit = 75.0
    
    last_historical_date = df['date'].max()
    future_forecast = forecast[forecast['ds'] > last_historical_date].head(controls['forecast_days'])
    
    # DEBUG: Check if future_forecast is empty
    if len(future_forecast) == 0:
        # Fallback: use tail of forecast if filtering by date fails
        future_forecast = forecast.tail(controls['forecast_days']).copy()
    
    forecast_revenue = future_forecast['yhat'].sum() * revenue_per_unit if revenue_per_unit > 0 else 0
    
    with col1:
        st.metric(
            "💰 Projected Revenue",
            f"${forecast_revenue:,.0f}",
            delta=f"{controls['forecast_days']} days",
            help=f"Based on ${revenue_per_unit:.2f} avg revenue/unit"
        )
        st.caption(f"📊 ${revenue_per_unit:.2f} per unit")
        
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
@st.cache_data(ttl=3600)
def _calculate_model_performance(daily_sales_dict):
    """Helper function to cache model performance calculations"""
    daily_sales = pd.DataFrame(daily_sales_dict)
    
    if  'ds' not in daily_sales_dict or 'y' not in daily_sales.columns:
        raise ValueError("DataFrame must have 'ds' and 'y' columns")
    
    daily_sales['ds'] = pd.to_datetime(daily_sales['ds'])
    daily_sales['y'] = pd.to_numeric(daily_sales['y'])
    
    train_size = int(len(daily_sales) * 0.8)
    train_data = daily_sales[:train_size].copy()
    test_data = daily_sales[train_size:].copy()
    
    if len(train_data) < 2:
        raise ValueError("Not enough data for training (minimum 2 records)")
    
    test_model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    test_model.fit(train_data)
    
    future_test = test_model.make_future_dataframe(periods=len(test_data))
    forecast_test = test_model.predict(future_test)
    
    return forecast_test, test_data

def display_model_performance(model, daily_sales, controls):
    st.subheader("🎯 How Reliable Are These Predictions?")
    st.caption("Check prediction accuracy")
    
    # Convert to tuple for caching
    daily_sales_dict = daily_sales.to_dict('list')
    
    try:
        forecast_test, test_data = _calculate_model_performance(daily_sales_dict)
    except ValueError as e:
        st.error(f"❌ Cannot calculate performance: {str(e)}")
        st.info("💡 Need at least 30 days of historical data for accuracy metrics")
        return
    except Exception as e:
        st.error(f"❌ Performance calculation failed: {str(e)}")
        return        
    
    y_true = test_data['y'].values
    y_pred = forecast_test.tail(len(test_data))['yhat'].values
    
    # Ensure both arrays are numpy arrays to avoid type issues
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # Calculate metrics with error handling
    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100  
        accuracy = max(0, 100 - mape)
    except Exception as e:
        st.error(f"❌ Metric calculation failed: {str(e)}")
        return
    
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
        error_percent = (rmse / (avg_sales + 1e-10)) * 100
        
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
        
    example_forecast = 1000
    lower_bound = max(0, example_forecast - rmse)
    upper_bound = example_forecast + rmse
    
    st.info(f"""
    **Understanding These Numbers:**
    
    ✅ The model is **{accuracy:.1f}% accurate** in predicting sales  
    📊 Predictions typically vary by **±{rmse:,.0f} units** from actual  
    🎯 **Example:** If forecast shows {example_forecast} units, actual sales will likely be between {lower_bound:,.0f} and {upper_bound:,.0f}
    
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
                options=sorted(df['store_id'].astype(str).unique()),
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
    
    with st.sidebar.expander("🎲 Generate Sample Data", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            num_days = st.selectbox(
                "Date Period:",
                [30, 90, 180, 365],
                index=1,
                help="How many days of data to generate"
            )
            
        with col2:
            num_stores = st.selectbox(
                "Stores:",
                [5, 10, 15, 20],
                index=1,
                help="Number of Retail locations"   
            )    
        
        if st.button("🎲 Generate Sample Data", type="primary", use_container_width=True):
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
                
                st.success(f"✅ Generated {len(sample_df):,} records!")
                st.success(f"📅 Period: {num_days} days • 🏪 Stores: {num_stores}")
                
                csv = sample_df.to_csv(index=False)
                st.download_button(
                    "💾 Download Sample Data",
                    csv,
                    f"sample_retail_data_{num_days}days.csv",
                    "text/csv",
                    help="Save this sample data for later use",
                    use_container_width=True
                )
                
                return sample_df
    return None


# Enhanced Sidebar Controls
def create_enhanced_sidebar_controls():
    st.sidebar.header("⚙️ Dashboard Settings")
    
    # Section 1: Data Upload (always visible)
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
    
    st.sidebar.markdown("---")
    
    # Section 2: Forecast Settings (collapsible)
    with st.sidebar.expander("📅 Forecast Settings", expanded=True):
        forecast_method = st.radio(
            "Forecast Method:",  
            ["📊 Days from Last Date", "📅 Specific Date Range", "🎯 Next N Business Days"],
            help="How to calculate forecast period"
        )

        if forecast_method == "📊 Days from Last Date":
            forecast_days = st.slider(
                "Number of days to predict:",
                min_value=7,
                max_value=365,
                value=30,
                step=7,
                help="How many days into the future you want to see"  
            )
            start_date = None
            end_date = None
        elif forecast_method == "📅 Specific Date Range":
            start_date = st.date_input("Forecast start date:")
            end_date = st.date_input("Forecast end date:")

            if start_date and end_date:
                forecast_days = (end_date - start_date).days
                if forecast_days <= 0:
                    st.error("End date must be after start date!")
                    forecast_days = 7
            else:
                forecast_days = 7
        else:
            business_days = st.slider(
                "Number of business days:",  
                min_value=5,
                max_value=250,
                value=20,
                step=5,
                help="Weekends will be excluded automatically"
            )
            forecast_days = int(business_days * 1.4)
            start_date = None
            end_date = None
        
        auto_zoom_forecast = st.checkbox(
            "Auto-Zoom to Forecast Period",
            value=True,
            help="Automatically focus chart on predicted period"
        )

    # Section 3: Model Settings (collapsible)
    with st.sidebar.expander("🔬 Advanced Settings", expanded=False):
        model_type = st.selectbox(
            "Forecasting Algorithm:",  
            ["Prophet (Default)", "Prophet with Holidays", "Prophet Enhanced"],
            help="Select forecasting model variant"  
        )

        confidence_level = st.slider(
            "Confidence level:",  
            min_value=80,
            max_value=99,
            value=95,
            help="Prediction Confidence (%)" 
        )

        include_holidays = st.checkbox(
            "Include holidays",
            help="Account for Indian holidays"
        )

        holiday_country = st.selectbox(
            "Holiday Region:", ["IN", "US", "UK", "AU"], 
            index=0
        )

        seasonal_adjustment = st.selectbox(
            "Seasonal pattern focus:",
            ["Auto", "Weekly", "Monthly", "Quarterly"],
            help="Seasonal patterns to emphasize"
        )
        
    # Section 4: Display Options (collapsible)
    with st.sidebar.expander("👁️ Display Options", expanded=True):
        show_confidence = st.checkbox(
            "Show Confidence Bands",
            value=True,
            help="Display Prediction Ranges"
        )

        show_raw_data = st.checkbox(
            "Show Data Tables",
            value=False,
            help="View raw data"
        )

        show_model_details = st.checkbox(
            "Show Model Accuracy",  
            value=True,
            help="Display Performance Metrics"
        )

        show_business_dashboard = st.checkbox(
            "Show Business Dashboard", 
            value=True,
            help="Display Actions and KPIs"
        )

        show_data_quality = st.checkbox(
            "Show Data Quality Report",  
            value=False,
            help="Check Data Quality"
        )

        show_alerts = st.checkbox(
            "Show Business Alerts",  
            value=True,
            help="Get Trend Notifications"
        )
    
    # Section 5: Visual Settings (collapsible)
    with st.sidebar.expander("🎨 Visual Settings", expanded=True):
        chart_theme = st.selectbox(
            "Chart Theme:",
            ["Default", "Dark", "Colorful", "Minimal"],
            help="Chart Visual Appearance"
        )

        chart_height = st.slider(
            "Chart Height (px):",
            min_value=300,
            max_value=800,
            value=500,
            step=50
        )
    
    return {
        'uploaded_file': uploaded_file,
        'forecast_method': forecast_method,
        'forecast_days': forecast_days,
        'auto_zoom_forecast': auto_zoom_forecast,
        'start_date': start_date,
        'end_date': end_date,
        'model_type': model_type,
        'confidence_level': confidence_level,
        'include_holidays': include_holidays,
        'holiday_country': holiday_country,
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
    
    last_historical_date = daily_sales['ds'].max()
    future_forecast = forecast[forecast['ds'] > last_historical_date].head(controls['forecast_days'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Prediction Table")
        
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
        st.markdown("#### 📋 Business Report")
        
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
    st.title("📈 RetailVision")
    st.markdown("### 🎯 *Data-Powered Professional Forecasting System for Retail Business Intelligence*")
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
    
    # Train model 
    with st.spinner("🧠 Training forecasting model..."):
        model, train_data = train_forecasting_model(
            daily_sales,
            model_type=controls['model_type'],
            confidence_level=controls['confidence_level'],
            include_holidays=controls['include_holidays'],
            seasonal_adjustment=controls['seasonal_adjustment'],
            holiday_country=controls.get('holiday_country', 'IN')
        )
    
    # Create forecast chart
    forecast = create_enhanced_forecast_chart(daily_sales, model, controls)
    
    # Business insights
    display_business_insights(forecast, daily_sales, controls)
    
    # Business dashboard (ADD THIS)
    create_business_dashboard(df, forecast, controls)
    
    # Alert system (ADD THIS)
    create_alert_system(forecast, daily_sales, controls)
    
    # Model performance
    if controls['show_model_details']:
        display_model_performance(model, daily_sales, controls)
    
    # Data quality report (ADD THIS)
    if controls.get('show_data_quality', False):
        create_data_quality_report(df)
    
    # Data explorer
    display_data_explorer(df, daily_sales, controls)
    
    # Export section
    create_export_section(forecast, daily_sales, controls)
    
    st.markdown("""
    ---
    <div style='text-align: center; padding: 20px; color: #666;'>
        <p><strong>RetailVision</strong> • Built with Streamlit 🎨 • Prophet 🔮 • Plotly 📊</p>
        <p style='font-size: 12px;'>© 2025 • Rohit Navinchandra Kandpal</p>
    </div>
    """, unsafe_allow_html=True)    
    
if __name__ == "__main__":
    main()