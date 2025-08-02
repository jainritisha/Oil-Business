# main_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from fpdf import FPDF
import base64
from datetime import date, timedelta
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Cold-Pressed Oil Business Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---

def generate_sample_data():
    """Generates a sample sales DataFrame for demonstration."""
    today = date.today()
    dates = [today - timedelta(days=i) for i in range(365)]
    oil_types = ['Sesame Oil', 'Coconut Oil', 'Groundnut Oil', 'Mustard Oil', 'Almond Oil']
    customers = [f'Customer_{i}' for i in range(1, 51)]
    
    data = []
    for d in dates:
        num_sales = np.random.randint(5, 15)
        for _ in range(num_sales):
            oil = np.random.choice(oil_types)
            customer = np.random.choice(customers)
            quantity = np.random.randint(1, 5)
            price = {'Sesame Oil': 350, 'Coconut Oil': 300, 'Groundnut Oil': 400, 'Mustard Oil': 250, 'Almond Oil': 800}[oil]
            current_stock = np.random.randint(5, 50)
            data.append({
                'Date': d,
                'InvoiceNo': f'INV-{np.random.randint(1000, 9999)}',
                'CustomerName': customer,
                'ProductName': oil,
                'Quantity': quantity,
                'Price': price,
                'CurrentStock': current_stock
            })
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def create_download_link_pdf(val, filename):
    """Creates a download link for the generated PDF report."""
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download PDF Report</a>'

class PDF(FPDF):
    """Custom PDF class to define header and footer."""
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Monthly Business Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(kpis, top_oils_df, low_stock_df):
    """Generates a PDF report with key business insights."""
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    
    # Title
    pdf.cell(0, 10, f"Report for {date.today().strftime('%B %Y')}", 0, 1, 'L')
    pdf.ln(5)

    # KPIs Section
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Key Performance Indicators (KPIs)', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 8, f"  - Total Sales: Rs. {kpis['total_sales']:,}", 0, 1, 'L')
    pdf.cell(0, 8, f"  - Top-Selling Oil: {kpis['top_oil']}", 0, 1, 'L')
    pdf.cell(0, 8, f"  - Repeat Customer Percentage: {kpis['repeat_customer_pct']:.2f}%", 0, 1, 'L')
    pdf.ln(10)

    # Top Selling Oils Section
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Top 3 Selling Oils (by Quantity)', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    for index, row in top_oils_df.iterrows():
        pdf.cell(0, 8, f"  - {row['ProductName']}: {row['Quantity']} units", 0, 1, 'L')
    pdf.ln(10)

    # Low Stock Alerts Section
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Low Stock Items', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    if not low_stock_df.empty:
        for index, row in low_stock_df.iterrows():
            pdf.cell(0, 8, f"  - {row['ProductName']}: {row['CurrentStock']} units remaining", 0, 1, 'L')
    else:
        pdf.cell(0, 8, "  All stock levels are healthy.", 0, 1, 'L')
    
    return pdf.output(dest='S').encode('latin-1')


# --- Sidebar ---
st.sidebar.title("💧 Oil Business Dashboard")
st.sidebar.markdown("Upload your sales data to get started.")

# File Uploader
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV or Excel file", 
    type=['csv', 'xlsx']
)

# Sample Data Generation
if st.sidebar.button('Generate Sample Data'):
    df_sample = generate_sample_data()
    # Convert DataFrame to CSV in-memory
    towrite = io.BytesIO()
    df_sample.to_csv(towrite, index=False, encoding='utf-8')
    towrite.seek(0)
    st.sidebar.download_button(
        label="Download Sample CSV",
        data=towrite,
        file_name="sample_sales_data.csv",
        mime="text/csv",
    )
    st.sidebar.info("A sample CSV has been generated. Download it and upload it above to test the dashboard.")

# --- Main Dashboard ---
if uploaded_file is not None:
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # --- Data Cleaning and Preparation ---
        st.header("Raw Data Preview")
        st.dataframe(df.head())

        # Standardize column names (basic example)
        df.columns = [col.strip().replace(' ', '') for col in df.columns]

        # Ensure required columns exist
        required_cols = ['Date', 'CustomerName', 'ProductName', 'Quantity', 'Price', 'CurrentStock']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Error: Your file is missing one or more required columns. Please ensure it has: {', '.join(required_cols)}")
        else:
            # Data type conversion and feature engineering
            df['Date'] = pd.to_datetime(df['Date'])
            df['TotalSale'] = df['Quantity'] * df['Price']
            df_kpi = df.copy() # Use a copy for date-filtered KPIs if needed

            # --- KPI Calculations ---
            total_sales = int(df_kpi['TotalSale'].sum())
            top_oil = df_kpi.groupby('ProductName')['Quantity'].sum().idxmax()
            
            customer_counts = df_kpi['CustomerName'].value_counts()
            repeat_customers = customer_counts[customer_counts > 1].count()
            total_customers = df_kpi['CustomerName'].nunique()
            repeat_customer_pct = (repeat_customers / total_customers) * 100 if total_customers > 0 else 0

            kpis = {
                'total_sales': total_sales,
                'top_oil': top_oil,
                'repeat_customer_pct': repeat_customer_pct
            }

            # --- Dashboard Layout ---
            st.header("Business Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Sales", f"₹{total_sales:,}")
            col2.metric("Top-Selling Oil", top_oil)
            col3.metric("Repeat Customers", f"{repeat_customer_pct:.2f}%")

            st.markdown("---")

            # --- Charts ---
            c1, c2 = st.columns((6, 4))
            with c1:
                st.subheader("Daily Sales Trend")
                daily_sales = df_kpi.groupby(df_kpi['Date'].dt.date)['TotalSale'].sum().reset_index()
                fig_daily_sales = px.line(daily_sales, x='Date', y='TotalSale', title='Total Sales Over Time', labels={'TotalSale': 'Total Sales (₹)'})
                fig_daily_sales.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Total Sales (₹)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_daily_sales, use_container_width=True)

            with c2:
                st.subheader("Top 3 Selling Oils")
                top_oils_df = df_kpi.groupby('ProductName')['Quantity'].sum().nlargest(3).reset_index()
                fig_top_oils = px.bar(top_oils_df, x='ProductName', y='Quantity', title='Top 3 Oils by Quantity Sold', text='Quantity')
                fig_top_oils.update_layout(
                    xaxis_title='Oil Type',
                    yaxis_title='Quantity Sold',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_top_oils, use_container_width=True)
            
            st.markdown("---")

            # --- Alerts and Forecasting ---
            a1, a2 = st.columns(2)
            with a1:
                st.subheader("🚨 Low Stock Alerts")
                stock_threshold = st.slider("Set low stock threshold (units):", 1, 50, 10)
                latest_stock = df.sort_values('Date').drop_duplicates('ProductName', keep='last')
                low_stock_df = latest_stock[latest_stock['CurrentStock'] <= stock_threshold][['ProductName', 'CurrentStock']]
                
                if not low_stock_df.empty:
                    st.warning(f"The following items are below the {stock_threshold} unit threshold:")
                    st.dataframe(low_stock_df.style.highlight_max(axis=0, color='darkred'))
                else:
                    st.success("All stock levels are healthy!")
            
            with a2:
                st.subheader("📈 30-Day Demand Forecasting")
                
                # Prepare data for forecasting
                daily_demand = df.groupby(df['Date'].dt.date)['Quantity'].sum().reset_index()
                daily_demand['Date'] = pd.to_datetime(daily_demand['Date'])
                daily_demand['Time'] = (daily_demand['Date'] - daily_demand['Date'].min()).dt.days

                # Simple Linear Regression Model
                X = daily_demand[['Time']]
                y = daily_demand['Quantity']
                model = LinearRegression()
                model.fit(X, y)

                # Predict for the next 30 days
                last_time = daily_demand['Time'].max()
                future_times = np.arange(last_time + 1, last_time + 31).reshape(-1, 1)
                future_dates = [daily_demand['Date'].max() + timedelta(days=i) for i in range(1, 31)]
                
                future_preds = model.predict(future_times)
                future_preds = np.maximum(0, future_preds) # Demand can't be negative

                forecast_df = pd.DataFrame({'Date': future_dates, 'ForecastedDemand': future_preds.round().astype(int)})

                fig_forecast = px.line(forecast_df, x='Date', y='ForecastedDemand', title='Forecasted Demand for Next 30 Days', markers=True)
                fig_forecast.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Forecasted Demand (Units)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_forecast, use_container_width=True)

            st.markdown("---")

            # --- PDF Report Generation ---
            st.header("📄 Monthly Report")
            st.markdown("Generate a PDF summary of this month's performance.")
            if st.button("Generate PDF Report"):
                with st.spinner('Generating Report...'):
                    pdf_data = generate_pdf_report(kpis, top_oils_df, low_stock_df)
                    st.success('Report Generated!')
                    st.markdown(create_download_link_pdf(pdf_data, "monthly_report"), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        st.warning("Please ensure your file is a valid CSV or Excel file and the column names match the required format: Date, CustomerName, ProductName, Quantity, Price, CurrentStock.")

else:
    # --- Welcome Screen ---
    st.title("Welcome to the Cold-Pressed Oil Business Dashboard")
    st.markdown("This tool helps you analyze sales data, track key metrics, and plan for the future.")
    st.info("To get started, upload your sales data (CSV or Excel) using the sidebar on the left. If you don't have a file, you can generate and download a sample file to see how it works.")
    st.image("https://placehold.co/1200x400/2E3B4E/FFFFFF?text=Upload+Your+Data+to+Visualize+Insights&font=lato", use_column_width=True)
