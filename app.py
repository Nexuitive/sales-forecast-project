from flask import Flask, render_template, request, send_file
import os
import pandas as pd
from prophet import Prophet
import plotly.express as px
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# =========================
# HOME PAGE
# =========================

@app.route('/')
def home():
    return render_template('index.html')


# =========================
# MAIN FORECAST FUNCTION
# =========================

def generate_forecast(df):

    # AUTO DETECT COLUMNS

    date_column = None
    sales_column = None

    for col in df.columns:

        col_lower = col.lower()

        if 'date' in col_lower:
            date_column = col

        if (
            'sales' in col_lower
            or 'revenue' in col_lower
            or 'total' in col_lower
        ):
            sales_column = col

    if date_column is None or sales_column is None:
        return None

    # PREPARE DATA

    df = df[[date_column, sales_column]]

    df.columns = ['ds', 'y']

    df['ds'] = pd.to_datetime(df['ds'])

    df = df.dropna()

    # HOLIDAYS

    holidays = pd.DataFrame({

        'holiday': 'special_event',

        'ds': pd.to_datetime([
            '2025-01-01',
            '2025-12-25',
            '2025-11-28'
        ]),

        'lower_window': 0,

        'upper_window': 1

    })

    # MODEL

    model = Prophet(
        holidays=holidays
    )

    model.fit(df)

    # FUTURE

    future = model.make_future_dataframe(
        periods=30
    )

    forecast = model.predict(future)

    # RESULT TABLE

    result = forecast[['ds', 'yhat']].tail(30)

    result.columns = [
        'Forecast Date',
        'Predicted Sales'
    ]

    # SAVE CSV

    result.to_csv(
        'uploads/forecast.csv',
        index=False
    )

    # GRAPH

    fig = px.line(
        title="Retail Sales Forecasting Dashboard"
    )

    fig.add_scatter(
        x=df['ds'],
        y=df['y'],
        mode='lines',
        name='Actual Sales'
    )

    fig.add_scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Predicted Sales'
    )

    fig.add_scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(
            color='rgba(0,0,255,0.2)'
        )
    )

    fig.add_scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        name='Prediction Range',
        fill='tonexty',
        fillcolor='rgba(0,123,255,0.2)',
        line=dict(
            color='rgba(0,0,255,0.2)'
        )
    )

    graph_html = fig.to_html(

        full_html=False,

        config={
            'displaylogo': False
        }

    )

    # COMPONENT GRAPH

    components_fig = model.plot_components(
        forecast
    )

    components_fig.savefig(
        "static/components.png"
    )

    plt.close()

    # INSIGHTS

    avg_sales = round(
        result['Predicted Sales'].mean(),
        2
    )

    max_sales = round(
        result['Predicted Sales'].max(),
        2
    )

    min_sales = round(
        result['Predicted Sales'].min(),
        2
    )

    # TREND

    if (
        result['Predicted Sales'].iloc[-1]
        >
        result['Predicted Sales'].iloc[0]
    ):

        trend = (
            "Sales are expected to increase in the coming period."
        )

    else:

        trend = (
            "Sales are expected to decrease in the coming period."
        )

    # DEMAND

    if avg_sales > df['y'].mean():

        demand = (
            "Higher demand is predicted compared to historical data."
        )

    else:

        demand = (
            "Demand is expected to remain stable or lower."
        )

    insight = trend + " " + demand

    return {
        "graph_html": graph_html,
        "table_html": result.to_html(index=False),
        "avg_sales": avg_sales,
        "max_sales": max_sales,
        "min_sales": min_sales,
        "insight": insight
    }


# =========================
# INVENTORY FUNCTION
# =========================

def get_inventory():

    inventory = [

        {
            "product": "Product A",
            "stock": 35
        },

        {
            "product": "Product B",
            "stock": 90
        },

        {
            "product": "Product C",
            "stock": 240
        }

    ]

    for item in inventory:

        if item["stock"] < 50:

            item["status"] = "Low Stock"
            item["color"] = "red"

        elif item["stock"] < 100:

            item["status"] = "Warning"
            item["color"] = "orange"

        else:

            item["status"] = "Healthy"
            item["color"] = "black"

    return inventory


# =========================
# FILE UPLOAD
# =========================

@app.route('/upload', methods=['POST'])
def upload_file():

    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    file_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )

    file.save(file_path)

    app.config['LAST_FILE'] = file_path

    # READ FILE

    if file.filename.endswith('.csv'):

        try:
            df = pd.read_csv(
                file_path,
                encoding='utf-8'
            )

        except:
            df = pd.read_csv(
                file_path,
                encoding='latin1'
            )

    elif file.filename.endswith('.xlsx'):

        df = pd.read_excel(file_path)

    else:

        return "Unsupported file format"

    forecast_data = generate_forecast(df)

    if forecast_data is None:
        return "Could not detect Date or Sales columns"

    return render_template(

        'result.html',

        graph_html=forecast_data['graph_html'],

        table_html=forecast_data['table_html'],

        avg_sales=forecast_data['avg_sales'],

        max_sales=forecast_data['max_sales'],

        min_sales=forecast_data['min_sales'],

        insight=forecast_data['insight'],

        inventory=get_inventory(),

        selected_product="All Products"

    )


# =========================
# DATE FILTER
# =========================

@app.route('/filter', methods=['POST'])
def filter_data():

    file_path = app.config.get('LAST_FILE')

    if not file_path:
        return "No uploaded file found"

    # READ FILE

    if file_path.endswith('.csv'):

        try:
            df = pd.read_csv(
                file_path,
                encoding='utf-8'
            )

        except:
            df = pd.read_csv(
                file_path,
                encoding='latin1'
            )

    else:

        df = pd.read_excel(file_path)

    # FILTER DATES

    date_column = None

    for col in df.columns:

        if 'date' in col.lower():
            date_column = col

    start_date = request.form.get('start_date')

    end_date = request.form.get('end_date')

    if start_date and end_date:

        df[date_column] = pd.to_datetime(
            df[date_column]
        )

        df = df[
            (df[date_column] >= start_date)
            &
            (df[date_column] <= end_date)
        ]

    forecast_data = generate_forecast(df)

    return render_template(

        'result.html',

        graph_html=forecast_data['graph_html'],

        table_html=forecast_data['table_html'],

        avg_sales=forecast_data['avg_sales'],

        max_sales=forecast_data['max_sales'],

        min_sales=forecast_data['min_sales'],

        insight=forecast_data['insight'],

        inventory=get_inventory(),

        selected_product="All Products"

    )


# =========================
# PRODUCT FILTER
# =========================

@app.route('/filter-product', methods=['POST'])



# =========================
# DOWNLOAD CSV
# =========================

@app.route('/download')
def download_file():

    return send_file(

        'uploads/forecast.csv',

        as_attachment=True

    )


# =========================
# RUN APP
# =========================

if __name__ == '__main__':
    app.run()