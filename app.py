from flask import Flask, render_template, request
import os
import pandas as pd
from prophet import Prophet
import plotly.express as px

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():

    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    file.save(file_path)

    # Read CSV
    df = pd.read_csv(file_path)

    # Prophet Format
    df.columns = ['ds', 'y']

    # Prophet Model
    model = Prophet()

    model.fit(df)

    # Future Dates
    future = model.make_future_dataframe(periods=30)

    # Prediction
    forecast = model.predict(future)

    # Forecast Table
    result = forecast[['ds', 'yhat']].tail(30)

    # Graph
    fig = px.line(title="Sales Forecast (Actual vs Predicted)")

    # Actual Data
    fig.add_scatter(
    x=df['ds'],
    y=df['y'],
    mode='lines',
    name='Actual Sales')

    # Predicted Data
    fig.add_scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    mode='lines',
    name='Predicted Sales')

    # Upper Bound (Best Case)
    fig.add_scatter(
    x=forecast['ds'],
    y=forecast['yhat_upper'],
    mode='lines',
    name='Upper Bound',
    line=dict(dash='dot'))

    # Lower Bound (Worst Case)
    fig.add_scatter(
    x=forecast['ds'],
    y=forecast['yhat_lower'],
    mode='lines',
    name='Lower Bound',
    line=dict(dash='dot'))

    graph_html = fig.to_html(full_html=False)

    # Insights
    avg_sales = round(result['yhat'].mean(),2)
    max_sales = round(result['yhat'].max(),2)
    min_sales = round(result['yhat'].min(),2)
    
    # Trend Insight
    if result['yhat'].iloc[-1] > result['yhat'].iloc[0]:
    trend = "Sales are expected to increase in the coming period."
    else:
    trend = "Sales are expected to decrease in the coming period."

    # Demand Insight
    if avg_sales > df['y'].mean():
    demand = "Higher demand is predicted compared to past sales."
    else:
    demand = "Demand is expected to remain stable or slightly lower."

    # Final Insight
    insight = trend + " " + demand

    # Table HTML
    table_html = result.to_html(index=False)

    return render_template(
        'result.html',
        graph_html=graph_html,
        table_html=table_html,
        avg_sales=avg_sales,
        max_sales=max_sales,
        min_sales=min_sales
        insight=insight
    )


if __name__ == '__main__':
    app.run(debug=True)