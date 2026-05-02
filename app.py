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
    fig = px.line(
        forecast,
        x='ds',
        y='yhat',
        title='Future Sales Forecast'
    )

    graph_html = fig.to_html(full_html=False)

    # Insights
    avg_sales = round(result['yhat'].mean(),2)
    max_sales = round(result['yhat'].max(),2)
    min_sales = round(result['yhat'].min(),2)

    # Table HTML
    table_html = result.to_html(index=False)

    return render_template(
        'result.html',
        graph_html=graph_html,
        table_html=table_html,
        avg_sales=avg_sales,
        max_sales=max_sales,
        min_sales=min_sales
    )


if __name__ == '__main__':
    app.run(debug=True)