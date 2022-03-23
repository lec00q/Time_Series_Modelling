
import streamlit as st
from darts import TimeSeries
import pandas as pd
import numpy as np
import plotly.express as px
import forecast as fc
from darts.utils.utils import SeasonalityMode
from darts.dataprocessing.transformers import Scaler
from darts.models import ExponentialSmoothing, AutoARIMA, Prophet, Theta
from darts.models import TFTModel, NBEATSModel
from darts.metrics import mae


st.title('ConnectHub Shipments Forecast')

DATA_URL = '_Final_master_df_2022_2017_v9_with_Bolzano.csv'
PRE_COVID = pd.to_datetime('2020-03-02')
POST_COVID = pd.to_datetime('2020-05-25')


uploaded = st.file_uploader('Upload a new data file', type='csv', accept_multiple_files=False)

if uploaded:
    DATA_URL = uploaded

df = fc.load_dataframe(DATA_URL)

series_selected = st.selectbox(
     'Which time series you would like to forecast?',
     df.columns)

baseline_selected = st.selectbox(
     'Which time series you would like to compare with?',
     df.columns)

# TODO: infer frequency
ts = TimeSeries.from_dataframe(df, value_cols=series_selected, freq='W-MON')
ts = ts.astype(np.float32)

pred = TimeSeries.from_dataframe(df, value_cols=baseline_selected, freq='W-MON')
pred = pred.astype(np.float32)

start = pred.strip().start_time()
# 'Bolzano prediction at', start

# start = pd.to_datetime(st.date_input('Pick a different cutoff date', value=start, min_value=ts.start_time(),
                                         # max_value=ts.end_time()))
'Start prediction at', start

# Cutoff ts before covid
# ts = ts.drop_after(PRE_COVID)

models = {
    'Prophet': Prophet(country_holidays='IT'),
    'Arima': AutoARIMA(),
    'Exponential': ExponentialSmoothing(seasonal_periods=52),
    'Theta': Theta(seasonality_period=52, season_mode=SeasonalityMode.ADDITIVE),
    'TFT': TFTModel(input_chunk_length=12, output_chunk_length=12,
        add_encoders={
            # 'cyclic': {'future': ['month']},
            'datetime_attribute': {"past": ["weekofyear"], "future": ["weekofyear"]},
            # 'position': {'past': ['absolute'], 'future': ['relative']},
            # 'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
            # 'transformer': Scaler()
        }),
    'N-BEATS': NBEATSModel(input_chunk_length=12, output_chunk_length=12,
        add_encoders={
            'cyclic': {'future': ['month']},
            'datetime_attribute': {"past": ["weekofyear"]},
            # 'position': {'past': ['absolute'], 'future': ['relative']},
            # 'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
            # 'transformer': Scaler()
        }
    ),
}
model_selected = st.selectbox(
     'Which forecasting model you would like to try?',
     models.keys(), index=2)

if st.button("Train and test"):
    scaler = Scaler()
    ts_scale = scaler.fit_transform(ts)

    train, val = ts_scale.split_before(start)
    model = models[model_selected]
    with st.spinner():
        fore = fc.eval_model(model, ts_scale, val, start)
        # fore = fc.make_forecast(model, train)
        fore = scaler.inverse_transform(fore)
        val = scaler.inverse_transform(val)
        yhat = f'{series_selected} forecast'
        df = pd.concat([df, fore.pd_dataframe().rename(columns={series_selected: yhat})], axis=1)
        error = mae(val, fore)
else:
    yhat = series_selected
    error = None

fig = px.line(df, y=[series_selected, baseline_selected, yhat], title=f'{series_selected}')
st.plotly_chart(fig, use_container_width=True)

if error:
    st.write("### Mean Absolute Error")
    st.write("Forecasted ", error)
    st.write("Bolzano ", mae(val, pred))

    # This SHOULD be similar to the one above, excluding the fact
    # that it is averaged across each period.
    # historical_fcast = model.backtest(
    #     ts, start=start, forecast_horizon=12, stride=12, verbose=True,
    #     metric=mae
    # )
    # st.write(historical_fcast)
