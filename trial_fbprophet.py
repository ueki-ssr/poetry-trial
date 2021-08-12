### 参考記事
# https://touch-sp.hatenablog.com/entry/2020/09/16/131122#:~:text=4.49.0%0Aurllib3%3D%3D1.25.10-,%E3%82%B5%E3%83%B3%E3%83%97%E3%83%AB,-%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%97%E3%83%88
###
import pandas as pd
from fbprophet import Prophet

import requests
import io

url = 'https://raw.githubusercontent.com/facebook/prophet/master/examples/example_wp_log_peyton_manning.csv'
response = requests.get(url)
df = pd.read_csv(io.BytesIO(response.content),sep=",")

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

from fbprophet.plot import plot_plotly
import plotly.offline as py

fig = plot_plotly(m, forecast)
py.plot(fig)