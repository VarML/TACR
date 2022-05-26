import pandas as pd
from stock_env.apps import config
from preprocessor.yahoodownloader import YahooDownloader
from preprocessor.preprocessors import FeatureEngineer, data_split
import itertools

df = YahooDownloader(start_date = '2009-01-01',
                      end_date = '2020-09-24',
                     ticker_list = config.Dow_TICKER).fetch_data()

# df = YahooDownloader(start_date = '2006-10-20',
#                      end_date = '2013-11-21',
#                      ticker_list = config.HighTech_TICKER).fetch_data()

#df = YahooDownloader(start_date = '2009-01-01',
#                     end_date = '2021-12-31',
#                     ticker_list = config.SP_TICKER).fetch_data()


df.sort_values(['date','tic'],ignore_index=True).head()

fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                    use_turbulence=True
)

processed = fe.preprocess_data(df)

list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
combination = list(itertools.product(list_date,list_ticker))

processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date','tic'])
processed_full = processed_full.fillna(0)

processed_full.sort_values(['date','tic'],ignore_index=True).head(10)

train = data_split(processed_full, '2009-01-01','2019-01-01')
trade = data_split(processed_full, '2019-01-01','2020-09-24')

# train = data_split(processed_full, '2006-10-20','2012-11-16')
# trade = data_split(processed_full, '2012-11-16','2013-11-21')

#train = data_split(processed_full, '2009-01-01','2020-09-01')
#trade = data_split(processed_full, '2020-09-01','2021-12-31')

train.to_csv("datasets/train.csv")
trade.to_csv("datasets/trade.csv")