import pandas as pd
import stats

data_train = pd.read_csv('train.csv')   # 加载数据
data_test = pd.read_csv('test.csv')
data_test_results = pd.read_csv('sample_submission.csv')

train_num = data_train.shape[0]
y_train = data_train.SalePrice
y_train = stats.boxcox(y_train, 0.5)

all_data = pd.concat((data_train, data_test)).reset_index(drop=True)
all_data.drop(['SalePrice', 'Id'], axis=1, inplace=True)
