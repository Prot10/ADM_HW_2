# from functions import Dataset, get_types, optimize_cols, downcast_int, line_count, cool_ticks, clock_graph, EMOJIS, safe_detect
import pandas as pd
import numpy as np
import random
import datatable as dt # probably needed !pip install datatable
import re
import os
import matplotlib.pyplot as plt
from math import pi, floor, log10, ceil

class Dataset:
    def __init__(self, file:str):
        self.file_name : str = file
        self.types : dict = self.load_types()
        self.df : pd.DataFrame = None
        
    def name(self):
        return re.search(r'\/([\w\d]*)\.csv',self.file_name).group(1)
    
    def __types_file__(self):
        return 'types/' + self.name() +'.npy'
        
    def save_types(self):
        with open(self.__types_file__(), 'wb') as f:
            np.save(f, self.types)
                    
    def load_types(self):
        if os.path.isfile(self.__types_file__()):
            self.types = np.load(self.__types_file__(),allow_pickle='TRUE').item()
            return self.types
        return None

    def __len__(self):
        return dt.fread(self.file_name,
                           columns={'sid'},
                           sep='\t').shape[0]
    
    def col(self,columns:list, index=True,pandas=False,**dt_params)->pd.DataFrame:
        """Loads some columns of the dataframe out of the whole csv file
        
        :param columns: a list of the desired columns
        :type columns: list
        :param index: if True loads also the 'sid' column as the index
        :type index: bool
        :param **pd_params: any other params for pd.read_csv(...)
        :rtype: pd.DataFrame
        """
        col_list = columns
        if index==True:
            col_list.append('sid')
        if pandas:
            return pd.read_csv(self.file_name,
                           usecols=col_list,
                           dtype=self.types, 
                           index_col='sid' if index==True else index,
                           delimiter='\t',
                           **dt_params)
        d = dt.fread(self.file_name,
                           columns=set(col_list),
                           sep='\t',
                            **dt_params)
        if index == True:
            d.key = 'sid'
        elif index:
            d.key = index
        d = d.to_pandas().astype({k:v for k,v in self.types.items() if k in d.names})
        if 'cts' in col_list:
            d['cts'] = pd.to_datetime(d['cts'])
        return d

def get_types(signed=True, unsigned=True, custom=[]):
    '''Returns a pandas dataframe containing the boundaries of each integer dtype'''
    # based on https://stackoverflow.com/a/57894540/9419492
    pd_types = custom
    if signed:
        pd_types += [pd.Int8Dtype() ,pd.Int16Dtype() ,pd.Int32Dtype(), pd.Int64Dtype()]
    if unsigned:
        pd_types += [pd.UInt8Dtype() ,pd.UInt16Dtype(), pd.UInt32Dtype(), pd.UInt64Dtype()]
    type_df = pd.DataFrame(data=pd_types, columns=['pd_type'])
    type_df['np_type'] = type_df['pd_type'].apply(lambda t: t.numpy_dtype)
    type_df['min_value'] = type_df['np_type'].apply(lambda row: np.iinfo(row).min)
    type_df['max_value'] = type_df['np_type'].apply(lambda row: np.iinfo(row).max)
    type_df['allow_negatives'] = type_df['min_value'] < 0
    type_df['size'] = type_df['np_type'].apply(lambda row: row.itemsize)
    type_df.sort_values(by=['size', 'allow_negatives'], inplace=True)
    return type_df.reset_index(drop=True)
def downcast_int(file_path, column:str, chunksize=10000, delimiter=',', signed=True, unsigned=True):
    '''Assigns the smallest possible dtype to an integer column of a csv'''
    types = get_types(signed, unsigned)
    negatives = False
    print(delimiter)
    for chunk in pd.read_csv(file_path, 
                             usecols=[column],
                             delimiter=delimiter,
                             chunksize=chunksize):
        M = chunk[column].max()
        m = chunk[column].min()
        if not signed and not negatives and m < 0 :
            types = types[types['allow_negatives']] # removes unsigned rows
            negatives = True
        if m < types['min_value'].iloc[0]:
            types = types[types['min_value'] < m]
        if M > types['max_value'].iloc[0]:
            types = types[types['max_value'] > M]
        if len(types) == 1:
            print('early stop')
            break
    return types['pd_type'].iloc[0]

def optimize_cols(file, int_cols, delimiter=',', signed=True, unsigned=True):
    out = dict()
    for col in int_cols:
        out[col] = downcast_int(file, col, delimiter=delimiter, signed=signed, unsigned=unsigned)
    return out

def line_count(filename):
    with open(filename) as f:
        print(len(f.readlines()))

def cool_ticks(value, tick_number=None):
    '''based on https://stackoverflow.com/a/59973033/9419492'''
    num_thousands = 0 if abs(value) < 1000 else floor(log10(abs(value))/3)
    value = round(value / 1000**num_thousands)
    if value >= 1000:
        value /= 1000
        num_thousands += 1
    return f'{value:g}'+' KMGTPEZY'[num_thousands], value * 1000**num_thousands

def clock_graph(hours: pd.Series, labels:list=None):
    '''plots a Series in a Radar Chart.
        :hours: a pandas series where each row index will become an angle
    '''
    plt.figure(figsize=(7,7))

    # https://www.python-graph-gallery.com/390-basic-radar-chart
    N = len(hours)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    angles = angles[::-1]
    hours = pd.concat([
        hours.iloc[N//4:],
        hours.iloc[:N//4] ]) if N==24 else hours.sort_index(ascending=True)
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # Draw one axe per variable + add labels
    labels = labels if labels is not None else hours.index.map(lambda h: f"{int(h)}:00")
    plt.xticks(angles[:-1], labels, color='black', size=15)

    # Draw ylabels
    ax.set_rlabel_position(10)
    M = hours.max()
    ticks = [cool_ticks(M*x/5) for x in range(1,5)]
    plt.yticks([v[1] for v in ticks], [t[0] for t in ticks], color="black", size=10)
    plt.ylim(0,M*21/20)
    
    # Plot data
    ax.plot(angles, [*hours, hours.iloc[0]], linewidth=2, linestyle='solid')
    hour_max = hours.index.get_loc(hours.idxmax()) if N!=24 else hours.idxmax()-N//4
    ax.plot(angles[hour_max],M, 'bo', label=f"Max: {M}")

    # Fill area
    ax.fill(angles[:-1], hours, 'b', alpha=0.3)

EMOJIS = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "#@%"
                               "0-9"
                                "]+", flags=re.UNICODE)


def safe_detect(s, minsize=6):
    try:
        return detect(s)
    except:
        return pd.NA   

