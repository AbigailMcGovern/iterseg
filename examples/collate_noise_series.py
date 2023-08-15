import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def load_and_concat(*args):
    suffix = '_scores.csv'
    files = []
    for data_dir in args:
        files = files + [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(suffix)]
    dfs = [pd.read_csv(f) for f in files]
    dfs = pd.concat(dfs).reset_index(drop=True)
    dfs['percent_noise'] = dfs['model_name'].apply(get_pcnt)
    dfs['model'] = dfs['model_name'].apply(get_name)
    return dfs


def get_pcnt(s):
    idx = s.find('_')
    s = s[idx + 1:]
    return float(s[:-1])


def get_name(s):
    idx = s.find('_')
    return s[:idx]


def plot_var(df, var):
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(x='percent_noise', y=var, data=df, hue='model', palette='Set2')
    ax.set_xlabel('Percent noise (%)')
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    sns.despine()
    fig.set_size_inches(4, 3)
    plt.show()


data = [
    '/Users/abigailmcgovern/Data/iterseg/invivo_platelets/DoG_plateseg_comp/noise_series_DoG', 
    '/Users/abigailmcgovern/Data/iterseg/invivo_platelets/DoG_plateseg_comp/noise_series_PS'
]
df = load_and_concat(*data)
df = df[df['percent_noise'] > 2]
#plot_var(df, 'Count difference')
# VI: GT | Output
#plot_var(df, 'VI: GT | Output')
plot_var(df, 'VI: Output | GT')


