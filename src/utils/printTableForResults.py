import os

import pandas as pd
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--results')
args = argparser.parse_args()

exp = os.listdir(args.results)
exp.sort()

main_df = None
for e in exp:
    e_path = os.path.join(args.results,e,'pedestrian_summary.txt')
    df = pd.read_csv(str(e_path),sep=' ')
    df.index = [e]
    if main_df is None:
        main_df = df
    else:
        main_df = pd.concat([main_df,df])

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(main_df.loc[:,['IDP','IDR','DetA','AssA','IDF1','HOTA']])
