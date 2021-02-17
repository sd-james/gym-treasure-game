



import pandas as pd


A = pd.read_pickle('/media/hdd/full_treasure_data/treasure_data/1/0/50/transition.pkl', compression='gzip')
B = pd.read_pickle('/media/hdd/full_treasure_data/treasure_data/1/2/50/transition.pkl', compression='gzip')
C = pd.read_pickle('/media/hdd/full_treasure_data/treasure_data/1/4/50/transition.pkl', compression='gzip')

D = A == B