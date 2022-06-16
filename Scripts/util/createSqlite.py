import sqlite3
from Scripts.util.database import Database
database = Database.get_instance()

#%%



#%%
def dataPGtoSqlite(pgtable, sltable):
    dataFromPG = database.execute_sql('SELECT * FROM {}'.format(pgtable), False, True)
    dataFromPG.to_sql(name = sltable, con = cnx)

#%%

cnx = sqlite3.connect('Data/access.db')
dataPGtoSqlite('results_may2022.results_summary', 'results_summary')
print('Results Summary Loaded')
dataPGtoSqlite('semantic.oa', 'oa')
print('OA Loaded')
dataPGtoSqlite('semantic.poi', 'poi')
print('POI Loaded')


#%%
RESULTS_FOLDER = 'Data/otp_trips.csv'
query = 'select a.oa_id, a.poi_id, a.trip_id, b.stratum from model_may2022.otp_trips as a left join model_may2022.trips as b on a.trip_id = b.trip_id'

database.copy_table_to_csv(
    query,
    RESULTS_FOLDER,
)

#%%

RESULTS_FOLDER = 'Data/results_full.csv'
query = 'select total_time, initial_wait_time - 3600 as initial_wait_corrected, transit_time, fare, num_transfers, trip_id from results_may2022.results_full'

database.copy_table_to_csv(
    query,
    RESULTS_FOLDER,
)