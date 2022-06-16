import os
import subprocess
import pandas as pd
import sqlalchemy as db
from pathlib import Path

#%% Set Credntials

#psql_credentials = 


#%%

class Database:

    _instance = None

    @classmethod
    def get_instance(cls) -> None:
        if Database._instance is None:
            _instance = Database(cls)
        return _instance

    def __init__(self, cls):
        if cls == Database:
            credentials = {
                'host': 'localhost',
                'dbname': 'tfwm',
                'user': 'postgres',
                'password': 'admin',
                'port': 5432}
            db_url = db.engine.url.URL(
                drivername='postgresql',
                username=credentials['user'], 
                password=credentials['password'],
                host=credentials['host'],
                database=credentials['dbname'],
                port=credentials['port']
            )
            self._credentials = credentials
            self.engine = db.create_engine(db_url, echo=False)
        else:
            raise AssertionError(
                    "Database can only be created using get_instance")
            
            
    def execute_sql(self, string, read_file, return_df=False, chunksize=None, params=None):
        """
        Executes a SQL query from a file or a string using SQLAlchemy engine
        Note: Must only be basic SQL (e.g. does not run PSQL \copy and other commands)
        Note: SQL file CANNOT START WITH A COMMENT! There can be comments later on in the file, but for some reason
        doesn't work if you start with one (seems to treat the entire file as commented)

        Parameters
        ----------
        string : string
            Either a filename (with full path string '.../.../.sql') or a specific query string to be executed
            Can include "parameters" (in the form of {param_name}) whose values are filled in at the time of execution
        read_file : boolean
            Whether to treat the string as a filename or a query
        print_ : boolean
            Whether to print the 'Executed query' statement
        return_df : boolean
            Whether to return the result table of query as a Pandas dataframe
        chunksize : int
            Rows will be read in batches of this size at a time; all rows will be read at once if not specified
        params : dict
            In the case of parameterized SQL, the dictionary of parameters in the form of {'param_name': param_value}

        Returns
        -------
        ResultProxy : ResultProxy
            see SQLAlchemy documentation; results of query
        """
        if read_file:
            query = Path(string).read_text()
        else:
            query = string

        if params is not None:
            query = query.format(**params)

        if return_df:
            res_df = pd.read_sql_query(query, self.engine, chunksize=chunksize)
            return res_df
        else:  # Not all result objects return rows.
            self.engine.execute(query)

    def copy_table_to_csv(self, sqlQuery: str, dst_file: str):
        conn = self.engine.raw_connection()
        try:
            cursor = conn.cursor()
            copy_statement = "COPY ({}) TO STDOUT WITH CSV HEADER".format(sqlQuery)
            with open(dst_file, 'w') as csv_file:
                cursor.copy_expert(copy_statement, csv_file)
            cursor.close()
        finally:
            conn.close()