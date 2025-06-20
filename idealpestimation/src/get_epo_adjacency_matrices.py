import os 

os.environ["OMP_NUM_THREADS"] = "48"
os.environ["MKL_NUM_THREADS"] = "48"
os.environ["OPENBLAS_NUM_THREADS"] = "48"
os.environ["NUMBA_NUM_THREADS"] = "48"

import sys
import ipdb
import pathlib
import jsonlines
import numpy as np
import random
from idealpestimation.src.utils import pickle, connect_to_epo_db, check_sqlite_database, get_table_columns, \
                                    get_row_count, execute_create_sql, build_sparse_adjacency_matrix, print_matrix_info, save_matrix


if __name__ == "__main__":

    DIR_out = "/mnt/hdd2/ioannischalkiadakis/epodata_rsspaper/"
    pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)     

    database_path = "/mnt/hdd2/epodata/stage/20250416/anonymized_reproducibility/"
    # database_path = "/mnt/hdd2/ioannischalkiadakis/"
    databases = ["france_2020_anonymized_reproducibility.db", "france_2023_anonymized_reproducibility.db",
                "germany_2020_anonymized_reproducibility.db", "germany_2023_anonymized_reproducibility.db",
                "poland_2020_anonymized_reproducibility.db", "poland_2023_anonymized_reproducibility.db",
                "netherlands_2020_anonymized_reproducibility.db", "netherlands_2023_anonymized_reproducibility.db",
                "uk_2020_anonymized_reproducibility.db", "uk_2023_anonymized_reproducibility.db",
                "finland_2020_anonymized_reproducibility.db", "finland_2023_anonymized_reproducibility.db",
                "us_2023_anonymized_reproducibility.db"]
    # databases = ["netherlands_2023_anonymized_reproducibility.db"]
    
    for database in databases:
    
        dbconn = connect_to_epo_db("{}/{}".format(database_path, database))
        # check_sqlite_database(dbconn, "{}/{}".format(database_path, database), sample_rows=3)
        columns = get_table_columns(dbconn, table_name="mp_follower_graph")
        print(columns)
        
        adjacency_matrix, node_to_index_start, index_to_node_start, \
            node_to_index_end, index_to_node_end = build_sparse_adjacency_matrix(dbconn, table_name="mp_follower_graph", 
                                                                                start_col='follower_pseudo_id', end_col='mp_pseudo_id', 
                                                                                weighted=False, weight_col=None, directed=True)

        print_matrix_info(adjacency_matrix)
        nameparts = database.split("_")
        filename = "{}/Y_{}_{}".format(DIR_out, nameparts[0], nameparts[1])
        save_matrix(adjacency_matrix, node_to_index_start, index_to_node_start, node_to_index_end, index_to_node_end, filename)
