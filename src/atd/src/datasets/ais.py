import os

import networkx as nx
import numpy as np
import pandas as pd
from comptopo.filtrations import edge_weight_persistence_diagrams
from dlex import Params, logger
from dlex.datasets import DatasetBuilder
from dlex.datasets.sklearn import SklearnDataset
from tqdm import tqdm


class AIS(DatasetBuilder):
    def __init__(self, params: Params):
        username = "atd"
        password = "never-wonder-about-oranges"
        super().__init__(params, [(url, username, password) for url in [
            "https://grassmann.math.colostate.edu/ATD/AIS/AIS_Challenge_Problem_Set_0_1.csv",
            "https://grassmann.math.colostate.edu/ATD/AIS/AIS_Challenge_Problem_Set_0_2.csv",
            "https://grassmann.math.colostate.edu/ATD/AIS/AIS_Challenge_Problem_Set_0_3.csv",
            "https://grassmann.math.colostate.edu/ATD/AIS/AIS_Challenge_Problem_Set_0_1_withVID.csv",
            "https://grassmann.math.colostate.edu/ATD/AIS/AIS_Challenge_Problem_Set_0_2_withVID.csv",
            "https://grassmann.math.colostate.edu/ATD/AIS/AIS_Challenge_Problem_Set_0_3_withVID.csv"
        ]])

    def get_sklearn_wrapper(self, mode: str):
        return SklearnAIS(self)


class SklearnAIS(SklearnDataset):
    def __init__(self, builder: AIS):
        super().__init__(builder)

        df = None
        for i in range(1, 4):
            fn = os.path.join(builder.get_working_dir(), f"AIS_Challenge_Problem_Set_0_{i}_withVID.csv")
            if df is None:
                df = pd.read_csv(fn)
            else:
                df += pd.read_csv(fn)

        df = df[df['OBJECT_ID'] > 0]
        df['OBJECT_ID'] = df['OBJECT_ID'].astype('int')
        df['VID'] = df['VID'].astype('int')
        print(df.head())
        logger.info(f"No. samples: {len(df)}")
        logger.info(f"No. tracks: {len(df['VID'].unique())}")

        track_ids = df['VID'].unique()
        train_track_ids = track_ids[10:]
        test_track_ids = track_ids[:10]

        df_train = df[df['VID'].isin(train_track_ids)]
        df_test = df[df['VID'].isin(test_track_ids)]
        logger.info(f"Train size: {len(df_train)}; Test size: {len(df_test)}")

        def _build_X(_df: pd.DataFrame):
            if self.configs.features == "raw":
                return list(zip(
                    _df['LAT'].tolist(),
                    _df['LON'].tolist(),
                    [s for s in _df['SPEED_OVER_GROUND'].tolist()],
                    [c / 10. for c in _df['COURSE_OVER_GROUND'].tolist()]
                ))
            else:
                dgms = []

                for u in tqdm(range(len(_df)), desc="Calculating PDs"):
                    g = nx.DiGraph()
                    g.add_nodes_from(range(len(_df)))

                    for v in range(len(_df)):
                        if u == v:
                            continue

                        pt_u = np.array([_df.iloc[u]['LAT'], _df.iloc[u]['LON']])
                        pt_v = np.array([_df.iloc[v]['LAT'], _df.iloc[v]['LON']])
                        speed = _df.iloc[u]['SPEED_OVER_GROUND']
                        angle = _df.iloc[u]['COURSE_OVER_GROUND'] / 3600. * (2 * np.pi)
                        g.add_edge(u, v, weight=np.linalg.norm(pt_u - pt_v))

                    dgms.append(edge_weight_persistence_diagrams(g, tool='gudhi'))
                return dgms

        self.X_train = _build_X(df_train)
        self.Y_train = df_train['VID'].tolist()
        self.X_test = _build_X(df_test)
        self.Y_test = df_test['VID'].tolist()
