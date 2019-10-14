#This module has tools for logging experiments

from os.path import isfile as isfile
from pandas import DataFrame as df
from pandas import read_csv

class Experiments:

    path = None
    experiments = None
    columns = [
                "Task",
                "Time",
                "Data",
                "Features",
                "Preprocessing",
                "Algorithm",
                "Parameters",
                "Evaluation",
                "F1",
                "AUC",
                "ER",
                "ACC",
                "Precision",
                "Recall",
                "Details"
                    ]
    template = df(index=[0], columns = columns).to_dict('records')[0]

    def __init__(self, path=None):
        if path is None:
            self.path = None
            self.experiments = None
        elif isfile(path):
            self.load(path)
        else:
            self.create(path)

    def load(self, path):
        f = open(path, "rb")
        self.path = path
        self.experiments = read_csv(f)
        f.close()
        return self
        

    def create(self,path):
        if isfile(path):
            raise IOError("File already exists, use load(path)")
        else:
            self.path = path
            self.experiments = df(columns=self.columns)
            self.experiments.to_csv(self.path, mode='a', header=True, index=False)
            return self

    def save_all(self):
        if self.path is None:
            raise EOFError("path to file does not exist, please set path first")
        elif self.experiments is None:
            raise EOFError("No Experiments data to save")
        else:
            self.experiments.to_csv(self.path, header=True, index=False)

    def save(self, experiment):
        if self.path is None:
            raise EOFError("path to file does not exist, please set path first")
        else:
            import datetime
            experiment['Time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            self.experiments = self.experiments.append(experiment, ignore_index=True)
            self.experiments.reindex()
            self.experiments[-1:].to_csv(self.path, mode='a', header=False, index=False)


    
