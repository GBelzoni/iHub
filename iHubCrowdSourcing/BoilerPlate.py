import sys
import os

import pandas as pd
import numpy as np
from pandas.io.data import DataReader
from datetime import datetime
import sqlite3 
import pandas.io.sql as psql
import PSQLUtils as pu
import time
import pickle

os.getcwd()
os.chdir('/home/phcostello/Documents/workspace/iHubCrowdSourcing')
dbpath = "/home/phcostello/Documents/Data/iHub/S3_RawData/"
dbfile = "CrowdSourcingData.sqlite"
