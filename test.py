import graphMaker
import pandas as pd
data1 = ["W", "V", "W", "W", "X", "X", "A"]
column1 = "X"
data2 = [10, 5, 1, 5, 6, 25, 10]
column2 = "Y"
data = pd.DataFrame({'X': data1, 'Y': data2})
graphMaker.graph_double_maker(data, column1, column2)