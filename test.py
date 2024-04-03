import graphMaker
import pandas as pd
data1 = [True, False, False, True, False, True, True]
column1 = "X"
data2 = [10, 5, 1, 5, 6, 25, 10]
column2 = "Y"
data3 = [True, True, False, True, False, True, False]
column3 = "Z"
data = pd.DataFrame({'X': data1, 'Y': data2, 'Z': data3})
graphMaker.tableau_boolean(data, [column1, column3], [column2])