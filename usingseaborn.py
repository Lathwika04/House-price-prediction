import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt




dataset=pd.read_csv("HousingData.csv")
plotnumber=1
for column in dataset:
    if(plotnumber<=14):
        plt.subplot(3,5,plotnumber)
        sb.distplot(dataset[column])
        
    plotnumber+=1
plt.tight_layout()
plt.show()
