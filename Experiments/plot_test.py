import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from pylab import savefig

def Plot_confusion_matrix(pred, true, labels, name):
    c_m_array = confusion_matrix(pred, true)
    df_c_m = pd.DataFrame(c_m_array, index = labels, columns = labels)
    heatmap = sn.heatmap(df_c_m, annot=True, cmap='coolwarm', linecolor='white', linewidths=1)
    figure = heatmap.get_figure()
    figure.savefig('XLNet_%s.png' % name, dpi=400)

polarity_pred = [1, 1, 1, 1, 1, 1, 1]
polarity_true = [0, 2, 1, 0, 2, 1, 1]
Plot_confusion_matrix(polarity_pred, polarity_true, ["POSITIVE", "NEUTRAL", "NEGATIVE"], 'polarity')
#Plot_confusion_matrix(factuality_pred, factuality_true, ["EXPERIENCE", "OPINION", "FACT"], 'factuality')
     
