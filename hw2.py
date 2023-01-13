import numpy as np
import pandas as pd
import scikitplot as skplt
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import confusion_matrix
from statsmodels.distributions.empirical_distribution import ECDF
from scikitplot.metrics import plot_roc, plot_ks_statistic, plot_cumulative_gain
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import multilabel_confusion_matrix


class performance_measure:
    def predict_y(self, prob_of_class1, classes, thres=0.5): 
        predicted_y_list=[]
        for i in prob_of_class1:
            if i>thres:
                pred_class=1
            else:
                pred_class=0
            cls=classes[pred_class]
            predicted_y_list.append(cls)
        return predicted_y_list
    
    def confusion_based(self, target_y, predicted_y, class1_label):
        labels=class1_label
        cm=confusion_matrix(target_y, predicted_y)
        tn=cm[0][0]
        fp=cm[0][1]
        fn=cm[1][0]
        tp=cm[1][1]
        tpr=tp/(tp+fn)
        fpr=fp/(tn+fp)
        tnr=tn/(tn+fp)
        fnr=fn/(tp+fn)
        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        F1measure=2*(precision*recall)/(precision+recall)
        result = {'confusion matrix:':cm, \
                    'TPR':tpr, 'TNR':tnr, 'FPR':fpr, 'FNR':fnr,
                    'precision':precision, 'recall':recall, 'F1':F1measure}
        return result 
        
         
    def ave_class_accuracy(self, target_y, predicted_y):
        cr=metrics.classification_report(target_y, predicted_y, output_dict=True)
        n=len(cr.items())-3
        keys_list=list(cr)
        recall=[]
        for i in range(0,n):
            keys=keys_list[i]
            rec = list(cr[keys].values())[1]
            recall.append(rec)
        arithmatic=sum(recall)/n
        harmonic=n/sum(1/recall[k] for k in range(0,n))
        return { 'classification report':cr, \
                 'ave':arithmatic, 'HM':harmonic}
    
    def roc(self,  target_y, predicted_prob_class1, class1_label):   
        fpr, tpr, thres = metrics.roc_curve(target_y, predicted_prob_class1, pos_label=class1_label)
        auc=metrics.auc(fpr, tpr)
        roc_index=sum([(fpr[i]-fpr[i-1])*(tpr[i]+tpr[i-1] ) for i in range(2,len(thres))])/2
        gini=2*roc_index-1

        return {'fpr':fpr, 'tpr':tpr, 'AUC':auc, 'ROC_index':roc_index, 'Gini_coef':gini}

    def KS(self, target_y, predicted_prob_class1):  
        df = pd.DataFrame(data=dict(predicted_prob_class1=predicted_prob_class1))
        df['target_y']=target_y
        classes=df.target_y.unique()
        class0 = df[target_y == classes[0]]
        class1 = df[target_y == classes[1]]
        KS_stat=ks_2samp(class0['predicted_prob_class1'], class1['predicted_prob_class1'])[0]
        skplt.metrics.plot_ks_statistic(target_y, predicted_prob_class1)
        plt.show()
        return KS_stat
 
    def cum_gain(self, target_y, predicted_prob_class1, target_percent, class1_label):  
        df=pd.DataFrame(data=dict(predicted_prob_class1=predicted_prob_class1))
        df['target_y']=target_y
        sorted_df=df.sort_values(by=['predicted_prob_class1'], ascending=False)
        gain=[]
        count_value=sorted_df.target_y.value_counts()[class1_label]
        expected=sorted_df.target_y
        for i in expected:
            if i==class1_label:
                profit=1/count_value
            else:
                profit=0
            gain.append(profit)
        sorted_df['gain']=gain
        sorted_df['cumulative_gain']=sorted_df['gain'].cumsum()
        score_cutoff=sorted_df.loc[sorted_df['cumulative_gain'] >= target_percent, 'predicted_prob_class1'].iloc[0]
        cum_gain_at_cut_off=sorted_df.loc[sorted_df['cumulative_gain'] >= target_percent, 'cumulative_gain'].iloc[0]
    
        return (sorted_df, 
                score_cutoff, cum_gain_at_cut_off)

