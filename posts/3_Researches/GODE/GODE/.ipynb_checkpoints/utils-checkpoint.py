import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# utils
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)
from haversine import haversine
from IPython.display import HTML
import plotly.graph_objects as go
import copy

from pygsp import graphs, filters, plotting, utils

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc

def GODE_Anomalous(df,contamination = 0.05):
    if 'yhat' in df:
            mse = ((df['yhat'] - df['y'])**2).tolist()
    elif 'y' in df:
            mse = ((df['fhat'] - df['f'])**2).tolist()
    else : pass
        
    sorted_data = sorted(mse,reverse=True)
    index = int(len(sorted_data) * contamination)
    percent = sorted_data[index]
    outlier = list(map(lambda x: 1 if x > percent else 0,mse))
    outlier_index = [i for i, value in enumerate(mse) if value > percent]
    return mse, outlier, outlier_index
        
def Linear_plot(df, true_outlier, outlier_index, *args, ref=20,cuts=0,cutf=995,**kwargs):
    fig,ax = plt.subplots(figsize=(10,10))
    ax.scatter(df['x'], df['y'],color='gray',s=50,alpha=0.7)
    ax.scatter(df['x'][true_outlier],df['y'][true_outlier],color='red',s=50)
    ax.plot(df['x'][cuts:cutf],df['yhat'][cuts:cutf], '--k',lw=3)
    ax.scatter(df['x'][outlier_index],df['y'][outlier_index],color='red',s=550,facecolors='none', edgecolors='r')
    fig.tight_layout()
    
     # fig.savefig('fig1_231103.eps',format='eps')

def Orbit_plot(df,true_outlier, outlier_index, *args, ref=60, **kwargs):
    
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(30,15),subplot_kw={"projection":"3d"})
    ax1.grid(False)
    ax1.scatter3D(df['x'],df['y'],df['f'],zdir='z',s=50,marker='.',color='gray')
    ax1.scatter3D(df['x'][true_outlier],df['y'][true_outlier],df['f'][true_outlier],zdir='z',s=50,marker='.',color='red')
    ax1.scatter3D(df['x'][outlier_index],df['y'][outlier_index],df['f'][outlier_index],edgecolors='red',zdir='z',s=100,facecolors='none')
    
    ax1.plot3D(df['x'],df['y'],df['f1'],'--k',lw=3)
    ax2.view_init(elev=30., azim=60)
    
    ax2.grid(False)
    ax2.scatter3D(df['x'],df['y'],df['f'],zdir='z',s=50,marker='.',color='gray')
    ax2.scatter3D(df['x'][true_outlier],df['y'][true_outlier],df['f'][true_outlier],zdir='z',s=50,marker='.',color='red') 
    ax2.scatter3D(df['x'][outlier_index],df['y'][outlier_index],df['f'][outlier_index], edgecolors='red',zdir='z',s=100,facecolors='none')
    ax2.plot3D(df['x'],df['y'],df['f1'],'--k',lw=3)
    ax2.view_init(elev=30., azim=40)
    
    ax3.grid(False)
    ax3.scatter3D(df['x'],df['y'],df['f'],zdir='z',s=50,marker='.',color='gray')
    ax3.scatter3D(df['x'][true_outlier],df['y'][true_outlier],df['f'][true_outlier],zdir='z',s=50,marker='.',color='red') 
    ax3.scatter3D(df['x'][outlier_index],df['y'][outlier_index],df['f'][outlier_index],edgecolors='red',zdir='z',s=100,facecolors='none')
    ax3.plot3D(df['x'],df['y'],df['f1'],'--k',lw=3)
    ax3.view_init(elev=30., azim=10)
    
    # fig.savefig('fig2_231103.eps',format='eps')

def Bunny_plot(df,true_outlier, outlier_index, *args, ref=6, **kwargs):

    fig = plt.figure(figsize=(30,12),dpi=400)
    ax1 = fig.add_subplot(251, projection='3d')
    ax1.grid(False)
    ax1.scatter3D(df['x'],df['y'],df['z'],c='gray',zdir='z',alpha=0.5,marker='.')
    ax1.view_init(elev=60., azim=-90)

    ax2= fig.add_subplot(252, projection='3d')
    ax2.grid(False)
    ax2.scatter3D(df['x'],df['y'],df['z'],c=df['f1'],cmap='hsv',zdir='z',marker='.',alpha=0.5,vmin=-12,vmax=10)
    ax2.view_init(elev=60., azim=-90)

    ax3= fig.add_subplot(253, projection='3d')
    ax3.grid(False)
    ax3.scatter3D(df['x'],df['y'],df['z'],c=df['f'],cmap='hsv',zdir='z',marker='.',alpha=0.5,vmin=-12,vmax=10)
    ax3.view_init(elev=60., azim=-90)
    
    ax4= fig.add_subplot(254, projection='3d')
    ax4.grid(False)
    ax4.scatter3D(df['x'],df['y'],df['z'],c=df['f'],cmap='hsv',zdir='z',marker='.',vmin=-12,vmax=10,s=1)
    ax4.scatter3D(df['x'][true_outlier],df['y'][true_outlier],df['z'][true_outlier],c=df['f'][true_outlier],cmap='hsv',zdir='z',marker='.',s=50)
    ax4.view_init(elev=60., azim=-90)

    ax5= fig.add_subplot(255, projection='3d')
    ax5.grid(False)
    ax5.scatter3D(df['x'],df['y'],df['z'],c=df['f'],cmap='hsv',zdir='z',marker='.',vmin=-12,vmax=10,s=1)
    ax5.scatter3D(df['x'][true_outlier],df['y'][true_outlier],df['z'][true_outlier],c=df['f'][true_outlier],cmap='hsv',zdir='z',marker='.',s=50)
    ax5.scatter3D(df['x'][outlier_index],df['y'][outlier_index],df['z'][outlier_index],zdir='z',s=550,marker='.',edgecolors='red',facecolors='none')
    ax5.view_init(elev=60., azim=-90)
    
    ax6 = fig.add_subplot(256, projection='3d')
    ax6.grid(False)
    ax6.scatter3D(df['x'],df['y'],df['z'],c='gray',zdir='z',alpha=0.5,marker='.')
    ax6.view_init(elev=-60., azim=-90)

    ax7= fig.add_subplot(257, projection='3d')
    ax7.grid(False)
    ax7.scatter3D(df['x'],df['y'],df['z'],c=df['f1'],cmap='hsv',zdir='z',marker='.',alpha=0.5,vmin=-12,vmax=10)
    ax7.view_init(elev=-60., azim=-90)

    ax8= fig.add_subplot(258, projection='3d')
    ax8.grid(False)
    ax8.scatter3D(df['x'],df['y'],df['z'],c=df['f'],cmap='hsv',zdir='z',marker='.',alpha=0.5,vmin=-12,vmax=10)
    ax8.view_init(elev=-60., azim=-90)
    
    ax9= fig.add_subplot(259, projection='3d')
    ax9.grid(False)
    ax9.scatter3D(df['x'],df['y'],df['z'],c=df['f'],cmap='hsv',zdir='z',marker='.',vmin=-12,vmax=10,s=1)
    ax9.scatter3D(df['x'][true_outlier],df['y'][true_outlier],df['z'][true_outlier],c=df['f'][true_outlier],cmap='hsv',zdir='z',marker='.',s=50)
    ax9.view_init(elev=-60., azim=-90)

    ax10= fig.add_subplot(2,5,10, projection='3d')
    ax10.grid(False)
    ax10.scatter3D(df['x'],df['y'],df['z'],c=df['f'],cmap='hsv',zdir='z',marker='.',vmin=-12,vmax=10,s=1)
    ax10.scatter3D(df['x'][true_outlier],df['y'][true_outlier],df['z'][true_outlier],c=df['f'][true_outlier],cmap='hsv',zdir='z',marker='.',s=50)
    ax10.scatter3D(df['x'][outlier_index],df['y'][outlier_index],df['z'][outlier_index],zdir='z',s=550,marker='.',edgecolors='red',facecolors='none')
    ax10.view_init(elev=-60., azim=-90)    
    
    # fig.savefig('fig_bunny.eps',format='eps')

class Conf_matrx:
    def __init__(self,original,compare):
        self.original = original
        self.compare = compare
    def conf(self,name="Method"):
        self.name = name
        self.conf_matrix = confusion_matrix(self.original, self.compare)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(self.conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
        for i in range(self.conf_matrix.shape[0]):
            for j in range(self.conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=self.conf_matrix[i, j], va='center', ha='center', size='xx-large')
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix of ' + str(name), fontsize=18)
        plt.show()
        
        self.acc = accuracy_score(self.original, self.compare)
        self.pre = precision_score(self.original, self.compare)
        self.rec = recall_score(self.original, self.compare)
        self.f1 = f1_score(self.original, self.compare)

        print('Accuracy: %.3f' % self.acc)
        print('Precision: %.3f' % self.pre)
        print('Recall: %.3f' % self.rec)
        print('F1 Score: %.3f' % self.f1)

    def __call__(self):
        return {'Accuracy': self.acc, 'Precision':self.pre, 'Recall':self.rec, 'F1 Score':self.f1}


