import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import concurrent.futures
from matplotlib.pyplot import figure
import seaborn as sns
#from scipy.optimize import minimize
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd



def Canvia_Nom_NNet_amb_guio(f_NNet,verbose=False):
    """La sortida del Nnet canvia el nom del pac. Canvia - a .
    En el cas del SET1, això passa si hi ha 3 punts al nom. Són dos i guió"""
    Num_pacs=f_NNet.shape[0]
    for i_pac in range(Num_pacs):
        name_pac=f_NNet.iloc[i_pac,0]
        #print (name_pac)
        num_punts_nom=name_pac.count('.')
        if num_punts_nom >=3:
            if verbose==True:
                print( num_punts_nom,name_pac )
            name_subs=name_pac.replace('.','-',1)   # Substitueix només la primera aparició
            f_NNet.iloc[i_pac,0]=name_subs
        if num_punts_nom > 3:
            print ('WARNING, Nombre punts > 3')
    return
        
    
    
def Check_si_Unamed_son_diferents(pd1,pd2):

    # Determinar valores diferentes

    # concatenate the two dataframes
    combined_df = pd.concat([pd1, pd2], ignore_index=True)

    # drop duplicates
    combined_df = combined_df.drop_duplicates(subset='Unnamed: 0', keep=False)
    print('Difefències en Unnamed: 0:',list(combined_df['Unnamed: 0']) )
    
    
def Plot_gens_IMC(df_allinfo_GSE1,sel_gens,titulo=''):
    """Bar plot dels gens en llista sel_gens, respecte a col prediction"""

    gens_i_prediction=sel_gens+['prediction']

    index_cluster1=df_allinfo_GSE1[gens_i_prediction]['prediction']=='Cluster_1'
    index_cluster2=df_allinfo_GSE1[gens_i_prediction]['prediction']=='Cluster_2'
    index_cluster3=df_allinfo_GSE1[gens_i_prediction]['prediction']=='Cluster_3'

    mean_IMC1=df_allinfo_GSE1[index_cluster1][sel_gens].mean()
    mean_IMC2=df_allinfo_GSE1[index_cluster2][sel_gens].mean()
    mean_IMC3=df_allinfo_GSE1[index_cluster3][sel_gens].mean()


    n = len(mean_IMC1.loc[sel_gens].index)
    x = np.arange(n)
    width = 0.25
    plt.title(titulo)
    plt.ylabel("Gene expression")
    plt.bar(x - width, mean_IMC1.loc[sel_gens].values, width=width, label='IMC1')
    plt.bar(x, mean_IMC2.loc[sel_gens].values, width=width, label='IMC2')
    plt.bar(x + width, mean_IMC3.loc[sel_gens].values, width=width, label='IMC3')
    plt.xticks(x, mean_IMC1.loc[sel_gens].index)
    plt.legend(loc='best')
    plt.show()
    
def Plot_HeatMap_IMC(df_allinfo_GSE1,sel_gens,factorAug=1.5,titulo='',nom_figure='figure.png',typeFig=''):
    """Bar plot dels gens en llista sel_gens, respecte a col prediction"""
    #global mean_ALL, mean_IMC1,pd_mean_IMC
    # Pel format
    figure(dpi=400)
    sns.set(font_scale=1.4) # font size 2
    if typeFig=='Size20':
        figure(figsize=(4, 30), dpi=400)
        sns.set(font_scale=1.4) # font size 2
    #else:
    #    plt.suptitle(titulo)
    
    
    gens_i_prediction=sel_gens+['prediction']

    index_cluster1=df_allinfo_GSE1[gens_i_prediction]['prediction']=='Cluster_1'
    index_cluster2=df_allinfo_GSE1[gens_i_prediction]['prediction']=='Cluster_2'
    index_cluster3=df_allinfo_GSE1[gens_i_prediction]['prediction']=='Cluster_3'

    mean_IMC1=df_allinfo_GSE1[index_cluster1][sel_gens].mean(axis=0)
    mean_IMC2=df_allinfo_GSE1[index_cluster2][sel_gens].mean(axis=0)
    mean_IMC3=df_allinfo_GSE1[index_cluster3][sel_gens].mean(axis=0)
   
    mean_ALL=df_allinfo_GSE1[sel_gens].mean(axis=0)
    std_ALL=df_allinfo_GSE1[sel_gens].std(axis=0)
    
    pd_mean_IMC=pd.DataFrame()
    pd_mean_IMC['IMC1']=(mean_IMC1[sel_gens].values-mean_ALL[sel_gens].values)/std_ALL[sel_gens]
    pd_mean_IMC['IMC2']=(mean_IMC2[sel_gens].values-mean_ALL[sel_gens].values)/std_ALL[sel_gens]
    pd_mean_IMC['IMC3']=(mean_IMC3[sel_gens].values-mean_ALL[sel_gens].values)/std_ALL[sel_gens]
    #HeatMap
    sns.heatmap(pd_mean_IMC*factorAug,
                linewidth=1, linecolor='w',
            cmap='vlag',vmin=-1,vmax=1,cbar=True,xticklabels=True, yticklabels=True)

    #plt.title('IMC1             IMC2               IMC3')
    plt.ylabel('')
    #plt.savefig(nom_figure,dpi=600)
    plt.savefig(nom_figure, bbox_inches='tight', pad_inches=0) # Nou
    plt.show()
    return pd_mean_IMC


def Tukey_HeadMap_IMC(df_allinfo_GSE1,sel_gens,printmsg=True):
    """ANOVA and Tukey test  dels gens en llista sel_gens, respecte a col prediction"""
    global mean_ALL, mean_IMC1,pd_mean_IMC,index_cluster1
    # Pel format
    #if typeFig=='Frontiers':
    #    figure(figsize=(4, 20), dpi=100)
    #else:
    #    plt.suptitle(titulo)
    
    
    gens_i_prediction=sel_gens+['prediction']

    index_cluster1=df_allinfo_GSE1[gens_i_prediction]['prediction']=='Cluster_1'
    index_cluster2=df_allinfo_GSE1[gens_i_prediction]['prediction']=='Cluster_2'
    index_cluster3=df_allinfo_GSE1[gens_i_prediction]['prediction']=='Cluster_3'

    mean_IMC1=df_allinfo_GSE1[index_cluster1][sel_gens].mean(axis=0)
    mean_IMC2=df_allinfo_GSE1[index_cluster2][sel_gens].mean(axis=0)
    mean_IMC3=df_allinfo_GSE1[index_cluster3][sel_gens].mean(axis=0)
   
    mean_ALL=df_allinfo_GSE1[sel_gens].mean(axis=0)
    std_ALL=df_allinfo_GSE1[sel_gens].std(axis=0)
    
#    pd_mean_IMC=pd.DataFrame()
#    pd_mean_IMC['IMC1']=(mean_IMC1[sel_gens].values-mean_ALL[sel_gens].values)/std_ALL[sel_gens]
#    pd_mean_IMC['IMC2']=(mean_IMC2[sel_gens].values-mean_ALL[sel_gens].values)/std_ALL[sel_gens]
#    pd_mean_IMC['IMC3']=(mean_IMC3[sel_gens].values-mean_ALL[sel_gens].values)/std_ALL[sel_gens]
#    #HeatMap
    
    pd_stat=pd.DataFrame()

          
        
    # ANOVA
    global tukey
    for gen1 in sel_gens:
        if printmsg==True:
            print('#Gen: ',gen1)
    
        Result_Anova=f_oneway(list(df_allinfo_GSE1[index_cluster1][gen1]),
                              list(df_allinfo_GSE1[index_cluster2][gen1]),
                              list(df_allinfo_GSE1[index_cluster3][gen1]))
        
        if printmsg==True:
            print(Result_Anova)
        

       # TUKEY 
        tukey = pairwise_tukeyhsd(endog=list(df_allinfo_GSE1[gen1]),
                          groups=df_allinfo_GSE1['prediction'],
                          alpha=0.05)
        if printmsg==True:
            print(tukey)

        
        
        val_IMC1=(mean_IMC1[gen1]-mean_ALL[gen1])/std_ALL[gen1]
        val_IMC2=(mean_IMC2[gen1]-mean_ALL[gen1])/std_ALL[gen1]
        val_IMC3=(mean_IMC3[gen1]-mean_ALL[gen1])/std_ALL[gen1]
        
        if printmsg==True:
            print('Valor medio del cluster_1:',val_IMC1)        
        
        pd_stat[gen1]=[val_IMC1,val_IMC2,val_IMC3,Result_Anova,tukey.pvalues ]

    
    return pd_stat    


def Commom_genes(gens_path,gens_ref):
    """Torna els gens de gens_path que estiguin en gens_ref (com a llista)"""
    gens_common=[]
    for ele in gens_path:
        if ele in gens_ref:
            #print(ele)
            gens_common.append(ele) 
            continue
    print('Num. gens: {} de {}  {:.2f} % '.format(len(gens_common), len(gens_path),  100*len(gens_common)/len(gens_path),'%') )
    return gens_common



def Commom_genes2(gens_path,gens_ref):
    """Torna els gens de gens_path que estiguin en gens_ref (com a llista)"""
    gens_common=[]
    for ele in gens_path:
        if ele in gens_ref:
            #print(ele)
            gens_common.append(ele) 
            continue
    proportion=100*len(gens_common)/len(gens_path)
    print('Num. gens: {} de {}  {:.2f} % '.format(len(gens_common), len(gens_path), proportion  ,'%') )
    return gens_common,proportion


def leer_gens_columna(nom_file):
    with open(nom_file) as f1:
        f2=f1.readlines()
        stripped = [s.strip()  for s in f2]
        if stripped[1][0] == '>':   # Eliminar 2 linea de comentario
            stripped.pop(1)
    return  stripped


def leer_gens_gmt(nom_file,nom_path):
    """Llegeix el nom_path del fitxer gmt
    retorna i[0]: nom_del_path_llegit=nom_path
            i[1]: link 
            i[2:]: Llista de gens """
    with open(nom_file) as f1:
        for lin in f1:
            words=lin.split()
            #print(words[0],nom_path)
            if words[0]==nom_path:
                return words
                #print('Link:',words[1])
    return -1,'FILE or PATH FOT FOUND'


###   Se utiliza el Plot_HeatMap_PATHS_IMC_stat
# Falta corregir la SE para paths
def Plot_HeatMap_PATHS_IMC_old(df_allinfo_GSE1,dict_path_gens,factorAug=2.0,titulo='',typeFig='',nom_figure='figure.png'):
    """HeatMap plot dels PATHS, respecte a column prediction"""
    #fuente_medida=10
    if typeFig=='Llarga1':
        figure(figsize=(4, 12), dpi=400)
    elif typeFig=='Llarga2':
        figure(figsize=(4, 8), dpi=400)
        #fuente_medida=16
        sns.set(font_scale=1.8) # font size 2
    else:
        plt.suptitle(titulo) 
    
    
    pd_mean_IMC=pd.DataFrame()
    
    for nom_path in dict_path_gens.keys():
        sel_gens=dict_path_gens[nom_path]
        print('Nom path: {}, num. gens:{}'.format(nom_path,len(sel_gens)))   
    
        gens_i_prediction=sel_gens+['prediction']

        index_cluster1=df_allinfo_GSE1[gens_i_prediction]['prediction']=='Cluster_1'
        index_cluster2=df_allinfo_GSE1[gens_i_prediction]['prediction']=='Cluster_2'
        index_cluster3=df_allinfo_GSE1[gens_i_prediction]['prediction']=='Cluster_3'
    
        mean_IMC1=df_allinfo_GSE1[index_cluster1][sel_gens].mean(axis=0)
        mean_IMC2=df_allinfo_GSE1[index_cluster2][sel_gens].mean(axis=0)
        mean_IMC3=df_allinfo_GSE1[index_cluster3][sel_gens].mean(axis=0)
   
        mean_ALL=df_allinfo_GSE1[sel_gens].mean(axis=0)
        std_ALL=df_allinfo_GSE1[sel_gens].std(axis=0)
    
        val_gens_path_IMC1=(mean_IMC1.values-mean_ALL.values)/std_ALL
        val_path_IMC1=np.average(val_gens_path_IMC1)
        val_gens_path_IMC2=(mean_IMC2.values-mean_ALL.values)/std_ALL
        val_path_IMC2=np.average(val_gens_path_IMC2)
        val_gens_path_IMC3=(mean_IMC3.values-mean_ALL.values)/std_ALL
        val_path_IMC3=np.average(val_gens_path_IMC3)    
        #pd_mean_IMC=pd_mean_IMC.append(pd.DataFrame( {'IMC1': (val_path_IMC1), 'IMC2': (val_path_IMC2), 'IMC3': (val_path_IMC3) },index=[nom_path]  ))
        pd_mean_IMC=pd.concat([pd_mean_IMC,pd.DataFrame( {'IMC1': (val_path_IMC1), 'IMC2': (val_path_IMC2), 'IMC3': (val_path_IMC3) },index=[nom_path]  ) ] )

    
    #HeatMap
    sns.heatmap(pd_mean_IMC*factorAug,
                linewidth=1, linecolor='w',
            cmap='vlag',vmin=-1,vmax=1,cbar=True,xticklabels=True, yticklabels=True)

    #plt.title('IMC1             IMC2               IMC3')
    plt.ylabel('')
    #plt.savefig(nom_figure,dpi=600)
    plt.savefig(nom_figure, bbox_inches='tight', pad_inches=0) # Nou
    plt.show()
    return pd_mean_IMC

def Plot_HeatMap_PATHS_IMC_stat(df_allinfo_GSE1,dict_path_gens,factorAug=2.0,titulo='',typeFig='',nom_figure='figure.png'):
    """ Com la versió anterior, però el pandas que retorna, dona informació sobre vals avg. i std."""
    #global mean_IMC1,mean_ALL,std_ALL
    if typeFig=='Llarga1':
        figure(figsize=(4, 12), dpi=400)
    elif typeFig=='Llarga2':
        figure(figsize=(4, 8), dpi=400)
        #fuente_medida=16
        sns.set(font_scale=1.8) # font size 2
    else:
        plt.suptitle(titulo) 
    
    
    pd_mean_IMC=pd.DataFrame()
    
    for nom_path in dict_path_gens.keys():
        sel_gens=dict_path_gens[nom_path]
        print('Nom path: {}, num. gens:{}'.format(nom_path,len(sel_gens)))   
    
        gens_i_prediction=sel_gens+['prediction']

        index_cluster1=df_allinfo_GSE1[gens_i_prediction]['prediction']=='Cluster_1'
        index_cluster2=df_allinfo_GSE1[gens_i_prediction]['prediction']=='Cluster_2'
        index_cluster3=df_allinfo_GSE1[gens_i_prediction]['prediction']=='Cluster_3'
   
        index_ALL= (index_cluster1)|(index_cluster2)|(index_cluster3)

        mean_IMC1=df_allinfo_GSE1[index_cluster1][sel_gens].mean(axis=1) #Average pathway for patien
        mean_IMC2=df_allinfo_GSE1[index_cluster2][sel_gens].mean(axis=1)
        mean_IMC3=df_allinfo_GSE1[index_cluster3][sel_gens].mean(axis=1)
        
        std_IMC1=np.std(mean_IMC1)  # Es escalar
        std_IMC2=np.std(mean_IMC2)
        std_IMC3=np.std(mean_IMC3)
         
        mean_ALL=df_allinfo_GSE1[index_ALL][sel_gens].mean(axis=1)
        std_ALL=np.std(mean_ALL)


        val_path_IMC1=(np.mean(mean_IMC1.values)-np.mean(mean_ALL.values))/std_ALL
        val_path_IMC2=(np.mean(mean_IMC2.values)-np.mean(mean_ALL.values))/std_ALL
        val_path_IMC3=(np.mean(mean_IMC3.values)-np.mean(mean_ALL.values))/std_ALL
    
        pd_mean_IMC=pd.concat([pd_mean_IMC,pd.DataFrame( {'IMC1': (val_path_IMC1), 'IMC2': (val_path_IMC2), 'IMC3': (val_path_IMC3),
        'avgIMC1': np.average(mean_IMC1), 'avgIMC2': np.average(mean_IMC2), 'avgIMC3': np.average(mean_IMC3),
         'stdIMC1': np.average(std_IMC1), 'stdIMC2': np.average(std_IMC2), 'stdIMC3': np.average(std_IMC3),'stdALL': np.average(std_ALL),},index=[nom_path]  ) ] )

    
    #HeatMap
    sns.heatmap(pd_mean_IMC[['IMC1','IMC2','IMC3']]*factorAug,
                linewidth=1, linecolor='w',
            cmap='vlag',vmin=-1,vmax=1,cbar=True,xticklabels=True, yticklabels=True)

    #plt.title('IMC1             IMC2               IMC3')
    plt.ylabel('')
    #plt.savefig(nom_figure,dpi=600)
    plt.savefig(nom_figure, bbox_inches='tight', pad_inches=0) # Nou
    plt.show()
    return pd_mean_IMC





def Tukey_HeadMap_PATHS_IMC(df_allinfo_GSE1,dict_path_gens,printmsg=True):
    """ANOVA and Tukey test  dels gens en llista sel_gens, respecte a col prediction"""  
    # ANOVA
    global tukey,val_gens_path_IMC1,mean_IMC1
    
    #pd_mean_IMC=pd.DataFrame()
    pd_stat=pd.DataFrame()
    
    for nom_path in dict_path_gens.keys():
        sel_gens=dict_path_gens[nom_path]
        print('Nom path: {}, num. gens:{}'.format(nom_path,len(sel_gens)))   
    
        gens_i_prediction=sel_gens+['prediction']

        index_cluster1=df_allinfo_GSE1[gens_i_prediction]['prediction']=='Cluster_1'
        index_cluster2=df_allinfo_GSE1[gens_i_prediction]['prediction']=='Cluster_2'
        index_cluster3=df_allinfo_GSE1[gens_i_prediction]['prediction']=='Cluster_3'
        # Actualizado, mean para los paths    
        index_ALL= (index_cluster1)|(index_cluster2)|(index_cluster3)

        mean_IMC1=df_allinfo_GSE1[index_cluster1][sel_gens].mean(axis=1) #Average pathway for patien
        mean_IMC2=df_allinfo_GSE1[index_cluster2][sel_gens].mean(axis=1)
        mean_IMC3=df_allinfo_GSE1[index_cluster3][sel_gens].mean(axis=1)
        
        std_IMC1=np.std(mean_IMC1)  # Es escalar
        std_IMC2=np.std(mean_IMC2)
        std_IMC3=np.std(mean_IMC3)
         
        mean_ALL=df_allinfo_GSE1[index_ALL][sel_gens].mean(axis=1)
        std_ALL=np.std(mean_ALL)

        val_path_IMC1=(np.mean(mean_IMC1.values)-np.mean(mean_ALL.values))/std_ALL
        val_path_IMC2=(np.mean(mean_IMC2.values)-np.mean(mean_ALL.values))/std_ALL
        val_path_IMC3=(np.mean(mean_IMC3.values)-np.mean(mean_ALL.values))/std_ALL


        #pd_mean_IMC=pd_mean_IMC.append(pd.DataFrame( {'IMC1': (val_path_IMC1), 'IMC2': (val_path_IMC2), 'IMC3': (val_path_IMC3) },index=[nom_path]  ))
        #pd_mean_IMC=pd.concat([pd_mean_IMC,pd.DataFrame( {'IMC1': (val_path_IMC1), 'IMC2': (val_path_IMC2), 'IMC3': (val_path_IMC3) },index=[nom_path]  ) ] )
    
        Result_Anova=f_oneway(list(df_allinfo_GSE1[index_cluster1][sel_gens].mean(axis=1)),
                list(df_allinfo_GSE1[index_cluster2][sel_gens].mean(axis=1)),
                list(df_allinfo_GSE1[index_cluster3][sel_gens].mean(axis=1)) )
        
        if printmsg==True:
            print(Result_Anova)
        

       # TUKEY 
        tukey = pairwise_tukeyhsd(endog=list(df_allinfo_GSE1[sel_gens].mean(axis=1)),
                          groups=df_allinfo_GSE1['prediction'],
                          alpha=0.05)
        if printmsg==True:
            print(tukey)
        

        pd_stat[nom_path]=[val_path_IMC1,val_path_IMC2,val_path_IMC3,Result_Anova,tukey.pvalues ]        
        
    return pd_stat 

def print_ANOVA2(pd_ANOVA,nom_excel=None):
    pd_Tabla=pd.DataFrame()
    pd_Tabla['GeneName']=pd_ANOVA.columns
    num_gens=len(pd_ANOVA.loc[0])
    pd_Tabla['IMC1']=[pd_ANOVA.loc[0][i] for i in range(num_gens)]
    pd_Tabla['IMC2']=[pd_ANOVA.loc[1][i] for i in range(num_gens)]
    pd_Tabla['IMC3']=[pd_ANOVA.loc[2][i] for i in range(num_gens)]
    pd_Tabla['F_oneway']=[pd_ANOVA.loc[3][i][0] for i in range(num_gens)]
    pd_Tabla['P_oneway']=[pd_ANOVA.loc[3][i][1] for i in range(num_gens)]
    pd_Tabla['p-Tukey12']=[pd_ANOVA.loc[4][i][0] for i in range(num_gens)]
    pd_Tabla['p-Tukey13']=[pd_ANOVA.loc[4][i][1] for i in range(num_gens)]
    pd_Tabla['p-Tukey23']=[pd_ANOVA.loc[4][i][2] for i in range(num_gens)]
    
    pd_Tabla['compara1']=(pd_Tabla['p-Tukey12']<0.05) & (pd_Tabla['p-Tukey13']<0.05) 
    pd_Tabla['compara2']=(pd_Tabla['p-Tukey12']<0.05) & (pd_Tabla['p-Tukey23']<0.05) 
    pd_Tabla['compara3']=(pd_Tabla['p-Tukey13']<0.05) & (pd_Tabla['p-Tukey23']<0.05) 
    
    # pasar a entero
    
    pd_Tabla['compara1'] = pd_Tabla['compara1'].replace({True: 1, False: 0})
    pd_Tabla['compara2'] = pd_Tabla['compara2'].replace({True: 1, False: 0})
    pd_Tabla['compara3'] = pd_Tabla['compara3'].replace({True: 1, False: 0})
    
     
    pd_Tabla=pd_Tabla.round({'p-Tukey12': 3, 'p-Tukey13': 3, 'p-Tukey23': 3,'F_oneway': 3 })
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None,'display.precision', 3):
    #    print(pd_Tabla)
    if nom_excel!=None:
        pd_Tabla.to_excel(nom_excel)
    
    
    return pd_Tabla


                
