from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
# Creating random dataset
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


df = pd.read_excel('FL_UV所有浓度.xlsx')
savefilename='final_fig/FL_UV所有浓度三维不加权简略版.png'
UV_weight=0.2
color1 = ['r', 'g', 'b', 'c', 'y','m', 'k', '#ff00ff', '#99ff66','#8C7853','#7622f8','#Cf9922']
shape = ['o', '^', '+', '<', '>', 'x', 's', 'd', 'p', 'h']

df.drop(axis=0,index=0, inplace=True)
# df.drop(axis=1,columns=0, inplace=True)
title=df.columns[0]
temp=df.values
label1=list(temp[:,0])
label2=list(temp[:,1])
# temp=np.delete(temp,0,axis=1)
UV=temp[:,2:5]
FL=temp[:,5:8]

final=np.vstack((UV,FL))
label1_final=label1+label1
label2_final=label2+label2
pca = PCA(n_components=3)	#实例化
final_NEW = pca.fit_transform(final)			#拟合模型


final_ratio=pca.explained_variance_ratio_


x1=final_NEW[:,0]
y1=final_NEW[:,1]
z1=final_NEW[:,2]

# z1.fill(1)


# y_kmeans = kmeans.fit_predict(X)
# Creating figure
fig= plt.figure(figsize = (16,16))
ax = plt.axes(projection ="3d")
# Add x, and y gridlines for the figure
# ax.grid(b = True, color ='blue',linestyle ='-.', linewidth = 0.5,alpha = 0.3)
# Creating the color map for the plot
# my_cmap = plt.get_cmap('hsv')
# Creating the 3D plot
# sctt = ax.scatter3D(x, y, z,alpha = 0.8,c = (x + y + z),cmap = my_cmap,marker ='^')


type0_1x = []
type0_1y = []
type0_1z = []
type0_2x = []
type0_2y = []
type0_2z = []
type0_3x = []
type0_3y = []
type0_3z = []
type0_4x = []
type0_4y = []
type0_4z = []
type0_5x = []
type0_5y = []
type0_5z = []
type0_6x = []
type0_6y = []
type0_6z = []
type0_7x = []
type0_7y = []
type0_7z = []
type0_8x = []
type0_8y = []
type0_8z = []
type0_9x = []
type0_9y = []
type0_9z = []
type0_10x = []
type0_10y = []
type0_10z = []

type1_1x = []
type1_1y = []
type1_1z = []
type1_2x = []
type1_2y = []
type1_2z = []
type1_3x = []
type1_3y = []
type1_3z = []
type1_4x = []
type1_4y = []
type1_4z = []
type1_5x = []
type1_5y = []
type1_5z = []
type1_6x = []
type1_6y = []
type1_6z = []
type1_7x = []
type1_7y = []
type1_7z = []
type1_8x = []
type1_8y = []
type1_8z = []
type1_9x = []
type1_9y = []
type1_9z = []
type1_10x = []
type1_10y = []
type1_10z = []

type2_1x = []
type2_1y = []
type2_1z = []
type2_2x = []
type2_2y = []
type2_2z = []
type2_3x = []
type2_3y = []
type2_3z = []
type2_4x = []
type2_4y = []
type2_4z = []
type2_5x = []
type2_5y = []
type2_5z = []
type2_6x = []
type2_6y = []
type2_6z = []
type2_7x = []
type2_7y = []
type2_7z = []
type2_8x = []
type2_8y = []
type2_8z = []
type2_9x = []
type2_9y = []
type2_9z = []
type2_10x = []
type2_10y = []
type2_10z = []

type3_1x = []
type3_1y = []
type3_1z = []
type3_2x = []
type3_2y = []
type3_2z = []
type3_3x = []
type3_3y = []
type3_3z = []
type3_4x = []
type3_4y = []
type3_4z = []
type3_5x = []
type3_5y = []
type3_5z = []
type3_6x = []
type3_6y = []
type3_6z = []
type3_7x = []
type3_7y = []
type3_7z = []
type3_8x = []
type3_8y = []
type3_8z = []
type3_9x = []
type3_9y = []
type3_9z = []
type3_10x = []
type3_10y = []
type3_10z = []

type4_1x = []
type4_1y = []
type4_1z = []
type4_2x = []
type4_2y = []
type4_2z = []
type4_3x = []
type4_3y = []
type4_3z = []
type4_4x = []
type4_4y = []
type4_4z = []
type4_5x = []
type4_5y = []
type4_5z = []
type4_6x = []
type4_6y = []
type4_6z = []
type4_7x = []
type4_7y = []
type4_7z = []
type4_8x = []
type4_8y = []
type4_8z = []
type4_9x = []
type4_9y = []
type4_9z = []
type4_10x = []
type4_10y = []
type4_10z = []
label1class=[]
label2class=[]
for item in label1:
    if item not in label1class:
        label1class.append(item)
for item in label2:
    if item not in label2class:
        label2class.append(item)
label_finalclass=[]
for item in label1class:
    for item1 in label2class:
        label_finalclass.append(item+'_'+item1)
t=len(x1)
for i in range(t):
    if label1_final[i]==label1class[0]:
        if label2_final[i]==label2class[0]:
            type0_1x.append(final_NEW[:, 0][i])
            type0_1y.append(final_NEW[:, 1][i])
            type0_1z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[1]:
            type0_2x.append(final_NEW[:, 0][i])
            type0_2y.append(final_NEW[:, 1][i])
            type0_2z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[2]:
            type0_3x.append(final_NEW[:, 0][i])
            type0_3y.append(final_NEW[:, 1][i])
            type0_3z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[3]:
            type0_4x.append(final_NEW[:, 0][i])
            type0_4y.append(final_NEW[:, 1][i])
            type0_4z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[4]:
            type0_5x.append(final_NEW[:, 0][i])
            type0_5y.append(final_NEW[:, 1][i])
            type0_5z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[5]:
            type0_6x.append(final_NEW[:, 0][i])
            type0_6y.append(final_NEW[:, 1][i])
            type0_6z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[6]:
            type0_7x.append(final_NEW[:, 0][i])
            type0_7y.append(final_NEW[:, 1][i])
            type0_7z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[7]:
            type0_8x.append(final_NEW[:, 0][i])
            type0_8y.append(final_NEW[:, 1][i])
            type0_8z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[8]:
            type0_9x.append(final_NEW[:, 0][i])
            type0_9y.append(final_NEW[:, 1][i])
            type0_9z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[9]:
            type0_10x.append(final_NEW[:, 0][i])
            type0_10y.append(final_NEW[:, 1][i])
            type0_10z.append(final_NEW[:, 2][i])
    if label1_final[i]==label1class[1]:
        if label2_final[i]==label2class[0]:
            type1_1x.append(final_NEW[:, 0][i])
            type1_1y.append(final_NEW[:, 1][i])
            type1_1z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[1]:
            type1_2x.append(final_NEW[:, 0][i])
            type1_2y.append(final_NEW[:, 1][i])
            type1_2z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[2]:
            type1_3x.append(final_NEW[:, 0][i])
            type1_3y.append(final_NEW[:, 1][i])
            type1_3z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[3]:
            type1_4x.append(final_NEW[:, 0][i])
            type1_4y.append(final_NEW[:, 1][i])
            type1_4z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[4]:
            type1_5x.append(final_NEW[:, 0][i])
            type1_5y.append(final_NEW[:, 1][i])
            type1_5z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[5]:
            type1_6x.append(final_NEW[:, 0][i])
            type1_6y.append(final_NEW[:, 1][i])
            type1_6z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[6]:
            type1_7x.append(final_NEW[:, 0][i])
            type1_7y.append(final_NEW[:, 1][i])
            type1_7z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[7]:
            type1_8x.append(final_NEW[:, 0][i])
            type1_8y.append(final_NEW[:, 1][i])
            type1_8z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[8]:
            type1_9x.append(final_NEW[:, 0][i])
            type1_9y.append(final_NEW[:, 1][i])
            type1_9z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[9]:
            type1_10x.append(final_NEW[:, 0][i])
            type1_10y.append(final_NEW[:, 1][i])
            type1_10z.append(final_NEW[:, 2][i])
    if label1_final[i]==label1class[2]:
        if label2_final[i]==label2class[0]:
            type2_1x.append(final_NEW[:, 0][i])
            type2_1y.append(final_NEW[:, 1][i])
            type2_1z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[1]:
            type2_2x.append(final_NEW[:, 0][i])
            type2_2y.append(final_NEW[:, 1][i])
            type2_2z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[2]:
            type2_3x.append(final_NEW[:, 0][i])
            type2_3y.append(final_NEW[:, 1][i])
            type2_3z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[3]:
            type2_4x.append(final_NEW[:, 0][i])
            type2_4y.append(final_NEW[:, 1][i])
            type2_4z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[4]:
            type2_5x.append(final_NEW[:, 0][i])
            type2_5y.append(final_NEW[:, 1][i])
            type2_5z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[5]:
            type2_6x.append(final_NEW[:, 0][i])
            type2_6y.append(final_NEW[:, 1][i])
            type2_6z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[6]:
            type2_7x.append(final_NEW[:, 0][i])
            type2_7y.append(final_NEW[:, 1][i])
            type2_7z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[7]:
            type2_8x.append(final_NEW[:, 0][i])
            type2_8y.append(final_NEW[:, 1][i])
            type2_8z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[8]:
            type2_9x.append(final_NEW[:, 0][i])
            type2_9y.append(final_NEW[:, 1][i])
            type2_9z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[9]:
            type2_10x.append(final_NEW[:, 0][i])
            type2_10y.append(final_NEW[:, 1][i])
            type2_10z.append(final_NEW[:, 2][i])
    if label1_final[i]==label1class[3]:
        if label2_final[i]==label2class[0]:
            type3_1x.append(final_NEW[:, 0][i])
            type3_1y.append(final_NEW[:, 1][i])
            type3_1z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[1]:
            type3_2x.append(final_NEW[:, 0][i])
            type3_2y.append(final_NEW[:, 1][i])
            type3_2z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[2]:
            type3_3x.append(final_NEW[:, 0][i])
            type3_3y.append(final_NEW[:, 1][i])
            type3_3z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[3]:
            type3_4x.append(final_NEW[:, 0][i])
            type3_4y.append(final_NEW[:, 1][i])
            type3_4z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[4]:
            type3_5x.append(final_NEW[:, 0][i])
            type3_5y.append(final_NEW[:, 1][i])
            type3_5z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[5]:
            type3_6x.append(final_NEW[:, 0][i])
            type3_6y.append(final_NEW[:, 1][i])
            type3_6z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[6]:
            type3_7x.append(final_NEW[:, 0][i])
            type3_7y.append(final_NEW[:, 1][i])
            type3_7z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[7]:
            type3_8x.append(final_NEW[:, 0][i])
            type3_8y.append(final_NEW[:, 1][i])
            type3_8z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[8]:
            type3_9x.append(final_NEW[:, 0][i])
            type3_9y.append(final_NEW[:, 1][i])
            type3_9z.append(final_NEW[:, 2][i])
        if label2_final[i]==label2class[9]:
            type3_10x.append(final_NEW[:, 0][i])
            type3_10y.append(final_NEW[:, 1][i])
            type3_10z.append(final_NEW[:, 2][i])
    if label1_final[i]==label1class[4]:
        if label2_final[i]==label2class[0]:
            type4_1x.append(final_NEW[:, 0][i])
            type4_1y.append(final_NEW[:, 1][i])
            type4_1z.append(final_NEW[:, 1][i])
        if label2_final[i]==label2class[1]:
            type4_2x.append(final_NEW[:, 0][i])
            type4_2y.append(final_NEW[:, 1][i])
            type4_2z.append(final_NEW[:, 1][i])
        if label2_final[i]==label2class[2]:
            type4_3x.append(final_NEW[:, 0][i])
            type4_3y.append(final_NEW[:, 1][i])
            type4_3z.append(final_NEW[:, 1][i])
        if label2_final[i]==label2class[3]:
            type4_4x.append(final_NEW[:, 0][i])
            type4_4y.append(final_NEW[:, 1][i])
            type4_4z.append(final_NEW[:, 1][i])
        if label2_final[i]==label2class[4]:
            type4_5x.append(final_NEW[:, 0][i])
            type4_5y.append(final_NEW[:, 1][i])
            type4_5z.append(final_NEW[:, 1][i])
        if label2_final[i]==label2class[5]:
            type4_6x.append(final_NEW[:, 0][i])
            type4_6y.append(final_NEW[:, 1][i])
            type4_6z.append(final_NEW[:, 1][i])
        if label2_final[i]==label2class[6]:
            type4_7x.append(final_NEW[:, 0][i])
            type4_7y.append(final_NEW[:, 1][i])
            type4_7z.append(final_NEW[:, 1][i])
        if label2_final[i]==label2class[7]:
            type4_8x.append(final_NEW[:, 0][i])
            type4_8y.append(final_NEW[:, 1][i])
            type4_8z.append(final_NEW[:, 1][i])
        if label2_final[i]==label2class[8]:
            type4_9x.append(final_NEW[:, 0][i])
            type4_9y.append(final_NEW[:, 1][i])
            type4_9z.append(final_NEW[:, 1][i])
        if label2_final[i]==label2class[9]:
            type4_10x.append(final_NEW[:, 0][i])
            type4_10y.append(final_NEW[:, 1][i])
            type4_10z.append(final_NEW[:, 1][i])





# ax0=ax.scatter3D(type0_1x, type0_1y, type0_1z,c=color1[0],marker =shape[0],s=50)
# ax1=ax.scatter3D(type0_2x, type0_2y, type0_2z,c=color1[0],marker =shape[1],s=50)
# ax2=ax.scatter3D(type0_3x, type0_3y, type0_3z,c=color1[0],marker =shape[2],s=50)
# ax3=ax.scatter3D(type0_4x, type0_4y, type0_4z,c=color1[0],marker =shape[3],s=50)
# ax4=ax.scatter3D(type0_5x, type0_5y, type0_5z,c=color1[0],marker =shape[4],s=50)
# ax5=ax.scatter3D(type0_6x, type0_6y, type0_6z,c=color1[0],marker =shape[5],s=50)
# ax6=ax.scatter3D(type0_7x, type0_7y, type0_7z,c=color1[0],marker =shape[6],s=50)
ax7=ax.scatter3D(type0_8x, type0_8y, type0_8z,c=color1[0],marker =shape[7],s=50)
ax8=ax.scatter3D(type0_9x, type0_9y, type0_9z,c=color1[0],marker =shape[8],s=50)
ax9=ax.scatter3D(type0_10x, type0_10y, type0_10z,c=color1[0],marker =shape[9],s=50)

ax10=ax.scatter3D(type1_1x, type1_1y, type1_1z,c=color1[1],marker =shape[0],s=50)
ax11=ax.scatter3D(type1_2x, type1_2y, type1_2z,c=color1[1],marker =shape[1],s=50)
ax12=ax.scatter3D(type1_3x, type1_3y, type1_3z,c=color1[1],marker =shape[2],s=50)
# ax13=ax.scatter3D(type1_4x, type1_4y, type1_4z,c=color1[1],marker =shape[3],s=50)
# ax14=ax.scatter3D(type1_5x, type1_5y, type1_5z,c=color1[1],marker =shape[4],s=50)
# ax15=ax.scatter3D(type1_6x, type1_6y, type1_6z,c=color1[1],marker =shape[5],s=50)
# ax16=ax.scatter3D(type1_7x, type1_7y, type1_7z,c=color1[1],marker =shape[6],s=50)
# ax17=ax.scatter3D(type1_8x, type1_8y, type1_8z,c=color1[1],marker =shape[7],s=50)
# ax18=ax.scatter3D(type1_9x, type1_9y, type1_9z,c=color1[1],marker =shape[8],s=50)
# ax19=ax.scatter3D(type1_10x, type1_10y, type1_10z,c=color1[1],marker =shape[9],s=50)

# ax20=ax.scatter3D(type2_1x, type2_1y, type2_1z,c=color1[2],marker =shape[0],s=50)
# ax21=ax.scatter3D(type2_2x, type2_2y, type2_2z,c=color1[2],marker =shape[1],s=50)
ax22=ax.scatter3D(type2_3x, type2_3y, type2_3z,c=color1[2],marker =shape[2],s=50)
ax23=ax.scatter3D(type2_4x, type2_4y, type2_4z,c=color1[2],marker =shape[3],s=50)
ax24=ax.scatter3D(type2_5x, type2_5y, type2_5z,c=color1[2],marker =shape[4],s=50)
ax25=ax.scatter3D(type2_6x, type2_6y, type2_6z,c=color1[2],marker =shape[5],s=50)
ax26=ax.scatter3D(type2_7x, type2_7y, type2_7z,c=color1[2],marker =shape[6],s=50)
ax27=ax.scatter3D(type2_8x, type2_8y, type2_8z,c=color1[2],marker =shape[7],s=50)
ax28=ax.scatter3D(type2_9x, type2_9y, type2_9z,c=color1[2],marker =shape[8],s=50)
ax29=ax.scatter3D(type2_10x, type2_10y, type2_10z,c=color1[2],marker =shape[9],s=50)

# ax30=ax.scatter3D(type3_1x, type3_1y, type3_1z,c=color1[3],marker =shape[0],s=50)
# ax31=ax.scatter3D(type3_2x, type3_2y, type3_2z,c=color1[3],marker =shape[1],s=50)
# ax32=ax.scatter3D(type3_3x, type3_3y, type3_3z,c=color1[3],marker =shape[2],s=50)
# ax33=ax.scatter3D(type3_4x, type3_4y, type3_4z,c=color1[3],marker =shape[3],s=50)
# ax34=ax.scatter3D(type3_5x, type3_5y, type3_5z,c=color1[3],marker =shape[4],s=50)
# ax35=ax.scatter3D(type3_6x, type3_6y, type3_6z,c=color1[3],marker =shape[5],s=50)
ax36=ax.scatter3D(type3_7x, type3_7y, type3_7z,c=color1[3],marker =shape[6],s=50)
ax37=ax.scatter3D(type3_8x, type3_8y, type3_8z,c=color1[3],marker =shape[7],s=50)
ax38=ax.scatter3D(type3_9x, type3_9y, type3_9z,c=color1[3],marker =shape[8],s=50)
ax39=ax.scatter3D(type3_10x, type3_10y, type3_10z,c=color1[3],marker =shape[9],s=50)

# ax40=ax.scatter3D(type4_1x, type4_1y, type4_1z,c=color1[4],marker =shape[0],s=50)
# ax41=ax.scatter3D(type4_2x, type4_2y, type4_2z,c=color1[4],marker =shape[1],s=50)
# ax42=ax.scatter3D(type4_3x, type4_3y, type4_3z,c=color1[4],marker =shape[2],s=50)
ax43=ax.scatter3D(type4_4x, type4_4y, type4_4z,c=color1[4],marker =shape[3],s=50)
ax44=ax.scatter3D(type4_5x, type4_5y, type4_5z,c=color1[4],marker =shape[4],s=50)
ax45=ax.scatter3D(type4_6x, type4_6y, type4_6z,c=color1[4],marker =shape[5],s=50)
ax46=ax.scatter3D(type4_7x, type4_7y, type4_7z,c=color1[4],marker =shape[6],s=50)
ax47=ax.scatter3D(type4_8x, type4_8y, type4_8z,c=color1[4],marker =shape[7],s=50)
# ax48=ax.scatter3D(type4_9x, type4_9y, type4_9z,c=color1[4],marker =shape[8],s=50)
# ax49=ax.scatter3D(type4_10x, type4_10y, type4_10z,c=color1[4],marker =shape[9],s=50)


# color1_1=color1[kmeans1[i]]
# color2_2=color2[kmeans2[i]]
# ax.scatter3D(UV_NEW[:,0][i], UV_NEW[:,1][i], z1[i],c=color1_1,marker ='^',s=50)
# ax.scatter3D(FL_NEW[:,0][i], FL_NEW[:,1][i], z2[i],c=color2_2,marker ='o',s=50)
# ax.title("3D scatter plot in Python")
ax.set_xlabel('Factor(1)'+' '+str("%.2f" % (final_ratio[0]*100))+"%", fontdict={'family' : 'Times New Roman', 'size'   : 16})
ax.set_ylabel('Factor(2)'+' '+str("%.2f" % (final_ratio[1]*100))+"%", fontdict={'family' : 'Times New Roman', 'size'   : 16})
ax.set_zlabel('Factor(3)'+' '+str("%.2f" % (final_ratio[2]*100))+"%", fontdict={'family' : 'Times New Roman', 'size'   : 16})
# ticks = np.arange(0, 2, 1)
# # ticks = ['UV','FL']
# ax.set_zticks(ticks)

# handles = [ax0,ax1, ax2, ax3, ax4,ax5,ax6, ax7, ax8, ax9,ax10,ax11, ax12, ax13, ax14,ax15,ax16, ax17, ax18, ax19,ax20,ax21, ax22, ax23, ax24,ax25,ax26, ax27, ax28, ax29,ax30,ax31, ax32, ax33, ax34,ax35,ax36, ax37, ax38, ax39,ax40,ax41, ax42, ax43, ax44,ax45,ax46, ax47, ax48, ax49]
# handles = [ax2, ax3, ax4,ax5,ax6, ax7, ax8, ax9,ax12, ax13, ax14,ax15,ax16, ax17, ax18, ax19,ax22, ax23, ax24,ax25,ax26, ax27, ax28, ax29, ax32, ax33, ax34,ax35,ax36, ax37, ax38, ax39, ax42, ax43, ax44,ax45,ax46, ax47, ax48, ax49]
handles = [ax7, ax8, ax9,ax10,ax11, ax12, ax22, ax23, ax24,ax25,ax26, ax27, ax28, ax29,ax36, ax37, ax38, ax39, ax43, ax44,ax45,ax46, ax47]
# ax.set_title(title, fontsize=20)
label_classtemp=['SO$_3$$^2$$^-$_60μM','SO$_3$$^2$$^-$_80μM','SO$_3$$^2$$^-$_100μM','SO$_4$$^2$$^-$_0.5μM','SO$_4$$^2$$^-$_1μM','SO$_4$$^2$$^-$_10μM','S$_2$O$_3$$^2$$^-$_10μM','S$_2$O$_3$$^2$$^-$_20μM','S$_2$O$_3$$^2$$^-$_30μM','S$_2$O$_3$$^2$$^-$_40μM','S$_2$O$_3$$^2$$^-$_50μM','S$_2$O$_3$$^2$$^-$_60μM','S$_2$O$_3$$^2$$^-$_80μM','S$_2$O$_3$$^2$$^-$_100μM','S$_2$O$_8$$^2$$^-$_50μM','S$_2$O$_8$$^2$$^-$_60μM','S$_2$O$_8$$^2$$^-$_80μM','S$_2$O$_8$$^2$$^-$_100μM','S$^2$$^-$_20μM','S$^2$$^-$_25μM','S$^2$$^-$_40μM','S$^2$$^-$_50μM','S$^2$$^-$_60μM']
ax.legend(handles=handles,labels=label_classtemp,loc='upper center', mode="expand", ncol = 6, borderaxespad = 0)
plt.savefig(savefilename)
# fig.colorbar(sctt, ax = ax, shrink = 0.6, aspect = 5)
# display the plot
plt.show()
