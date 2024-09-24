from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
# Creating random dataset
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


df = pd.read_excel('UV+FL单一S浓度线性得分图.xlsx')
savefilename='final_fig/UV+FL单一S浓度线性得分图S2-_1.png'
UV_weight=1
color1 = ['r', 'g', 'b', 'c', 'y']
color2 = ['m', 'k', '#ff00ff', '#99ff66','#8C7853']
# df.drop(axis=0,index=0, inplace=True)
# df.drop(axis=1,columns=0, inplace=True)
title=df.columns[0]
temp=df.values
label=temp[:,0]
temp=np.delete(temp,0,axis=1)
UV=temp[:,0:3]
FL=temp[:,3:6]

final=UV_weight*UV+(1-UV_weight)*FL

pca = PCA(n_components=3)	#实例化
final_NEW = pca.fit_transform(final)			#拟合模型


final_ratio=pca.explained_variance_ratio_


x1=final_NEW[:,0]
y1=final_NEW[:,1]
z_1=final_NEW[:,2]
# z1.fill(1)


# y_kmeans = kmeans.fit_predict(X)
# Creating figure
fig = plt.figure(figsize = (8, 8))
ax = plt.axes(projection ="3d")
# Add x, and y gridlines for the figure
# ax.grid(b = True, color ='blue',linestyle ='-.', linewidth = 0.5,alpha = 0.3)
# Creating the color map for the plot
# my_cmap = plt.get_cmap('hsv')
# Creating the 3D plot
# sctt = ax.scatter3D(x, y, z,alpha = 0.8,c = (x + y + z),cmap = my_cmap,marker ='^')
type0_x = []
type0_y = []
type0_z = []
type1_x = []
type1_y = []
type1_z = []
type2_x = []
type2_y = []
type2_z = []
type3_x = []
type3_y = []
type3_z = []
type4_x = []
type4_y = []
type4_z = []


# for j in range(numclass):
#     if
for i in range(len(z_1)):
    if label[i]=='20μM':
        type0_x.append(final_NEW[:, 0][i])
        type0_y.append(final_NEW[:, 1][i])
        type0_z.append(final_NEW[:, 2][i])

    if label[i]=='40μM':
        type1_x.append(final_NEW[:, 0][i])
        type1_y.append(final_NEW[:, 1][i])
        type1_z.append(final_NEW[:, 2][i])

    if label[i]=='60μM':
        type2_x.append(final_NEW[:, 0][i])
        type2_y.append(final_NEW[:, 1][i])
        type2_z.append(final_NEW[:, 2][i])

    if label[i]=='80μM':
        type3_x.append(final_NEW[:, 0][i])
        type3_y.append(final_NEW[:, 1][i])
        type3_z.append(final_NEW[:, 2][i])

    if label[i]=='100μM':
        type4_x.append(final_NEW[:, 0][i])
        type4_y.append(final_NEW[:, 1][i])
        type4_z.append(final_NEW[:, 2][i])


z0= np.ones(len(type0_x))
z1= np.ones(len(type1_x))
z2= np.ones(len(type2_x))
z3= np.ones(len(type3_x))
z4= np.ones(len(type4_x))




ax0=ax.scatter3D(type0_x, type0_y, type0_z,c=color1[0],marker ='o',s=50)
ax1=ax.scatter3D(type1_x, type1_y, type1_z,c=color1[1],marker ='o',s=50)
ax2=ax.scatter3D(type2_x, type2_y, type2_z,c=color1[2],marker ='o',s=50)
ax3=ax.scatter3D(type3_x, type3_y, type3_z,c=color1[3],marker ='o',s=50)
ax4=ax.scatter3D(type4_x, type4_y, type4_z,c=color1[4],marker ='o',s=50)


# color1_1=color1[kmeans1[i]] 
# color2_2=color2[kmeans2[i]]
# ax.scatter3D(UV_NEW[:,0][i], UV_NEW[:,1][i], z1[i],c=color1_1,marker ='^',s=50)
# ax.scatter3D(FL_NEW[:,0][i], FL_NEW[:,1][i], z2[i],c=color2_2,marker ='o',s=50)
# plt.title("3D scatter plot in Python")
ax.set_xlabel('Factor(1)'+' '+str("%.2f" % (final_ratio[0]*100))+"%", fontdict={'family' : 'Times New Roman', 'size'   : 16})
ax.set_ylabel('Factor(2)'+' '+str("%.2f" % (final_ratio[1]*100))+"%", fontdict={'family' : 'Times New Roman', 'size'   : 16})
ax.set_zlabel('Factor(3)'+' '+str("%.2f" % (final_ratio[2]*100))+"%", fontdict={'family' : 'Times New Roman', 'size'   : 16})
# ax.set_zlabel('Method', fontweight ='bold', fontdict={'family' : 'Times New Roman', 'size'   : 16})
# ticks = np.arange(0, 2, 1)
# # ticks = ['UV','FL']
# ax.set_zticks(ticks)

handles = [ax0,ax1, ax2, ax3, ax4]
labels=['20μM','40μM','60μM','80μM','100μM']
ax.set_title(title, fontsize=20)
ax.legend(handles=handles,labels=labels, mode="expand", ncol = 5, borderaxespad = 0)
plt.savefig(savefilename)
# fig.colorbar(sctt, ax = ax, shrink = 0.6, aspect = 5)
# display the plot
plt.show()