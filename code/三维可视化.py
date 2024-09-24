from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
# Creating random dataset
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

numclass=5
df = pd.read_excel('test.xlsx')

color1 = ['r', 'g', 'b', 'c', 'y']
color2 = ['m', 'k', '#ff00ff', '#99ff66','#8C7853']
df.drop(axis=0,index=0, inplace=True)
# df.drop(axis=1,columns=0, inplace=True)
temp=df.values
label=temp[:,0]
temp=np.delete(temp,0,axis=1)
UV=temp[:,0:3]
FL=temp[:,3:6]

pca = PCA(n_components=2)	#实例化
UV_NEW = pca.fit_transform(UV)			#拟合模型

pca1=PCA(n_components=2)
FL_NEW = pca1.fit_transform(FL)

UV_ratio=pca.explained_variance_ratio_
FL_ratio=pca1.explained_variance_ratio_

x1=UV_NEW[:,0]
y1=UV_NEW[:,1]
z_1= np.zeros(len(x1))
# z1.fill(1)

x2=FL_NEW[:,0]
y2=FL_NEW[:,1]
z_2= np.ones(len(x2))

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
type1_x = []
type1_y = []
type2_x = []
type2_y = []
type3_x = []
type3_y = []
type4_x = []
type4_y = []

type5_x = []
type5_y = []
type6_x = []
type6_y = []
type7_x = []
type7_y = []
type8_x = []
type8_y = []
type9_x = []
type9_y = []

# for j in range(numclass):
#     if
for i in range(len(z_1)):
    if label[i]=='SO32-':
        type0_x.append(UV_NEW[:,0][i])
        type0_y.append(UV_NEW[:, 1][i])

        type5_x.append(FL_NEW[:,0][i])
        type5_y.append(FL_NEW[:, 1][i])
    if label[i]=='SO42-':
        type1_x.append(UV_NEW[:,0][i])
        type1_y.append(UV_NEW[:, 1][i])

        type6_x.append(FL_NEW[:,0][i])
        type6_y.append(FL_NEW[:, 1][i])
    if label[i]=='S2O32-':
        type2_x.append(UV_NEW[:,0][i])
        type2_y.append(UV_NEW[:, 1][i])

        type7_x.append(FL_NEW[:,0][i])
        type7_y.append(FL_NEW[:, 1][i])
    if label[i]=='S2O82-':
        type3_x.append(UV_NEW[:,0][i])
        type3_y.append(UV_NEW[:, 1][i])

        type8_x.append(FL_NEW[:,0][i])
        type8_y.append(FL_NEW[:, 1][i])
    if label[i]=='S2-':
        type4_x.append(UV_NEW[:,0][i])
        type4_y.append(UV_NEW[:, 1][i])

        type9_x.append(FL_NEW[:,0][i])
        type9_y.append(FL_NEW[:, 1][i])

z0= np.zeros(len(type0_x))
z1= np.zeros(len(type1_x))
z2= np.zeros(len(type2_x))
z3= np.zeros(len(type3_x))
z4= np.zeros(len(type4_x))
z5= np.ones(len(type5_x))
z6= np.ones(len(type5_x))
z7= np.ones(len(type7_x))
z8= np.ones(len(type8_x))
z9= np.ones(len(type9_x))



ax0=ax.scatter3D(type0_x, type0_y, z0,c=color1[0],marker ='^',s=50)
ax1=ax.scatter3D(type1_x, type1_y, z1,c=color1[1],marker ='^',s=50)
ax2=ax.scatter3D(type2_x, type2_y, z2,c=color1[2],marker ='^',s=50)
ax3=ax.scatter3D(type3_x, type3_y, z3,c=color1[3],marker ='^',s=50)
ax4=ax.scatter3D(type4_x, type4_y, z4,c=color1[4],marker ='^',s=50)

ax5=ax.scatter3D(type5_x, type5_y, z5,c=color2[0],marker ='o',s=50)
ax6=ax.scatter3D(type6_x, type6_y, z6,c=color2[1],marker ='o',s=50)
ax7=ax.scatter3D(type7_x, type7_y, z7,c=color2[2],marker ='o',s=50)
ax8=ax.scatter3D(type8_x, type8_y, z8,c=color2[3],marker ='o',s=50)
ax9=ax.scatter3D(type9_x, type9_y, z9,c=color2[4],marker ='o',s=50)

# color1_1=color1[kmeans1[i]]
# color2_2=color2[kmeans2[i]]
# ax.scatter3D(UV_NEW[:,0][i], UV_NEW[:,1][i], z1[i],c=color1_1,marker ='^',s=50)
# ax.scatter3D(FL_NEW[:,0][i], FL_NEW[:,1][i], z2[i],c=color2_2,marker ='o',s=50)
# plt.title("3D scatter plot in Python")
ax.set_xlabel('Factor(1)'+' '+str("%.2f" % (UV_ratio[0]*100))+"%"+'(UV)'+' '+str("%.2f" % (FL_ratio[0]*100))+"%"+'(FL)', fontdict={'family' : 'Times New Roman', 'size'   : 16})
ax.set_ylabel('Factor(2)'+' '+str("%.2f" % (UV_ratio[1]*100))+"%"+'(UV)'+' '+str("%.2f" % (FL_ratio[1]*100))+"%"+'(FL)', fontdict={'family' : 'Times New Roman', 'size'   : 16})
ax.set_zlabel('Method', fontweight ='bold', fontdict={'family' : 'Times New Roman', 'size'   : 16})
ticks = np.arange(0, 1.1, 1)
# ticks = ['UV','FL']
ax.set_zticks(ticks)
ax.set_zticks(ticks)
handles = [ax0,ax1, ax2, ax3, ax4,ax5, ax6, ax7, ax8,ax9]
labels=['UV_SO$_3$$^2$$^-$','UV_SO$_4$$^2$$^-$','UV_S$_2$O$_3$$^2$$^-$','UV_S$_2$O$_8$$^2$$^-$','UV_S$^2$$^-$','FL_SO$_3$$^2$$^-$','FL_SO$_4$$^2$$^-$','FL_S$_2$O$_3$$^2$$^-$','FL_S$_2$O$_8$$^2$$^-$','FL_S$^2$$^-$']
ax.legend(handles=handles,labels=labels, mode="expand", ncol = 5, borderaxespad = 0)
# fig.colorbar(sctt, ax = ax, shrink = 0.6, aspect = 5)
# display the plot
plt.show()