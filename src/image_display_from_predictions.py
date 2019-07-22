import matplotlib
import matplotlib.pyplot as plt
import numpy as np

file_path="C:/Users/karde/OpenCV/hw6/final_output/"
l=[1,2,5,9,11,40]
#Loading the predicted labels
pred_label=np.empty(shape=(6,352,1216))
for i in range(len(l)):
    pred_label[i] = np.load(file_path+'predicted'+str(l[i])+'.npy')

#Loading the ground truth labels
y_label=np.empty(shape=(6,352,1216))
for i in range(len(l)):
    y_label[i]=np.load(file_path+'actual' + str(l[i]) + '.npy')

def image_plot(prd_lbl):
    ##Function to plot the predicted image
    img=np.zeros([prd_lbl.shape[0],prd_lbl.shape[1],3])
    ind_road= np.argwhere(prd_lbl==1)
    for i in range(ind_road.shape[0]):
        img[ind_road[i,0],ind_road[i,1],:]=[255,0,255]

    ind_nonroad=np.argwhere(prd_lbl==0)
    for j in range(ind_nonroad.shape[0]):
        img[ind_nonroad[j,0],ind_nonroad[j,1],:]=[255,0,0]
    return img

def actual_img_plot(y_lbl):
    #Fuction to plot the ground_truth image from the actual label
    img = np.zeros([y_lbl.shape[0], y_lbl.shape[1], 3])
    ind_road = np.argwhere(y_lbl == 1)
    for i in range(ind_road.shape[0]):
        img[ind_road[i, 0], ind_road[i, 1], :] = [255, 0, 255]

    ind_nonroad = np.argwhere(y_lbl == 0)
    for j in range(ind_nonroad.shape[0]):
        img[ind_nonroad[j, 0], ind_nonroad[j, 1], :] = [255, 0, 0]

    ind_void = np.argwhere(y_lbl == -1)
    for k in range(ind_void.shape[0]):
        img[ind_void[k, 0], ind_void[k, 1], :] = [0, 0, 0]
    return img

#Plotting the images corresponding to the 6 labels

for m in range(6):
    img=image_plot(pred_label[m])
    img.astype('uint8')
    fig=plt.figure()
    plt.title("Predicted image")
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    fig1 = plt.figure()
    plt.title("Actual image")
    act_im=actual_img_plot(y_label[m])
    act_im.astype('uint8')
    plt.imshow(act_im)
    plt.xticks([])
    plt.yticks([])
    plt.show()
