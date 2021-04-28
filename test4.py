import skimage
import numpy as np
from PIL import Image, ImageFont, ImageDraw,  ImageFilter
import random
import math


im1=Image.open("1.jpg")
img_rgb1=im1.convert("RGB")
im2=Image.open("2.jpg")
img_rgb2=im2.convert("RGB")
im3=Image.open("3.jpg")
img_rgb3=im3.convert("RGB")
im4=Image.open("4.jpg")
img_rgb4=im4.convert("RGB")
im5=Image.open("5.jpg")
img_rgb5=im5.convert("RGB")
def images(img_rgb):
    width, height=img_rgb.size
    porog=124
    for i in range(width):
        for j in range(height):
           r = img_rgb.getpixel((i, j))[0]
           g = img_rgb.getpixel((i, j))[1]
           b = img_rgb.getpixel((i, j))[2]
           if r>porog or g>porog or b>porog:
               r=255
               g=255
               b=255
           else:
               r = 0
               g = 0
               b = 0
           img_rgb.putpixel((i, j), (r,g,b))



out1 = np.array([1, 0, 0, 0])
out2 = np.array([0,1,0,0])
out3 = np.array([0,0,1,0])
out4 = np.array([0,0,0,1])

massiv_s_ves1=[]
massiv_s_ves2=[]
massiv_s_ves3=[]

def education(img, out):
    vector=np.zeros(img.width*img.height)
    y=0
    for i in range(img.width):
        for j in range(img.height):
            br=img.getpixel((i,j))[0]
            vector[y]=(br/255)
            y+=1
    # print("vector", vector)  # !

    #веса
    raz=img.height*img.width
    # print(raz)
    e=2.71
    #для первого слоя задаем веса
    # weights = np.random.random((raz,raz)) #матрица с весами
    weights=np.random.uniform(-0.5, 0.5, [raz, raz])
    layer1=weights.dot(vector)  #переумнажение весов и вектора
    sigmoid1=np.zeros(raz) #сигмоиды для первого слоя

    for i in range(raz):
        sig1=1/(1+math.pow(e,layer1[i]*(-1)))
        sigmoid1[i] = sig1
    # print("vesa",weights) #!
    # print("sigmoid", sigmoid1) #!

    #для второго слоя задаем веса
    weights_2 = np.random.uniform(-0.5, 0.5, [raz, raz]) #матрица с весами
    layer2=weights_2.dot(sigmoid1)  #переумнажение весов и вектора на втором слое
    # print("sloi2", layer2) #!
    sigmoid2=np.zeros(raz) #сигмоиды для второго слоя
    for i in range(raz):
        sig2=1/(1+math.pow(e,layer2[i]*(-1)))
        sigmoid2[i] = sig2
    # print("d sigmod2", len(sigmoid2)) #!
    # print("vesa 2go sloia", weights_2)

    #для третьего слоя задаем веса
    l3=0
    weights_3 = np.random.uniform(-0.5, 0.5, [4, raz]) #матрица с весами
    # print("dlina w3",len(weights_3))
    layer3=np.zeros(len(weights_3))
    for i in range(len(weights_3)):
        l3 = 0
        for j in range(len(sigmoid2)):
            l_3=sigmoid2[j]*weights_3[i][j]
            l3+=l_3
        layer3[i]=l3
    # print("sloi3", layer3)
    sigmoid3=np.zeros(4) #сигмоиды для третьего слоя
    for i in range(len(layer3)):
        sig3=1/(1+math.pow(e,layer3[i]*(-1)))
        sigmoid3[i] = sig3
    # print("sigmoid3", sigmoid3)
    # print("vesa 3go sloia",weights_3)



    #ошибка
    delta=np.zeros(4)
    for i in range(len(sigmoid3)):
        for_d=out[i]- sigmoid3[i]
        delta[i]=for_d
    print("oshibka1", delta) #!
    # S+=sum(delta) #подсчет суммы


    #находим сигнал ошибки
    delta_2=np.zeros(len(sigmoid2)) #ошибка для вторго слоя
    for i in range(len(sigmoid2)):
        d_2=0
        for j in range(len(weights_3)):
            d2=delta[j]*weights_3[j][i]
            d_2+=d2
        delta_2[i]=d_2
    # print("d2", delta_2)
    # print("dlina d2", len(delta_2))#!
    delta_1=np.dot(weights_2,delta_2) #ошибка для первого слоя
    # print("d1", delta_1)#!


    #корректируем коэфф
    n=0.001 #скорость обучения
    for_derivative1=np.zeros(len(sigmoid1))
    for i in range(len(sigmoid1)):
        for_derivative1[i]=1-sigmoid1[i]
        # print("s", sigmoid1[i])
        # print("1-s",for_derivative1[i])

    #производные
    derivative1=np.zeros(len(sigmoid1))
    for i in range(len(derivative1)):
        derivative1[i]=for_derivative1[i]*sigmoid1[i]

    #считаем новые коэфф
    for_new_kof=np.zeros(len(derivative1))
    for i in range(len(derivative1)):
        for_new_kof[i]=n*delta_1[i]*derivative1[i]*vector[i]
    # print("chast new kof", for_new_kof)

    x=0
    for i in range(len(weights)):
        for j in range(len(weights)):
            weights[j][i]=weights[j][i]+for_new_kof[x]
        x+=1
    # print("новые веса", weights) #новые веса для первого слоя

    #для второго слоя
    for_derivative2=np.zeros(len(sigmoid2))
    for i in range(len(sigmoid2)):
        for_derivative2[i]=1-sigmoid2[i]


    #производные
    derivative2=np.zeros(len(sigmoid2))
    for i in range(len(derivative2)):
        derivative2[i]=for_derivative2[i]*sigmoid2[i]

    #считаем новые коэфф
    for_new_kof2=np.zeros(len(derivative2))
    for i in range(len(derivative2)):
        for_new_kof2[i]=n*delta_2[i]*derivative2[i]*sigmoid1[i]
    # print("chast new kof", for_new_kof2)

    x=0
    for i in range(len(weights_2)):
        for j in range(len(weights_2)):
            weights_2[j][i]=weights_2[j][i]+for_new_kof2[x]
        x+=1
    # print("новые веса второго слоя", weights_2) #новые веса для второго слоя

    #для третьего слоя
    for_derivative3=np.zeros(len(sigmoid3))
    for i in range(len(sigmoid3)):
        for_derivative3[i]=1-sigmoid3[i]


    #производные
    derivative3=np.zeros(len(sigmoid3))
    for i in range(len(derivative3)):
        derivative3[i]=for_derivative3[i]*sigmoid3[i]

    #считаем новые коэфф
    for_new_kof3=np.zeros(len(derivative3))
    for i in range(len(derivative3)):
        for_new_kof3[i]=n*delta[i]*derivative3[i]*sigmoid3[i]
    # print("dlina new kof3", len(for_new_kof3))
    # print("dlina vesov 3", len(weights_3))

    x=0
    for i in range(len(weights_3)):
        for j in range(len(weights_3)):
            weights_3[j][i]=weights_3[j][i]+for_new_kof3[x]
        x+=1
    # print("новые веса третьего слоя", weights_3) #новые веса для третьего слоя

    #новые коэфф посчитаем и потом заново пройдемся
    massiv_s_ves1.append(weights)
    massiv_s_ves2.append(weights_2)
    massiv_s_ves3.append(weights_3)
    layer1=weights.dot(vector)  #переумнажение весов и вектора
    sigmoid1=np.zeros(raz) #сигмоиды для первого слоя
    for i in range(raz):
        sig1=1/(1+math.pow(e,layer1[i]*(-1)))
        sigmoid1[i] = sig1

    layer2=weights_2.dot(sigmoid1)  #переумнажение весов и вектора на втором слое
    # print("sloi2", layer2) #!
    sigmoid2=np.zeros(raz) #сигмоиды для второго слоя
    for i in range(raz):
        sig2=1/(1+math.pow(e,layer2[i]*(-1)))
        sigmoid2[i] = sig2
    # print("s2", sigmoid2)
    # print("w3", weights_3)

    layer3=np.zeros(len(weights_3))
    for i in range(len(weights_3)):
        l3 = 0
        for j in range(len(sigmoid2)):
            l_3=sigmoid2[j]*weights_3[i][j]
            l3+=l_3
        layer3[i]=l3
    # print("sloi3", layer3)
    sigmoid3=np.zeros(4) #сигмоиды для третьего слоя
    for i in range(len(layer3)):
        sig3=1/(1+math.pow(e,layer3[i]*(-1)))
        sigmoid3[i] = sig3
    # print("s3",sigmoid3)

    # #ошибка
    delta=np.zeros(4)
    for i in range(len(sigmoid3)):
        for_d=out[i]- sigmoid3[i]
        delta[i]=for_d
    print("oshibka2", delta) #!
    # # S+=sum(delta) #подсчет суммы

images(img_rgb1)
education(img_rgb1,out1)

images(img_rgb2)
education(img_rgb2,out2)
#
images(img_rgb3)
education(img_rgb3,out3)

images(img_rgb4)
education(img_rgb4,out4)

# print("m1",massiv_s_ves1)
massiv=[]


def identification(img,y,out):
    vector=np.zeros(img.width*img.height)
    raz=img.width*img.height
    e=2.71
    weights=massiv_s_ves1[y]
    layer1 = weights.dot(vector)  # переумнажение весов и вектора
    # print("sloi1", layer1)  # !
    sigmoid1 = np.zeros(raz)  # сигмоиды для первого слоя
    for i in range(raz):
        sig1 = 1 / (1 + math.pow(e, layer1[i]*(-1)))
        sigmoid1[i] = sig1
    # print("s1",sigmoid1)

    weights_2=massiv_s_ves2[y]
    layer2 = weights_2.dot(sigmoid1)  # переумнажение весов и вектора на втором слое
    # print("sloi2", layer2) #!
    sigmoid2 = np.zeros(raz)  # сигмоиды для второго слоя
    for i in range(raz):
        sig2 = 1 / (1 + math.pow(e, layer2[i]*(-1)))
        sigmoid2[i] = sig2
    # print("s2", sigmoid2)
    l3 = 0
    weights_3 = massiv_s_ves3[y]
    layer3 = np.zeros(len(weights_3))
    for i in range(len(weights_3)):
        l3 = 0
        for j in range(len(sigmoid2)):
            l_3 = sigmoid2[j] * weights_3[i][j]
            l3 += l_3
        layer3[i] = l3
    print("sloi3", layer3)
    sigmoid3 = np.zeros(4)  # сигмоиды для третьего слоя
    for i in range(len(layer3)):
        sig3 = 1 / (1 + math.pow(e, layer3[i]*(-1)))
        sigmoid3[i] = sig3
    print("sigmoid3", sigmoid3)
    delta = np.zeros(4)
    for i in range(len(sigmoid3)):
        for_d = out[i] - sigmoid3[i]
        delta[i] = for_d
    print("delta kon",delta)
    max_s=delta[y]
    massiv.append(max_s)

images(img_rgb1)
identification(img_rgb1,0,out1)
identification(img_rgb1,1,out2)
identification(img_rgb1,2,out3)
identification(img_rgb1,3,out4)

print(massiv)
print("Это человек 1 на", massiv[0]*100,"%")
print("Это человек 2 на", massiv[1]*100,"%")
print("Это человек 3 на", massiv[2]*100,"%")
print("Это человек 4 на", massiv[3]*100,"%")


