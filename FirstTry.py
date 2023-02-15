import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import mysql.connector
import time 
import os
import sys 
from scipy.optimize import curve_fit
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages
import concurrent.futures
from pprint import pprint
############################################################
def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)
def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

############################################################
def save_multi_image(filename):
   pp = PdfPages(filename)
   fig_nums = plt.get_fignums()
   figs = [plt.figure(n) for n in fig_nums]
   for fig in figs:
      fig.savefig(pp, format='pdf')
   pp.close()
############################################################

def CalcGain(img_CCD16,NROW):#Recibe 1 imagen DE 1 CCD
    #step=int(NROW/(0.3*NROW))
    #data = np.array((img_CCD16[0:NROW:step][:]).flatten())
    data = (img_CCD16).flatten()
    #print(np.shape(data))
    a = list(range(-200,500))
    #a = list(range(-100,1000))
    fig = plt.figure()
    y,x,_ = plt.hist(data, bins = a)            
    try:
        x = (x[1:]+x[:-1])/2 
        expected = (0, 0.8, 200, 300, 0.6, 20)
        params, cov = curve_fit(bimodal, x, y, expected)
        gain = [str(params[3]-params[0])] 
        x_line = range(-200, 450, 1)
        #y_line = bimodal(x_line, params[0], params[1], params[2], params[3], params[4], params[5])
        y_line1 = gauss(x_line, params[0], params[1], params[2])
        y_line2 = gauss(x_line,params[3], params[4], params[5])
        plt.plot(x_line, y_line1, '--', color='red')
        plt.plot(x_line, y_line2, '--', color='green')
        plt.text(250, 600, f"Gain: {gain} \nmu1: {params[0]} \nsigma1: {params[1]} \nA1: {params[2]} \nmu2: {params[3]} \nsigma2: {params[4]} \nA2: {params[5]}",size=8)
        if (params[0]>-10 and params[0]<10):
            return [gain, params]
        else:
            return [-2, params]
    except:
        print("Can't Fit Model")
        return [-1]
############################################################

###Function in charge of demultiplexing
def GetSingleCCDImage(hdul,LTA_channel,ColInit,NCOL,tamy,ccdncol,NSAMP): 
    MuxedImage=hdul[LTA_channel].data
    LastCol=ColInit+(NCOL-1)*NSAMP 
    indexCol=list(range(ColInit,LastCol,NSAMP))
    DeMuxedImage=np.array(MuxedImage[:, indexCol],dtype=float)
    for p in range(tamy):
        Offset=np.mean(DeMuxedImage[p,(NCOL-int(NCOL-ccdncol/2)):NCOL])
        DeMuxedImage[p,:]=DeMuxedImage[p,:]-Offset
    return DeMuxedImage 

############################################################################################
if __name__ == '__main__': 
    Start_Time = time.time() ###Variable for time measurement
    args = sys.argv[1:]
    inputFile = str(args[0])
    #inputFile = str(sys.argv[1])
    baseName=os.path.splitext(inputFile)[0]
    st = time.time()
    img_CCD16=fits.HDUList([]) #Se crearan nro_imagenes*nsamp/16 (Ej: 5*112/6=35)
    hdulist = fits.open(inputFile)
    NAXIS1=int(hdulist[4].header['NAXIS1']) #Size X
    NAXIS2=int(hdulist[4].header['NAXIS2']) #Size Y
    NSAMP=int(hdulist[4].header['NSAMP']) #112
    NCOL=int(hdulist[4].header['NCOL'])
    CCDNCOL=int(hdulist[4].header['CCDNCOL'])
    scidata = hdulist[4].data
    tamxpimg=int(NAXIS1/NSAMP)
    div=16 #Variable multiplo de 2^n 
    LTA_channel=4
    CCDinMCM=16 
    gains=[]
    datos=np.zeros(shape=(NAXIS1,NAXIS2),dtype=float)
    img_CCD16=fits.HDUList([])
    Map=[1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16]
    Map=[1,5,12,16,2,9,13,6,3,10,14,7,4,11,15,8]
    for N in range(int(NSAMP/16)):  #Se recorre nsamp/16=7 veces por cada imagen    
    #for N in range(1):
        for CCD in Map:                               #Se recorre 16 veces para ir agarrando 16 CCD 
            img_parcial=GetSingleCCDImage(hdulist,LTA_channel,CCD-1+CCDinMCM*N,NCOL,NAXIS2,CCDNCOL,NSAMP)
            ############################################################
            if args[1]==str("yes"):
                gain=CalcGain(img_parcial,NAXIS2) #Lineas encargada del histograma.
                print("CCD ",(N*16+CCD)+1," ",gain)
                gains.append(gain)
            ############################################################
            img_CCD16.append(fits.ImageHDU(img_parcial))
        #Proceso de guardado
        Directory_Demux="Demuxed_"+baseName+"/"
        if not os.path.exists(Directory_Demux):
            os.makedirs(Directory_Demux)
        SaveName=str(Directory_Demux+"MCM"+str(N+1)+"_Demuxed_"+baseName+"_PROC.fits") 
        img_CCD16.writeto(SaveName,overwrite=True)
        img_CCD16.clear()
        print((N+1)*10,"% done...")
    hdulist.close()
    ############################################################
    if args[1]==str("yes"):
        filename = Directory_Demux+"Histogram_curvefit_pair_FULLimage_params_V2.pdf" #Lineas encargadas del pdf de los histogramas.
        save_multi_image(filename)
    ############################################################
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')