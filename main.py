import numpy as np
import matplotlib.pyplot as plt
import wfdb
import statistics
from sklearn.linear_model import LinearRegression
import pandas as pd
from scipy.signal import resample
import json
import scipy
import json
import math
import matplotlib.patches as patches


def interpolazione_lineare(y):
    '''
    Parametri: un segnale y
    Restituisce: il segnale interpolato 
    '''
    
    df = pd.DataFrame(y, columns=['sig'])

    # Sostituisci i valori 0 con NaN per facilitare l'interpolazione
    df['sig'] = df['sig'].replace(0, np.nan)

    # Applica l'interpolazione lineare
    df['interpolato'] = df['sig'].interpolate(method='linear')

    df['interpolato'] = df['interpolato'].bfill().ffill()

    
    return np.array(df['interpolato'])

def trova_intervalli_zeri(arr):
    intervalli = []
    inizio = None
    
    for i, valore in enumerate(arr):
        if valore == 0:
            if inizio is None:
                inizio = i  # Inizia un nuovo intervallo di zeri
        else:
            if inizio is not None:
                intervalli.append((inizio, i - 1))  # Chiudi l'intervallo corrente
                inizio = None

    if inizio is not None:  # Controlla se c'è un intervallo di zeri alla fine
        intervalli.append((inizio, len(arr) - 1))
        
    return intervalli

def interpolazione_lineare_UC(UC,fhr):
    intervals=trova_intervalli_zeri(fhr)
    uc_copy=UC.copy()
    
    mean=np.mean(UC)
    for interval in intervals:
        start=interval[0]-1
        end=interval[1]+1
        if end>=len(UC):
            end=len(UC)-1
        if start<0:
            start=0
        if UC[start]!=0 and UC[end]!=0:
            
            seg=interpolazione_lineare(UC[start:end])
            uc_copy[start:end]=seg
        else:
            
            for i in range(interval[0],interval[1]):
                uc_copy[i]=mean
            
            
        
    return uc_copy
        
def getConsecutiveNumbers(arr):
    count=0
    result=0
    num=arr[0]
    for i in range(0,len(arr)):
        
        if arr[i]!=num or i==len(arr)-1:
            if count>(len(arr)/100)*10:
                result+=count            
            count=1
            num=arr[i]
        else:
            count+=1
       
    return result    

#prende i dati salvati in ./dataset e li filtra eliminando casi non idonei all' elaborazione dei dati
def crea_dataset():
    '''Restituisce: il numero di positivi in tutto il dataset'''
    arr=[]
    for i in range(1001,1507):
        arr.append(i)
    for i in range(2001,2047):
        arr.append(i)
    d=dict()
    positivi=[]
    for n in arr:
        name=f'dataset/{n}'
        top = wfdb.rdrecord(name) 
        sig=top.p_signal
        wfdb.rdheader(name)
        data=top.__dict__['comments'][2].split()[1]
        
    
        fhr = top.p_signal[:, 0]  # Segnale del fetal heart rate (FHR)
        UC = top.p_signal[:, 1]  # Intensità delle contrazioni uterine
        fhr_red=fhr[len(fhr)-7200:len(fhr)]
        UC_red=UC[len(UC)-7200:len(UC)]
        fhr_red=np.array(fhr_red)
        UC_red=np.array(UC_red)
        
        
        zeros=np.count_nonzero(fhr_red==0)
        
        nums=getConsecutiveNumbers(fhr_red)
        nums_1= getConsecutiveNumbers(UC_red)
        zeros_1=np.count_nonzero(UC_red==0)
        all_zeros = not np.any(UC_red)
        if zeros >= len(fhr_red)/2 or nums>=len(fhr_red)/2 or all_zeros or nums_1==len(UC_red)-1:
            continue
        i_UC=interpolazione_lineare_UC(UC_red,fhr_red)
        i_fhr=interpolazione_lineare(fhr_red)

        
        i_nums=getConsecutiveNumbers(i_fhr)
        if i_nums>=len(fhr_red)/1.5 :
            continue
        elif not np.any(np.array(fhr_red)) :
            continue
        
        a=fhr_red.tolist()
        b=UC_red.tolist()
        if float(data)<7.05:
            positivi.append({'fhr':a,'UC':b, 'pH':data, 'number': n})
            
        
        d[f'{n}']={'fhr':a,'UC':b, 'pH':data, 'number': n}
    with open('./data.json','w') as f:
        json.dump(d,f)
    
    return positivi
        


#carica il dataset
def load_dataset():
    '''Restituisce: un dizionario contenete tutti i dati'''
    with open('data.json', 'r') as f:
        data = json.load(f)
    return data

    
def avg_signal(sig):
    '''Restituisce: un segnale mediato su periodi di 2.5s
    Input: un segnale campionato a 4Hz'''
    a=sig.copy()
    i=0
    mean_arr=[]
    for s in range(0,len(a)):
        mean_arr.append(a[s])
        i+=1
        if i==10:
            a[s]=np.mean(mean_arr)
            i=0
            mean_arr= []
        else:
            a[s]=0
    return a



def find_peak(hist,bin_edges):
    '''Restituisce: il valore del picco relativo di BPM
    Input: un istogramma contenente frequenze di BPM'''
    total_area = np.sum(hist)
    scanned_area = 0
    for i in range(len(hist)-1, -1, -1):
        scanned_area += hist[i]
        # Verifica se il picco supera almeno 1/8 dell'area totale
        if scanned_area >= total_area / 8:
            # Verifica se il valore supera le cinque classi successive
            if all(hist[i] > hist[i-1:i-6]):
                return bin_edges[i]
    return None
        
def backward_dummy(sig,P):
    '''Restituisce: lo stesso segnale eccetto al primo valore che è stato adattato per preparare il segnale al filtraggio
    Input: un segnale FHR e un picco P'''
    copy=sig.copy()
    for i in range(len(copy),1):
        if abs((60000/copy[i])-(60000/P))<=60:
            copy[0]=0.95*copy[0] + 0.05 * copy[i]
    return copy


def forward_pass(sig,P):
    '''Restituisce: il segnale filtrato
    Input: un segnale fhr e un picco P'''
    copy=sig.copy()
    for i in range(1,len(copy)):
        if abs((60000/copy[i])-(60000/P))<=60:
            copy[i]=0.95*copy[i-1] + 0.05*copy[i]
        else:
            copy[i]=copy[i-1]
    return copy


def backward_pass(sig,P):
    '''Restituisce: il segnale filtrato
    Input: un segnale fhr e un picco P'''
    copy=sig.copy()
    for i in range(len(copy)-1,1):
        copy[i]=0.95*copy[i+1]+0.05*copy[i]
    return copy


def baseline(fhr):
    '''Restituisce: la baseline del segnale
    Input: un segnale FHR
    '''
    copy_fhr=fhr.copy()
    sig=avg_signal(copy_fhr)
    int_sig=interpolazione_lineare(sig)
    mean=np.mean(fhr)
  
    A_filtered = int_sig[(int_sig >= 100) & (int_sig <= 200)]

# Crea la distribuzione di frequenza con larghezza di classe di 1 ms
    bins = np.arange(100, 201, 1)
    hist, bin_e = np.histogram(A_filtered, bins=bins)
    P = find_peak(hist,bin_e)
    copy=backward_dummy(int_sig,P)
    fw=forward_pass(copy,P)
    bw=backward_pass(fw,P)
    upper=bw.copy()
    upper=[x + 20 for x in upper]
    lower=bw.copy()
    lower = [x - 20 for x in lower]

    L=20
    U=[20,15,10,5]
    a=int_sig.copy()
    for j,u in enumerate(U):
        for i in range(0,len(bw)-1):
            
            if a[i]>bw[i]+u or a[i]<bw[i]-L:
                a[i]=bw[i]
        
        copy=backward_dummy(a,P)
        fw=forward_pass(copy,P)
        bw=backward_pass(fw,P)
    
    return bw


def trova_segmenti_inferiori_baseline(fhr, baseline):
    '''Restituisce: indici dei segmenti sotto livello baseline
    Input: segnale fhr e segnale baseline'''
    indici_segmenti = []
    inizio_segmento = None

    # Itera attraverso il segnale del battito cardiaco
    for i, valore in enumerate(fhr):
        if valore < baseline[i]:
            if inizio_segmento is None:
                inizio_segmento = i  # Inizia un nuovo segmento
        else:
            if inizio_segmento is not None:
                indici_segmenti.append((inizio_segmento, i - 1))  # Termina il segmento
                inizio_segmento = None
    
    # Gestisci il caso in cui il segnale finisca con un segmento inferiore alla baseline
    if inizio_segmento is not None:
        indici_segmenti.append((inizio_segmento, len(fhr) - 1))
    
    return indici_segmenti


def find_max_excursion(fhr,baseline):
    m=0
    for i in range(0,len(fhr)-1):
        if fhr[i]-baseline[i]<0:
            if abs(fhr[i]-baseline[i])>m:
                m=abs(fhr[i]-baseline[i])
    return m


def exclude_seconds(segments,fhr,baseline):
    s=[]
    for seg in segments:
        fhr_segment=np.array(fhr[seg[0]:seg[1]])
        baseline_segment=np.array(baseline[seg[0]:seg[1]])
        max_excursion=find_max_excursion(fhr_segment,baseline_segment)
        if (len(fhr_segment)<720 and  len(fhr_segment)>240 and max_excursion>10) or ( len(fhr_segment)<480 and len(fhr_segment)>120 and max_excursion>20):
            s.append(seg)            
    return s


def find_min_index(start,end,fhr):
    m=1000
    index=0
    for i,e in enumerate(fhr[start:end]):
        if e<m:
            index=i
            m=e
      
    return index+start
        
def curvearea(segs,fhr,bl):
    area=[]
    if len(segs)==0:
        return 0
    for seg in segs:
        r=range(seg[0],seg[1])
        a=fhr[seg[0]:seg[1]]
        n=bl[seg[0]:seg[1]]
        s=np.trapz(a,r)
        t=np.trapz(n,r)
        res=t-s
        area.append(res)
        me=0
        if len(area)>0:
            me=np.mean(area)
    return me


def slope_parameters(fhr,segments,bl):
    x=[] #array contenente la durata delle parti del segnale senza drop
    y=segments.copy() #array contenente la durata dei drop
    z=[] #array contenente la profondità dei drop
    for i,seg in enumerate(segments):
        if i==0 and seg[0]!=0: #se siamo a inizio segnale
            x.append((0,seg[0])) #aggiunge a x la parte tra l' inizio e il primo drop
        elif i==0 and seg[0]==0: #se corrispondono inizio segnale e inizio drop
            num=fhr[seg[0]]-min(fhr[seg[0]:seg[1]]) #cerca il punto del drop con massima escursione da bl
            z.append(num)
            continue
        elif i<len(fhr)-1: #se siamo a fine segnale
            x.append((segments[i-1][1],seg[0])) #aggiunge la parte senza drop seguente all' ultimo drop
        #num=find_max_excursion(fhr[seg[0]:seg[1]],bl[seg[0]:seg[1]]) #cerca il punto del drop con massima escursione da bl
        num=fhr[seg[0]]-min(fhr[seg[0]:seg[1]])
        z.append(num)
    if x and x[-1][1]<len(fhr)-1  :
        x.append((x[-1][1],len(fhr)-1))
    count=0
    slope_arr=[]
   
    
    for seg in segments: #itera su i segmenti con i drop
        y1=fhr[seg[0]]
        x1=seg[0]
        y2=min(fhr[seg[0]:seg[1]])
        x2=find_min_index(seg[0]+1,seg[1],fhr)
        slope_arr.append((y2-y1)/(x2-x1)) #calcola la slope del drop come coefficiente angolare della retta passante per il punto iniziale del drop e il punto più basso del drop
        
    slope=0
    if len(slope_arr)>0:
        slope=np.mean(slope_arr) #fa la media della slope
    mean_h=0
    if len(z)>0:
        mean_h=np.mean(z)
    return x,y,z,slope,mean_h 
        


def statistics(X,Y,Z):
    total_X=0
    total_Y=0
    total_area=[]
    for x in X: #calcola il tempo totale passato a livello bl(ovvero il tempo totale passato senza drop)
        total_X+=x[1]-x[0] #valore assoluto della differenza degli indici
   
    total_Y=sum(end-start for start,end in Y) #valore assoluto della differenza degli indici
    for i,z in enumerate(Z): #calcola area totale della decelerazione
        total_area.append((abs(Y[i][0]-Y[i][1])*z)/2) # calcola l' area della decelerazione approssimandola con un tiangolo 
    res=np.sum(total_area) #media delle aree delle decelerazioni
    if math.isnan(res):
        res=0
    if total_X==0:
        total_X==7200
    return total_X,total_Y,res




def normalizza_fhr(fhr):
    '''
    Parametri: un segnale fhr
    Restituisce: il segnale fhr normalizzato
    '''
    norm=fhr.copy()
    mean= np.mean(norm)   
    
    for i in range(0,len(norm)):
        norm[i]=norm[i]-mean
    return norm     


def normalizza_UC(UC):
    '''
    Parametri:un segnale uc
    Restituisce: il segnale UC normalizzato
    ''' 
    norm=UC.copy()
    std= np.std(norm)

    for i in range(0,len(norm)):
        norm[i]=norm[i]/std
    return norm


def coeff_phi(u,y,n_a,n_b):
     
    N=len(y)
    Phi=[]
    Y = y[max(n_a, n_b):]

    for k in range(max(n_a, n_b), N):
        row = []
    # Termini autoregressivi (y ritardato)
        row.extend(y[k-i] for i in range(1, n_a+1))
    # Termini esogeni (u ritardato)
        row.extend(u[k-j] for j in range(0, n_b))
        Phi.append(row)

    Phi = np.array(Phi)

    # Ridurre y per corrispondere a Phi

# Regressione lineare per trovare i coefficienti
    theta = np.linalg.lstsq(Phi, Y, rcond=None)[0]

    # Separare i coefficienti a e b
    a = theta[:n_a]
    b = theta[n_a:]
    return a,b 



def arx(u,y,n_a,n_b):
    Y = y[max(n_a, n_b):]

    a,b=coeff_phi(y,u,n_a,n_b)
    y_pred = np.zeros_like(Y)
    
 
        
    for t in range(len(Y)):
        # Sommatoria dei termini autoregressivi (ritardi dell'output)
        
        ar_term = sum(a[i-1] * y[t - i ] for i in range(1,n_a+1) if t - i  >= 0)
        
        
            # Sommatoria dei termini esogeni (ritardi dell'input)
        ex_term = sum(b[i-1] * u[t - i] for i in range(0,n_b) if t - i  >= 0)
    
    
        # Valore predetto
        y_pred[t] = ar_term + ex_term 

    mse = np.mean((Y - y_pred)**2)
    y_pred[0]=np.mean(y_pred[1:])
    return y_pred,a,b

def test_arx(A,B,u,y):
    '''
    Testa un modello arx
    Parametri: coefficienti autoregressivi A e esogeni B, segnale esogeno u segnale autoregressivo y
    Restituisce: y_pred, la predizione del modello, l' errore quadratico medio del modello
    '''
    n=len(A)
    m=len(B)
    y_reduced = y[max(n, m):]
    y_pred = np.zeros_like(y_reduced)
    
    
    for t in range(len(y_reduced)):
        # Sommatoria dei termini autoregressivi (ritardi dell'output)
        
        ar_term = sum(A[i-1] * y[t - i] for i in range(1,n+1) if t - i  >= 0)
        
        
            # Sommatoria dei termini esogeni (ritardi dell'input)
        ex_term = sum(B[i] * u[t - i ] for i in range(0,m) if t - i  >= 0)
    
    
        # Valore predetto
        y_pred[t] = ar_term + ex_term 

    mse = np.mean((y_reduced - y_pred)**2)
    y_pred[0]=np.mean(y_pred[1:])
    return y_pred,mse


def grid_search(fhr,UC):
    '''
    
    Parametri:
    fhr: segnale della frequenza cardiaca fetale
    UC: segnale delle contrazioni uterine 

    Restituisce:
    n: ordine ritardo autoregressivo
    m: ordine ritardo esogeno
    A: coefficienti autoregressivi
    B: coefficienti esogeni
    y_pred: predizione dell' ultimo 30% dell' output utilizzando i coefficienti stimati attraverso il modello ARX del primo 70%
    '''

    
    min_mse=10000
    n=0
    m=0
    v_70=int(((7200/4)/100)*70)
    v_30=int((7200/100)*30)
    A=[]
    B=[]
    yp=[]
    for i in range(1,6):
        for j in range(1,6):
            ypred,a,b=arx(UC[:v_70],fhr[:v_70],i,j)
            y_pred,mse_pred=test_arx(a,b,np.array(UC[v_70:]),np.array(fhr[v_70:]))
            if mse_pred<min_mse:
                min_mse=mse_pred
                n=i
                m=j
                A=a
                B=b
                yp=ypred
    return n,m
            

def make_X(arr):
    '''
    Parametri: un array 2d
    Restituisce: una versione appiattita dell- array
    '''
    X=[]
    for i in arr:
        for a in i:
            X.append(a)
    return X

def find_weight(r):
    pos=0
    neg=0
    for i in range(0,len(r)):

        if r[i]<7.05:
            pos+=1
        else:
            neg+=1
    val=int(neg/pos)
    return val

positivi=crea_dataset()
dataset=load_dataset()
N=[]
M=[]
for key, i in dataset.items():
    fhr = i['fhr']  # Segnale del fetal heart rate (FHR)
    UC = i['UC']  # Intensità delle contrazioni uterine
    int_fhr=interpolazione_lineare(fhr)
    int_UC=interpolazione_lineare_UC(UC,fhr)
    norm_fhr=normalizza_fhr(int_fhr)
     
    norm_UC=normalizza_UC(int_UC)
    res_fhr=resample(norm_fhr,int(len(norm_fhr)/4))

    res_UC=resample(norm_UC,int(len(norm_UC)/4))
    #n,m=grid_search(norm_fhr,norm_UC)
    n,m=grid_search(res_fhr,res_UC)
    N.append(n)
    M.append(m)
n_mean=int(np.mean(N))
m_mean=int(np.mean(M))
np.seterr(divide='ignore', invalid='ignore')


A_coeff=[]
B_coeff=[]
slopes_pos=[]
slopes_neg=[]
area_pos=[]
area_neg=[]
depth_pos=[]
depth_neg=[]
x_pos=[]
x_neg=[]
y_pos=[]
y_neg=[]
X_train=[]
trap_pos=[]
trap_neg=[]
stdevs_neg=[]
stdevs_pos=[]
indexes=[]


v_70=int((7200/100)*70)
real=[]

for key,i in dataset.items():
    A=0
    B=0
    indexes.append(key)

    a_fhr = i['fhr']  # Segnale del fetal heart rate (FHR)
    a_UC = i['UC']  # Intensità delle contrazioni uterine
    a_int_fhr=interpolazione_lineare(a_fhr)
    a_int_UC=interpolazione_lineare_UC(a_UC,a_fhr)
   
    a_bl=baseline(a_int_fhr)
    a_norm_fhr=normalizza_fhr(a_int_fhr)
    a_norm_UC=normalizza_UC(a_int_UC)

    a_r_fhr=resample(a_norm_fhr,int(len(a_norm_fhr)/4))

    a_r_UC=resample(a_norm_UC,int(len(a_norm_UC)/4))

    a_ypred,A,B=arx(a_r_fhr,a_r_UC,n_mean,m_mean)
    
    A_coeff.append({'coeffs':A,
                   'ph':float(i['pH'])})
    B_coeff.append({'coeffs':B,
                   'ph':float(i['pH'])})

    a=trova_segmenti_inferiori_baseline(a_int_fhr,a_bl)
    segs=exclude_seconds(a,a_int_fhr,a_bl)
    
    
    x,y,z,s,h=slope_parameters(a_int_fhr,segs,a_bl)
    stdfhr=np.std(a_int_fhr)
    trap=curvearea(segs,a_int_fhr,a_bl)
    total_X,total_Y,total_area=statistics(x,y,z)
    if float(i['pH'])<7.05:
        slopes_pos.append(s)
        area_pos.append(total_area)
        x_pos.append(total_X)
        y_pos.append(total_Y)
        depth_pos.append(h)
        trap_pos.append(trap)
        stdevs_pos.append(stdfhr)
        
        
    else: 
        slopes_neg.append(s)
        area_neg.append(total_area)
        x_neg.append(total_X)
        y_neg.append(total_Y)
        depth_neg.append(h)
        trap_neg.append(trap)
        stdevs_neg.append(stdfhr)
    print(f'\r Campione -- {key} ',end='') 
    


    
   

   
    X_train.append(make_X([[total_X],[trap],A,[stdfhr],[s]]))
    
   
    real.append(float(i['pH']))

y =real.copy()
positive_A=[]
positive_B=[]
negative_A=[]
negative_B=[]
for a in A_coeff:
    if a['ph']<7.05:
        positive_A.append(a['coeffs'][0])
    else:
        negative_A.append(a['coeffs'][0])

for b in B_coeff:

    if b['ph']<7.05:
        positive_B.append(b['coeffs'][0])
        
    else:
        negative_B.append(b['coeffs'][0])
        

numero=find_weight(real)

weights=[]
for i in y:
    if i<7.05:
        weights.append(numero)
    else:
        weights.append(1)

zeros_real=[]
for s in y:
    if s<7.05:
        zeros_real.append(1)
    else:
        zeros_real.append(0)

r=[]

for i in range(0,len(X_train)):
    X_copy=X_train.copy()
    y_copy=y.copy()
    weights_copy=weights.copy()
    del X_copy[i]
    del y_copy[i]
    del weights_copy[i]
    model1=LinearRegression()
   
    
    model1.fit(X_copy,y_copy,weights_copy)
    res=model1.predict([X_train[i]])
    r.append(res) 


zeros_pred=[]

pred=[]
for a in r:
    if a[0]<7.05:
        zeros_pred.append(1)
        
    else:
        zeros_pred.append(0)
    pred.append(a[0])

tp=0
tn=0
fp=0
fn=0

for i in range(0,len(real)):
        
        
        if float(real[i])<7.10 and pred[i]<7.10:
            tp+=1
        elif float(real[i])>=7.10 and pred[i]<7.10:
            fp+=1
        elif float(real[i])<7.10 and pred[i]>=7.10:
            fn+=1
        elif float(real[i])>=7.10 and pred[i]>=7.10:
            tn+=1

accuracy=(tp+tn)/(tp+tn+fp+fn)
err=(fp+fn)/(tp+tn+fp+fn)
precision=tp/(tp+fp)
sensitivity=tp/(tp+fn)
specificity=tn/(tn+fp)
F_score = 2 * (precision * sensitivity) / (precision + sensitivity)

l=scipy.stats.pearsonr(pred,real)
print(f'Coefficiente di correlazione predizioni e valori reali: {l.statistic}')
print(f'P value: {l.pvalue}')


se_vector=[]
sp_vector=[]
inv_sp_vector=[]
res=[]
for j in range(700,760):
    tresh=j*0.01
    tp=0
    tn=0
    fp=0
    fn=0
    s=0
    sp=0
    
    for i in range(0,len(real)):
       
            
        if float(real[i])<tresh and pred[i]<tresh:
            tp+=1
        elif float(real[i])>=tresh and pred[i]<tresh:
            fp+=1
        elif float(real[i])<tresh and pred[i]>=tresh:
            fn+=1
        elif float(real[i])>=tresh and pred[i]>=tresh:
            tn+=1
    if tp+fn==0:
        s=0
    else:
        s=tp/(tp+fn)
    if tn+fp==0:
        sp=0
    else:
        sp=tn/(tn+fp)
        
    sp_vector.append(sp)
    se_vector.append(s)
    inv_sp_vector.append(1-sp)
    res.append({'thresh':tresh,
               '1-sp':1-sp,
                'se':s
               })    
prodotti=-np.abs(np.array(se_vector)-np.array(sp_vector))
indice_ottimo = np.argmax(prodotti)



plt.figure(figsize=(6,6))
plt.plot(inv_sp_vector,se_vector,linestyle='-',marker='o')

plt.gca().set_xlabel('FPR', fontsize=15)
plt.gca().set_ylabel('TPR', fontsize=15)
x = np.linspace(-10, 10, 400)  # genera 400 punti tra -10 e 10
y = -x  # la bisettrice ha la forma y = x
point1 = [-1, 2]
point2 = [1, 0]
x_values = [point1[0], point2[0]]
y_values = [point1[1], point2[1]]
plt.plot(x_values, y_values, 'bo', linestyle="--")
plt.plot(res[indice_ottimo]['1-sp'],res[indice_ottimo]['se'],'ro')
plt.xlim(0, 1)
plt.title('ROC',fontsize=20)
plt.ylim(0,1)

final_thresh=res[indice_ottimo]['thresh']

tp=0
tn=0
fp=0
fn=0
false_negatives=[]
true_positives=[]
true_negatives=[]
for i in range(0,len(real)):
       
        
        if float(real[i])<final_thresh and pred[i]<final_thresh:
            tp+=1
            true_positives.append(int(i))
        elif float(real[i])>=final_thresh and pred[i]<final_thresh:
            fp+=1
            

        elif float(real[i])<final_thresh and pred[i]>=final_thresh:
            false_negatives.append(int(i))
            fn+=1
        elif float(real[i])>=final_thresh and pred[i]>=final_thresh:
            tn+=1
            true_negatives.append(int(i))

accuracy=(tp+tn)/(tp+tn+fp+fn)
err=(fp+fn)/(tp+tn+fp+fn)
precision=tp/(tp+fp)
sensitivity=tp/(tp+fn)
specificity=tn/(tn+fp)
F_score = 2 * (precision * sensitivity) / (precision + sensitivity)


print(f'accuracy: {accuracy}')
print(f'error: {err}')
print(f'precision: {precision}')
print(f'sensitivity: {sensitivity}')
print(f'specificity: {specificity}')
print(f'F-score: {F_score} ')


x = np.linspace(-10, 10, 400)  # genera 400 punti tra -10 e 10
y = x  # la bisettrice ha la forma y = x

plt.xlim(6.85, 7.5)
plt.ylim(6.85, 7.5)
plt.axhline(final_thresh, color='black',linewidth=1)  # Asse x
plt.axvline(final_thresh, color='black',linewidth=1) 
plt.plot(x, y, label="Bisettrice (y = x)", color="black",linestyle='--')  # Disegna la bisettrice

# Aggiungi etichette e titolo
plt.gca().set_xlabel('Valori predetti(pH)', fontsize=13)
plt.gca().set_ylabel('Valori reali(pH)', fontsize=13)
plt.scatter(pred,real)
a=0.3
verts = [(0, final_thresh), (final_thresh, final_thresh), (final_thresh, 0),(0, 0)]

# Crea il poligono (triangolo)
polygon = patches.Polygon(verts,label='positivi' ,closed=True, color='green', alpha=a)


verts1 = [(0, final_thresh), (final_thresh, final_thresh), (final_thresh, 10),(0, 10)]

# Crea il poligono (triangolo)
polygon1 = patches.Polygon(verts1,label='falsi positivi', closed=True, color='blue', alpha=a)

verts2 = [ (final_thresh, final_thresh),(10, final_thresh), (10, 10),(final_thresh, 10)]

# Crea il poligono (triangolo)
polygon2 = patches.Polygon(verts2,label='negativi', closed=True, color='red', alpha=a)


verts3 = [ (final_thresh, final_thresh),(10, final_thresh), (10, 0),(final_thresh, 0)]

# Crea il poligono (triangolo)
polygon3 = patches.Polygon(verts3,label='falsi negativi', closed=True, color='orange', alpha=a)



# Aggiungi il poligono alla figura
plt.gca().add_patch(polygon2)
plt.gca().add_patch(polygon3)
plt.gca().add_patch(polygon1)
plt.gca().add_patch(polygon)

plt.text(6.90,6.95,'TP',fontsize=22 )
plt.text(6.90,7.25,'FP',fontsize=22 )
plt.text(7.35,7.25,'TN',fontsize=22 )
plt.text(7.35,6.95,'FN',fontsize=22 )


def calcola_auc_manuale(fpr, tpr):
    """
    Calcola manualmente la AUC utilizzando il metodo trapezoidale.

    Args:
        fpr (array-like): Tasso di falsi positivi.
        tpr (array-like): Tasso di veri positivi.

    Returns:
        float: Valore della AUC.
    """
    return np.trapz( fpr,tpr)

auc=calcola_auc_manuale(se_vector,inv_sp_vector)

print(f'AUC: {auc}')

plt.plot(inv_sp_vector,se_vector,linestyle='-',marker='o',color='blue')

plt.gca().set_xlabel('FPR',fontsize=13)
plt.gca().set_ylabel('TPR',fontsize=13)

plt.fill_between(inv_sp_vector,se_vector,alpha=0.4,color='blue',label='AUC = 0.77')
plt.xlim(0, 1)
plt.legend()
plt.ylim(0,1)
