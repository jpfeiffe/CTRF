import numpy as np
def simu_confounding_data(n=2000,p=20,scenario=2,r=0.65,binary=1):
    simu_data={'Y':[],'S':[],'V':[],'X':[],'r':r,'scenario':scenario}

    p_s=int(p*0.4)
    p_v=int(p*0.6)

    i_grid=np.linspace(1,p_s,p_s)
    alpha=(-1)**i_grid*(i_grid%3+1)*p/3
    beta=p/2
    count=0
    while count<n:
        if scenario==1:
            S=np.random.normal(size=p_s)
            V = np.random.normal(size=p_v)
        elif scenario==2:
            #S Causes V
            S=np.random.normal(size=p_s)
            V = np.zeros(p_v)
            for j in range(p_v):
                V[j]=np.random.normal(loc=int(S[j%p_s]>0)+int(S[(j+1)%p_s]>0))
        else:
            #V Causes S
            V=np.random.normal(size=p_v)
            S = np.zeros(p_s)
            for j in range(p_s):
                S[j]=np.random.normal(loc=int(V[j%p_v]>0)+int(V[(j+1)%p_v]>0))

        S_obs=np.zeros(p_s)
        V_obs=np.zeros(p_v)
        S_obs[np.where(S>0)]=1
        V_obs[np.where(V>0)]=1

        logit=np.sum(alpha*S_obs)+(np.sum(S_obs[1:]*S_obs[:p_s-1]))*beta

        Y=1/(1+np.exp(-logit))+np.random.normal(scale=0.2)
        if binary==1:
            Y_obs=0
            if Y>0.5:
                Y_obs=1
        else:
            Y_obs=Y

        noisy_mean=np.mean(V_obs)
        inclusion=np.random.uniform()
        ###Positive Correlation
        if (Y_obs>0.5 and noisy_mean>0.5) or (Y_obs<0.5 and noisy_mean<0.5):
            if inclusion<r:
                simu_data['Y'].append(Y_obs)
                simu_data['S'].append(S_obs)
                simu_data['V'].append(V_obs)
                simu_data['X'].append(np.hstack([S_obs,V_obs]))
                count+=1
        ###Negative Correlation
        else:
            if inclusion<(1-r):
                simu_data['Y'].append(Y_obs)
                simu_data['S'].append(S_obs)
                simu_data['V'].append(V_obs)
                simu_data['X'].append(np.hstack([S_obs,V_obs]))
                count+=1
    #Into Array
    simu_data['Y']=np.asarray(simu_data['Y'])
    simu_data['X']=np.asarray(simu_data['X'])
    simu_data['V']=np.asarray(simu_data['V'])
    simu_data['S']=np.asarray(simu_data['S'])
    
    return simu_data

