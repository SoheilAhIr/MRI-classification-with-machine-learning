########### AD vs MCI classification with clustring white matter #############


if True:
    print('Loading libraries ......................')
    print('Loading Numpy ..................')
    import numpy as np
    print('Loading pywt ...................')
    import pywt
    print('Loading scipy ..................')
    from scipy.stats import skew
    print('Loading sklearn ................')
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_validate
    from sklearn.utils import shuffle
    print('Loading Time ..................')
    import time
    print('Loading libraries finished ..................')
    

### Select Type of Patients
Select_data = int(input('Select dataset; AD: 0, MCI: 1: '))

########################### AD Feature extraction #######################
if Select_data==0:
    t0 = time.time()
    ####### Loading White matter segmented MRI 
    I_WM_seg_AD = np.load('E:/BIO_PhD/DrNazemzadeh/result/3D/paper_result/I_WM_seg_AD_T.npy')
    for Sub in range(1,41): # for all subjects
        print('Subject: ',Sub)
        if Sub==18 or Sub==20 or Sub==23 or Sub==24 or Sub==25 or Sub==26 or Sub==27 or Sub==28:
            continue
        else:
            for Sam in range(20):
                ############## feature group 1  extarct wavelet approximate of level 3
                print('Slice: ',Sam+1)
                print('Feature group 1 (wavelet approximate) ..................')
                C = pywt.wavedec2(I_WM_seg_AD[Sub-1,:,:,Sam],'haar',level=3)
                ############## feature group 2 Statistical parameters
                print('Feature group 2 (Statistical parameters) ...............')
                F_mean = np.mean(I_WM_seg_AD[Sub-1,:,:,Sam])
                F_var = np.var(I_WM_seg_AD[Sub-1,:,:,Sam])  
                F_skew = skew(I_WM_seg_AD[Sub-1,:,:,Sam],axis = None)
                if Sam==0 and Sub==1:
                    I_wav_AD = C[0].reshape((1,C[0].shape[0]*C[0].shape[1]))
                    I_hos_AD = np.array(([[F_mean,F_var,F_skew]]))
                else:
                    n = C[0].reshape((1,C[0].shape[0]*C[0].shape[1]))
                    I_wav_AD = np.concatenate((I_wav_AD,n),axis=0)
                    I_hos_AD = np.concatenate((I_hos_AD,np.array(([[F_mean,F_var,F_skew]]))),axis=0)
            
    ############### Dimension Reduction Group 1 with PCA Feature Extraction Method
    print('Feature group 1 dimension reduction with PCA..........')
    I_wav_AD = StandardScaler().fit_transform(I_wav_AD)
    pca = PCA(n_components=7)
    pca.fit(I_wav_AD)
    I_pca_AD = pca.transform(I_wav_AD)
    ############### Merge Features from PCA and Group 2
    F_space_AD = np.concatenate((I_pca_AD,I_hos_AD),axis=1)
    ############### Creating Outputs 
    YUC_AD = np.zeros((640,1))
    t1 = time.time()
    print('AD Feature extraction duration(sec): ',int(t1-t0))
    
########################### MCI Feature extraction #######################          
if Select_data==1:
    t0 = time.time()
    ####### Loading White matter segmented MRI 
    I_WM_seg_MCI = np.load('E:/BIO_PhD/DrNazemzadeh/result/3D/paper_result/I_WM_seg_MCI_T.npy')
    for Sub in range(1,41):# for all 32 patients
        print('Subject: ',Sub)
        if Sub==18 or Sub==20 or Sub==23 or Sub==24 or Sub==25 or Sub==26 or Sub==27 or Sub==28:
            continue
        else:
            for Sam in range(20):
                ## feature group 1  extarct wavelet approximate of level 3
                print('Slice: ',Sam+1)
                print('Feature group 1 (wavelet approximate) ..................')
                C = pywt.wavedec2(I_WM_seg_MCI[Sub-1,:,:,Sam],'haar',level=3)
                ############## feature group 2 Statistical parameters
                print('Feature group 2 (Statistical parameters) ...............')
                F_mean = np.mean(I_WM_seg_MCI[Sub-1,:,:,Sam])
                F_var = np.var(I_WM_seg_MCI[Sub-1,:,:,Sam])  
                F_skew = skew(I_WM_seg_MCI[Sub-1,:,:,Sam],axis = None)
                if Sam==0 and Sub==1:
                    I_wav_MCI = C[0].reshape((1,C[0].shape[0]*C[0].shape[1]))
                    I_hos_MCI = np.array(([[F_mean,F_var,F_skew]]))
                else :
                    n = C[0].reshape((1,C[0].shape[0]*C[0].shape[1]))
                    I_wav_MCI = np.concatenate((I_wav_MCI,n),axis=0)
                    I_hos_MCI = np.concatenate((I_hos_MCI,np.array(([[F_mean,F_var,F_skew]]))),axis=0)                 

          
    ############### Dimension Reduction Group 1 with PCA Feature Extraction Method
    print('Feature group 1 dimension reduction with PCA..........')
    I_wav_MCI = StandardScaler().fit_transform(I_wav_MCI)
    pca = PCA(n_components=7)
    pca.fit(I_wav_MCI)
    I_pca_MCI = pca.transform(I_wav_MCI)
    ############### Merge Features from PCA and Group 2
    F_space_MCI = np.concatenate((I_pca_MCI,I_hos_MCI),axis=1) 
    ############### Creating Outputs 
    YUC_MCI = np.ones((640,1))
    t1 = time.time()
    print('MCI duration(sec): ',int(t1-t0))



###### Cross validation #######
if True:
    X_Total = np.concatenate((F_space_AD,F_space_MCI),axis=0)
    Y_Total = np.concatenate((YUC_AD,YUC_MCI),axis=0)    
################ Automatic kfold ########################
##### Shufflig #####
if True:
    Data = np.concatenate((X_Total,Y_Total),axis=1)
    Data_shuff = shuffle(Data,random_state=0)
          
############## Ensemble ###############

if True:
    ## manual 10 Fold
    Data_10Fold = np.zeros((10,128,11))
    c = 0
    for i in range(0,1279,128):
        Data_10Fold[c,:,:] = Data_shuff[i:i+128,:]
        c = c+1 # for fold



if True:
    t0 = time.time()
    ######  10Fold
    for k in range(10):
        # Test
        x_test = Data_10Fold[k,:,0:10]
        y_test = Data_10Fold[k,:,10]
        # Train
        x_train = np.zeros((1,10)) # initialize  x_train then remove this row
        y_train = np.zeros((1))# initialize y_train then remove this row
        for j in range(10):
            if j!=k:
                x_train = np.concatenate((x_train,Data_10Fold[j,:,0:10]),axis=0)
                y_train = np.concatenate((y_train,Data_10Fold[j,:,10]),axis=0)
        # Primary Machine Models
        knn_model = KNeighborsClassifier(n_neighbors = 9)
        DT_model = DecisionTreeClassifier(criterion='entropy',max_depth=8)
        LDA_model = LinearDiscriminantAnalysis()
        # Training Machine Models
        x_train = np.delete(x_train,0,0) # remove additianal row
        y_train = np.delete(y_train,0,0) # remove additianal row
        print('Training Machines...............')
        print('Training KNN...')
        knn_model.fit(x_train, y_train.flatten())
        print('Training DT...')
        DT_model.fit(x_train, y_train.flatten())
        print('Training LDA...')
        LDA_model.fit(x_train, y_train.flatten())
        
        # Testing evaluate Machine Models
        pre_test_knn =  knn_model.predict(x_test)
        pre_test_DT =  DT_model.predict(x_test)
        pre_test_LDA =  LDA_model.predict(x_test)
        # Training evaluate Machine Models
        pre_train_knn =  knn_model.predict(x_train)
        pre_train_DT =  DT_model.predict(x_train)
        pre_train_LDA =  LDA_model.predict(x_train)
        # Votting Train
        VOT_Train = np.zeros((len(y_train)))
        for i in range(len(y_train)):
            vot = np.zeros((3))
            vot[2] = pre_train_LDA[i]
            vot[1] = pre_train_DT[i]
            vot[0] = pre_train_knn[i]
            s = np.sum(vot)
            if s>=2:
                VOT_Train[i]=1
            else:
                VOT_Train[i]=0
        # Votting Test   
        VOT_Test = np.zeros((len(y_test)))
        for i in range(len(y_test)):
            vot = np.zeros((3))
            vot[2] = pre_test_LDA[i]
            vot[1] = pre_test_DT[i]
            vot[0] = pre_test_knn[i]
            s = np.sum(vot)
            if s>=2:
                VOT_Test[i]=1
            else:
                VOT_Test[i]=0

        ## Show Test performance in each fold
 #if True:
        print('Test Performence.........................................fold: ',k)
        # knn
        print('KNN Performance in Test....')
        print('Total report:',classification_report(pre_test_knn, y_test,digits=4))
        print('Accuracy report:',accuracy_score(y_test, pre_test_knn))        
        # DT
        print('DT Performance in Test....')
        print('Total report:',classification_report(pre_test_DT, y_test,digits=4))
        print('Accuracy report:',accuracy_score(y_test, pre_test_DT))
        # LDA
        print('LDA Performance in Test....')
        print('Total report:',classification_report(pre_test_LDA, y_test,digits=4))
        print('Accuracy report:',accuracy_score(y_test, pre_test_LDA))
        #####
        # Ensemble
        print('Ensemble Performance in Test....')
        print('Total report:',classification_report(VOT_Test, y_test,digits=4))
        print('Accuracy report:',accuracy_score(y_test, VOT_Test))
        
        
        ## Show Train performance in each fold
        # Linear SVM
        print('Train Performence.........................................fold: ',k)
        # knn
        print('KNN Performance in Train....')
        print('Total report:',classification_report(pre_train_knn, y_train,digits=4))
        print('Accuracy report:',accuracy_score(y_train, pre_train_knn)) 
        # DT
        print('DT Performance in Train....')
        print('Total report:',classification_report(pre_train_DT, y_train,digits=4))
        print('Accuracy report:',accuracy_score(y_train, pre_train_DT))
        # LDA
        print('LDA Performance in Train....')
        print('Total report:',classification_report(pre_train_LDA, y_train,digits=4))
        print('Accuracy report:',accuracy_score(y_train, pre_train_LDA))
        # Ensemble
        print('Ensemble Performance in Train....')
        print('Total report:',classification_report(VOT_Train, y_train,digits=4))
        print('Accuracy report:',accuracy_score(y_train, VOT_Train))
        
    t1 = time.time()
    print('Ensemble duration(sec): ',int(t1-t0))

        
        
        
    


