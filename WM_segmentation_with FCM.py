##################### Clustering White matter (WM) with 3D primary FCM ##################


##################### Loading Libraries ######################
if True:
    print('Loading libraries ......................')
    print('Loading Numpy ...............')
    import numpy as np
    print('Loading fcmeans .............')
    from fcmeans import FCM
    print('Loading nibabel .............')
    import nibabel as nib
    print('Loading matplotlib ..........')
    #import matplotlib.pyplot as plt
    from matplotlib import pyplot as plt
    print('Loading Time ...........')
    import time
    print('1) Loading libraries is finished ......................')

##################### Loading MRI Data and Segmented MRI Data ######################
Select_data = int(input('Select dataset; AD: 0, MCI: 1: '))
################# AD ####################  
if Select_data==0:
    #for Sub in range(1,33):# for all subjects
    for Sub in range(1,3):
        T0 = time.time()
        ###################### Loading MRI data
        print('Loading Nifti data of AD patient ................................')
        print('Subject: ',Sub)
        address ='E:/BIO_PhD/DrNazemzadeh/data/ADNI/AD'+'/'+str(Sub)+'.nii'
        I = nib.load(address).get_fdata()
        if Sub==1:
            # WM_labelAD = np.zeros((32,20)) # for all subjects
            WM_labelAD = np.zeros((2,20))
            # I_WM_seg_AD = np.zeros((32,I.shape[0],I.shape[1],20)) # for all subjects
            I_WM_seg_AD = np.zeros((2,I.shape[0],I.shape[1],20))
            
        ###################### Two operation: 1)Extract 20 slices from all slices  2)Transform MRI image to Vector
        print('Designing ML System (FCM clustering) ............................')
        I_vec = I[:,:,50:70].reshape(((I.shape[0])*(I.shape[1])*(20),1))
        ###################### Designing ML System (FCM clustering) 
        print('Designing ML System (FCM clustering) ............................')
        FCM_model = FCM(n_clusters=5) # we use two cluster as an example m=2;
        ###################### Training ML System (FCM clustering) 
        print('Training ML System (FCM clustering) based on Input Data .........')
        FCM_model.fit(I_vec) ## X is numpy array. rows:samples  and columns:features
        ###################### Segmenting Brain MRI with ML System
        print('Clustering (Segmenting) MRI data with ML System.........')
        labels = FCM_model.predict(I_vec)
        ###################### Transform Vector to Image
        print('Clustering (Segmenting) MRI data with ML System.........')
        I_seg_T = labels.reshape((I[:,:,50:70].shape))
        ###################### Selecting the Correct Label for White matter based on Plotting Slices
        print('Selecting the Correct Label for White matter based on Plotting Slices and Quality Control ........')
        for Sam in range(20):
            if Sam==0 or Sam==10 or Sam==19:
                ################ Plotting Original MRI and FCM result MRI
                print('Slice: ', Sam+50)
                I_seg = I_seg_T[:,:,Sam]
                fig = plt.figure(figsize=(20,10),dpi=72)
                grid = plt.GridSpec(1,2)      
                ax1 = fig.add_subplot(grid[0,0]) 
                plt.imshow(I[:,:,Sam+50],cmap='gray')
                ax1.set_title('Subject ID: '+str(Sub)+' Slice Number: '+str(Sam+50))
                ax2 = fig.add_subplot(grid[0,1])
                plt.imshow(I_seg)
                ax2.set_title('FCM Clustering Segmentation')
                plt.show()
                ################# Select Correct Label
                WM_labelAD[Sub-1,:] = np.ones((20))*int(input('Enter whitematter Label: '))
            
            ################# Creating Mask from User Label    
            I_WM_mask = I_seg==int(WM_labelAD[Sub-1,Sam])
            ################ Segmenting White matter from MRI with Mask 
            I_WM_seg_AD[Sub-1,0:I.shape[0],0:I.shape[1],Sam] = I[:,:,Sam+50]*I_WM_mask
            
     
        ## Saving output of each subject for avoiding missing data
        print('Saving output of each subject..........')
        np.save('E:/BIO_PhD/Workshop/AI_MRI/Code/Code_reuslt/I_WM_seg_AD_1', I_WM_seg_AD[0:Sub,:,:,0:20])
        np.save('E:/BIO_PhD/Workshop/AI_MRI/Code/Code_reuslt/WM_labelAD_1', WM_labelAD[0:Sub,0:20])
        print('Loading data and preprocessing for AD subject number' ,Sub,' are finished .....')
        T1 = time.time()
        print('Loading and Preprocessing Total Duration(sec): ',round(T1-T0))
        
    ## Saving output of all AD subjects
    print('Saving output of all  subjects..........')
    np.save('E:/BIO_PhD/Workshop/AI_MRI/Code/Code_reuslt/I_WM_seg_AD_T', I_WM_seg_AD)
    np.save('E:/BIO_PhD/Workshop/AI_MRI/Code/Code_reuslt/WM_labelAD_T', WM_labelAD)
    print('2) Loading data and preprocessing for all AD subjects are finished .......')
    print('..........................................................................')

        

    ## Ploting T1 MRI and White matter MRI
    if True:
        Subject = int(input('Enter Subject ID[1,2,3,..] (Ex:1): '))       
        address ='E:/BIO_PhD/DrNazemzadeh/data/ADNI/AD'+'/'+str(Subject)+'.nii'
        I = nib.load(address).get_fdata()    
        fig = plt.figure(figsize=(20,20),dpi=72)
        grid = plt.GridSpec(1,2)
        # Sample slice: 60       
        ax1 = fig.add_subplot(grid[0,0]) 
        plt.imshow(I[:,:,60],cmap='gray')
        ax1.set_title('60th Slice')
        ax2 = fig.add_subplot(grid[0,1])
        plt.imshow(I_WM_seg_AD[Subject-1,:,:,10],cmap='gray')
        ax2.set_title('WM Segmentation')
        plt.show()



################# MCI #####################
if Select_data==1:
    #for Sub in range(1,33):
    for Sub in range(1,3):
        T0 = time.time()
        ###################### Loading MRI data
        print('Loading Nifti data of MCI patient ................................')
        print('Subject: ',Sub)
        address ='E:/BIO_PhD/DrNazemzadeh/data/ADNI/MCI'+'/'+str(Sub)+'.nii'
        I = nib.load(address).get_fdata()
        if Sub==1:
            # WM_labelMCI = np.zeros((32,20)) # for all subjects
            WM_labelMCI = np.zeros((2,20))
            # I_WM_seg_MCI = np.zeros((32,I.shape[0],I.shape[1],20)) # for all subjects
            I_WM_seg_MCI = np.zeros((2,I.shape[0],I.shape[1],20))
            
        ###################### Two operation: 1)Extract 20 slices from all slices  2)Transform MRI image to Vector
        print('Designing ML System (FCM clustering) ............................')
        I_vec = I[:,:,50:70].reshape(((I.shape[0])*(I.shape[1])*(20),1))
        ###################### Designing ML System (FCM clustering) 
        print('Designing ML System (FCM clustering) ............................')
        FCM_model = FCM(n_clusters=5) # we use two cluster as an example m=2;
        ###################### Training ML System (FCM clustering) 
        print('Training ML System (FCM clustering) based on Input Data .........')
        FCM_model.fit(I_vec) ## X is numpy array. rows:samples  and columns:features
        ###################### Segmenting Brain MRI with ML System
        print('Clustering (Segmenting) MRI data with ML System.........')
        labels = FCM_model.predict(I_vec)
        ###################### Transform Vector to Image
        print('Clustering (Segmenting) MRI data with ML System.........')
        I_seg_T = labels.reshape((I[:,:,50:70].shape))
        ###################### Selecting the Correct Label for White matter based on Plotting Slices
        print('Selecting the Correct Label for White matter based on Plotting Slices and Quality Control ........')
        for Sam in range(20):
            if Sam==0 or Sam==10 or Sam==19:
                ################ Plotting Original MRI and FCM result MRI
                print('Slice: ', Sam+50)
                I_seg = I_seg_T[:,:,Sam]
                fig = plt.figure(figsize=(20,10),dpi=72)
                grid = plt.GridSpec(1,2)      
                ax1 = fig.add_subplot(grid[0,0]) 
                plt.imshow(I[:,:,Sam+50],cmap='gray')
                ax1.set_title('Subject ID: '+str(Sub)+' Slice Number: '+str(Sam+50))
                ax2 = fig.add_subplot(grid[0,1])
                plt.imshow(I_seg)
                ax2.set_title('FCM Clustering Segmentation')
                plt.show()
                ################# Select Correct Label
                WM_labelMCI[Sub-1,:] = np.ones((20))*int(input('Enter whitematter Label: '))
                
            ################# Creating Mask from User Label     
            I_WM_mask = I_seg==int(WM_labelMCI[Sub-1,Sam])
            ################ Segmenting White matter from MRI with Mask
            I_WM_seg_MCI[Sub-1,0:I.shape[0],0:I.shape[1],Sam] = I[:,:,Sam+50]*I_WM_mask
            
        ## Saving output of each subject for avoiding missing data
        print('Saving output of each subject..........')
        np.save('E:/BIO_PhD/Workshop/AI_MRI/Code/Code_reuslt/I_WM_seg_MCI_0', I_WM_seg_MCI[0:Sub,:,:,0:20])
        np.save('E:/BIO_PhD/Workshop/AI_MRI/Code/Code_reuslt/WM_labelMCI_0', WM_labelMCI[0:Sub,0:20])
        T1 = time.time()
        print('Loading and Preprocessing Total Duration(sec): ',round(T1-T0))

    ## Saving output of all AD subjects
    print('Saving output of all  subjects..........')
    np.save('E:/BIO_PhD/Workshop/AI_MRI/Code/Code_reuslt/I_WM_seg_MCI_T', I_WM_seg_MCI)
    np.save('E:/BIO_PhD/Workshop/AI_MRI/Code/Code_reuslt/WM_labelMCI_T', WM_labelMCI)
    print('2) Loading data and preprocessing for all MCI subjects are finished ......')
    print('..........................................................................')

## Ploting
    if True:
        Subject = int(input('Enter Subject ID[1,2,3,...] (Ex:1): '))       
        address ='E:/BIO_PhD/DrNazemzadeh/data/ADNI/MCI'+'/'+str(Subject)+'.nii'
        I = nib.load(address).get_fdata()    
        fig = plt.figure(figsize=(15,15),dpi=72)
        grid = plt.GridSpec(1,2)
        # Sample slice: 60       
        ax1 = fig.add_subplot(grid[0,0]) 
        plt.imshow(I[:,:,60],cmap='gray')
        ax1.set_title('60th Slice')
        ax2 = fig.add_subplot(grid[0,1])
        plt.imshow(I_WM_seg_MCI[Subject-1,:,:,10],cmap='gray')
        ax2.set_title('WM Segmentation')
        plt.show()



