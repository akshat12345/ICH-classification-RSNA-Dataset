import traceback
import glob
import pandas as pd
import pydicom as dicom
import numpy as np
import warnings

from pathlib import Path
import os


# def savepreprocess(folder_dir):
#     # classtype = ['Cap', 'Covid', 'Normal']
#     rawdcm = folder_dir + 'stage_2_train/'
#     savepathnpy = folder_dir + 'stage_2_train_numpy/'
#     with open(, 'rb') as n:
#         npyrawcase = np.load(n, allow_pickle=True)
#
#
#     for ctype in classtype:
#         if not os.path.isdir(savepathnpy + ctype + '/'):
#             os.mkdir(savepathnpy + ctype + '/')
#         cpathraw = rawdcm + ctype + '/'
#         caselist = glob.glob(cpathraw + r'/*.npy')
#         for tcase in caselist:
#             with open(tcase, 'rb') as n:
#                 npyrawcase = np.load(n, allow_pickle=True)
#             caseno = tcase.split('/')[-1].split('.npy')[0]
#             # print(npyrawcase.shape,caseno)
#             roilist = []
#             for rawob in npyrawcase:
#                 winc, winw = getcentrewidth(rawob)
#                 # print(winc,winw)
#                 oc = rawob.pixel_array
#                 vximg = converttovoxel(rawob)
#                 # print('OC',oc.min(),oc.max())
#                 # print('VOX',vximg.min(),vximg.max())
#                 focvox = focusvoxel(vximg, -1350.0, 150)
#                 # print('FVX',focvox.min(),focvox.max())
#                 # plt.subplot(131),plt.imshow(oc,cmap=plt.cm.bone),plt.title('OC')
#                 # plt.subplot(132),plt.imshow(vximg,cmap=plt.cm.bone),plt.title('VX')
#                 # plt.subplot(133),plt.imshow(focvox,cmap=plt.cm.gray),plt.title('FVX')
#                 # plt.show()
#                 roilist.append(focvox)
#             #     break
#             # break
#             roisavepath = savepathnpy + ctype + '/' + caseno + '.npy'
#             # #UnComment To save
#             with open(roisavepath, 'wb') as a:
#                 np.save(a, np.array(roilist))
def savedicomtonumpy():
    ROOTFOLDER = '/home/student/Akshat/Mini Project/output/'
    image_train_data = '/home/student/Akshat/Mini Project/stage_2_train/'
    savepathraw = '/home/student/Akshat/Mini Project/stage_2_train_numpy/'
    train_df = pd.read_csv(ROOTFOLDER + 'stage_2_train_with_metadata_sorted2.csv')

    cur_instance_id = train_df['StudyInstanceUID'][0]
    dicom_list = []
    # count = 1
    sum = 0
    for id in train_df.index:
        # if id > 110:
        #     break
        if train_df['StudyInstanceUID'][id] == cur_instance_id:
            dcm_file = image_train_data + train_df['ID'][id] + '.dcm'
            dicom_list.append(dicom.read_file(dcm_file))
        else:
            print('LIST ')
            # print(count)
            print(len(dicom_list))
            sum = sum + len(dicom_list)
            # for i in dicom_list:
            #     print(i)
            # count = count+1
            with open(savepathraw+cur_instance_id+'.npy','wb') as a:
                np.save(a,np.array(dicom_list),allow_pickle=True)
            cur_instance_id = train_df['StudyInstanceUID'][id]
            dicom_list = []
            dcm_file = image_train_data + train_df['ID'][id] + '.dcm'
            dicom_list.append(dicom.read_file(dcm_file))

    if(len(dicom_list) > 0):
        sum = sum + len(dicom_list)
        with open(savepathraw+cur_instance_id+'.npy','wb') as a:
            np.save(a,np.array(dicom_list),allow_pickle=True)
    print('total ')
    print(sum)

def dicomtonumpy(folder_dir):
    csv_file = 'stage_2_train_with_metadata_sorted2.csv'
    dataframe = pd.read_csv(folder_dir + '/' + csv_file)
    caselist = glob.glob(folder_dir + r'*/')
    savepathraw = '/home/student/Akshat/Mini Project/'
    count = 0
    cur_instance_id = dataframe['StudyInstanceUID'][0]
    dicomlist = [] # dataframe['ID']
    print(' cur instance id'  + cur_instance_id)
    for id in dataframe.index:
        if count > 100:
            break
        count = count + 1
        # print(id)
        if  cur_instance_id == dataframe['StudyInstanceUID'][id]:
            dicomobj = dicom.read_file(folder_dir+'stage_2_train/' + dataframe['ID'][id] + '.dcm')
            dicomlist.append(dicomobj)
        else:
            print(len(dicomlist))
            for i in dicomlist:
                print(i['StudyInstanceUID'])
            print(20 * '-')
            with open(savepathraw + 'stage_2_train_numpy/' + cur_instance_id + '.npy', 'wb') as a:
                np.save(a, np.array(dicomlist), allow_pickle=True)
            cur_instance_id = dataframe['StudyInstanceUID'][id]
            dicomlist = []
    if dicomlist != []:
        with open(savepathraw + 'stage_2_train_numpy/' + cur_instance_id + '.npy', 'wb') as a:
            np.save(a, np.array(dicomlist), allow_pickle=True)


def threeDsliceframe(current_instance_id,savepath,case_df):
    #### slice images based on required slicecountpercase and step as stride
    final_df = {'StudyInstanceUID':current_instance_id,'3Dslice':[],'any' : [], 'edh' :[], 'iph' : [], 'ivh' :[],'sah':[],'sdh':[]}
    id = 0
    print(case_df[0])
    print(case_df[1])
    slice_list= []
    while id < case_df.index.size:
        if case_df['any'][id] == 1:
            if id == 0:
                for i in range(0, 3, 1):
                    final_df['3dslice'].append(case_df['npy'][id + i])
                id = id + 1
            elif id == case_df.index.size - 1:
                for i in range(-2, 1, 1):
                    final_df['3dslice'].append(case_df['npy'][id + i])
                id = id + 1
            else:
                for i in range(-1,2,1):
                    final_df['3dslice'].append(case_df['npy'][id+i])
                id = id + 2
        else:
            id = id + 1
        slice_list.append(final_df)

    print(slice_list)




    # slicedcase = []
    # fcount = casedata.shape[0]
    # if fcount > slicecountpercase:
    #     for i in range(0, fcount, step):
    #         strt = i
    #         end = i + slicecountpercase
    #         if end <= fcount:
    #             # print('\n--',strt,end,casedata[strt:end,:,:].shape)
    #             slicedcase.append(casedata[strt:end, :, :])
    #         else:
    #             break
    # else:
    #     slicedcase.append(casedata)
    #
    # return np.array(slicedcase)


def shallowslicing():
    savepathnpy = '/home/student/Akshat/Mini Project/stage_2_numpy_preprocess_sliced/'
    csv_path = '/home/student/Akshat/Mini Project/output/stage_2_train_with_metadata_sorted2.csv'
    folder_dir = '/home/student/Akshat/Mini Project/stage_2_numpy_preprocess/'
    sorted_csv = pd.read_csv(csv_path)
    current_instance_id = sorted_csv['StudyInstanceUID'][0]
    df = pd.DataFrame()
    col_list = { 'StudyInstanceUID' : [],
        'npy' : [],
        'any' : [] ,
        'edh' : [] ,
        'iph' : [] ,
        'ivh' : [] ,
        'sah' : [] ,
        'sdh' : []}
    counter = 0
    id = 0
    while id in range(0,sorted_csv.index.size):
        if counter > 2:
            break
        counter = counter + 1
        count = 0
        with open(folder_dir + current_instance_id +'.npy', 'rb') as n:
            npyrawcase = np.load(n, allow_pickle=True)
        npyrawcase = npyrawcase[0]
        while sorted_csv['StudyInstanceUID'][id] == current_instance_id :
            for column in col_list:
                if column == 'npy':
                    col_list[column].append(npyrawcase[count])
                elif column == 'StudyInstanceUID':
                    col_list[column].append(current_instance_id)
                else:
                    col_list[column].append(sorted_csv[column][id])
            id = id + 1
            count = count+1
        # print(count)
        # print(id)
        for col in col_list:
            df[col] = col_list[col]
            col_list[col] = []
        threeDsliceframe(current_instance_id,savepathnpy,df)

        if id >= sorted_csv.index.size:
            break
        current_instance_id = sorted_csv['StudyInstanceUID'][id]
        # print(df)
        df = pd.DataFrame()











    # i = 0
    #
    # for tcase in clist:
    #     casesamples = []
    #     with open(tcase, 'rb') as n:
    #         npycase = np.load(n)
    #     filename = tcase.split('/')[-1]
    #     # print(npycase.shape,filename)
    #     requiredframespercase = 8
    #     step = 8  # if same as requiredframespercase/depth it will be non-overlapping scenario
    #     slices = threeDsliceframe(npycase, step=step, slicecountpercase=requiredframespercase)
    #     casesamples.extend(slices)
    #     # print('3D Samples',slices.shape)
    #     i += slices.shape[0]
    #
    #     casesamples = np.array(casesamples).astype(np.float32)
    #
    #     # #UnComment To save
    #     # with open(savefilepath+filename,'wb') as a:
    #     #   np.save(a,casesamples)
    #
    # print('\t\t', split, '- Total Samples: ', i)


def getcentrewidth(slicedc):
    if isinstance(getattr(slicedc ,'WindowCenter') , dicom.valuerep.DSfloat) and isinstance(getattr(slicedc,'WindowWidth'),
                                                                               dicom.valuerep.DSfloat):
        c = int(slicedc.WindowCenter)
        w = int(slicedc.WindowWidth)
    else:
        c = int(slicedc.WindowCenter[0])
        w = int(slicedc.WindowWidth[0])
    return c, w


def converttovoxel(dcimg):
    image = dcimg.pixel_array.astype(np.float16)
    # print(image.min(),image.max())
    intercept = getattr(dcimg,'RescaleIntercept')
    slope = getattr(dcimg,'RescaleSlope')

    if int(slope) != 1:
        image = slope * image
    image += intercept
    return image


def focusvoxel(huimg, MINHU, MAXHU):
    rem = huimg.copy()
    # res = huimg.copy()
    # MINHU = (C - W/2)
    # MAXHU = (C + W/2)

    rem[rem <= MINHU] = 0.
    rem[rem > MAXHU] = 0.
    return rem


def savepreprocess(folder_dir):
    npy_path = folder_dir + 'stage_2_train_numpy/'
    npy_list = glob.glob(npy_path + r'/*.npy')
    savepath = folder_dir + 'stage_2_preprocess_3/'
    print(len(npy_list))
    print(npy_list[0])
    count = 0
    file_issues = []
    for npy_obj in npy_list:
        filename = npy_obj.split('/')[-1].split('.npy')[0]
        # if count > 29:
        #     break
        print(count)
        if not Path(savepath+filename+'.npy').is_file():
            # print('filename')
            # print(filename)
            with open(npy_obj, 'rb') as n:
                npyrawcase = np.load(n, allow_pickle=True)

            roilist = []
            for rawob in npyrawcase:
                try:
                    # winc,winw = getcentrewidth(rawob)
                    # setattr(rawob,'WindowCenter',winc)
                    # setattr(rawob, 'WindowWidth',winw)
                    # print(winc,winw)
                    # oc = rawob.pixel_array
                    vximg = converttovoxel(rawob)
                    # print('OC',oc.min(),oc.max())
                    # print('VOX',vximg.min(),vximg.max())
                    # focvox = focusvoxel(vximg, -1350.0, 150)
                    # print('FVX',focvox.min(),focvox.max())
                    roilist.append(vximg)
                except Exception as e:
                    print(filename)
                    file_issues.append(filename)
                    pass
            # print(npyrawcase.shape)
            # print(npyrawcase[0])
            if npyrawcase.shape[0] != len(roilist):
                file_issues.append(filename)
                print('Mismatch')
            roisavepath = folder_dir + 'stage_2_preprocess_3/' + filename + '.npy'
            with open(roisavepath,'wb') as a:
                np.save(a, np.array(roilist))
        else:
            print('skipped')
        count = count + 1
    print(file_issues)

def sliceforNormal(folder_dir):
    df_master = pd.read_csv(folder_dir + 'output/' + 'stage_2_train_with_metadata_sorted2.csv')
    df_master = df_master.groupby(by='StudyInstanceUID')
    count = 0
    total_normal_patient = 0
    for name, group in df_master:
        # if count > 20:
        #     break
        count = count + 1
        lbl = list(group['any'].unique())
        if len(lbl)==1 and lbl[0] == 0:
            total_normal_patient = total_normal_patient + 1
            print(group['StudyInstanceUID'].unique()[0])
            print(len(group))
            # print(20 *'----')
            index = 0
            # print('kasdhfjlkashdf',group['StudyInstanceUID'].unique()[0])
            filename = group['StudyInstanceUID'].unique()[0]
            with open(folder_dir + 'stage_2_preprocess_3/' + filename +'.npy', 'rb') as n:
                process_npy = np.load(n, allow_pickle=True)
            labels = []
            slices  = []
            # if len(group) == len(process_npy):
            #     print('length matches')
            # else:
            #     print('mismatch')
            #     break
            i = 0
            while i < len(process_npy):
                indi_slice = []     
                if i == len(process_npy) - 1:
                    for j in range(-2,1,1):
                        # print(i+j)
                        indi_slice.append(process_npy[i+j])    
                    break
                elif i == len(process_npy) - 2:
                    for j in range(-1,2,1):
                        # print(i+j)
                        indi_slice.append(process_npy[i+j])
                    break
                else:
                    for j in range(0,3):
                        # print(i+j)
                        indi_slice.append(process_npy[i+j])
                i = i + j + 1
                slices.append(indi_slice)
                labels.append(6*[0])
                # print(10 *'-')
            # print(slices[6])
            # for x in range(len(slices)):
            #     print(len(slices[x]),labels[x])
            with open('/home/student/Akshat/Mini Project/' + '3DSampleNormal/' + filename+'_data' + '.npy', 'wb') as a:
                np.save(a,np.array(slices),allow_pickle=True)
            with open('/home/student/Akshat/Mini Project/' + '3DSampleNormal/' + filename+'_label' + '.npy', 'wb') as a:
                np.save(a,np.array(labels),allow_pickle=True)
            # break
        else:
            continue
    print(total_normal_patient)

def sliceForICH(folder_dir):
    slicedlist = []
    slicedlabels = []
    casesdcmobjlist = glob.glob(folder_dir + 'stage_2_preprocess_3/'+r'/*.npy')
    df_master = pd.read_csv(folder_dir +'output/' + 'stage_2_train_with_metadata_sorted2.csv')
    count = 0
    casesdcmobjlist.sort()
    print(len(casesdcmobjlist))
    # for i in range(0, 100):
    #     print(casesdcmobjlist[i] )
    # return

    for filename in casesdcmobjlist:
        # if count <= 21000:
        #     count = count + 1
        #     continue
        # if count > 2000:
        #     break
        print(count)
        count = count + 1
        print(filename)
        # npysample = np.load(filename)
        with open(filename, 'rb') as n:
            npysample = np.load(n, allow_pickle=True)
        seqid = filename.split('/')[-1].split('.npy')[0]
        print(seqid)
        tempdf = df_master[df_master['StudyInstanceUID'] == seqid].sort_values(['z_pos'])[['any']].values
        # print(np.where(tempdf == 1)[0])
        list_label_with_any = np.where(tempdf == 1)[0]
        if len(list_label_with_any) == 0:
            continue
        tempdf = df_master[df_master['StudyInstanceUID'] == seqid].sort_values(['z_pos'])
        # print(tempdf)
        list_label_visted = (max(list_label_with_any)+3)*[False]
        # print(len(list_label_visted))
        final_df = {'StudyInstanceUID': seqid, '3Dslice': [], 'any': [], 'edh': [], 'iph': [], 'ivh': [],
                    'sah': [], 'sdh': []}
        base_index = tempdf.index[0]
        print(base_index)
        # return
        for id in list_label_with_any:
            slice = []
            index = id
            any,edh,iph,ivh,sah,sdh = 0,0,0,0,0,0
            if list_label_visted[index] == False:
                if id == 0:
                    print('taking',end=' ')
                    for i in range(0, 3, 1):
                        if list_label_visted[index + i] == True:
                            break
                        any, edh, iph, ivh, sah, sdh = any | tempdf['any'][base_index + index + i], edh | tempdf['edh'][base_index + index + i], iph | tempdf['iph'][base_index + index + i], ivh | tempdf['ivh'][base_index + index + i], sah | tempdf['sah'][base_index + index + i], sdh | tempdf['sdh'][base_index + index + i]
                        # print('any',any, 'edh', edh, 'iph', iph, 'ivh', ivh, 'sah', sah, 'sdh', sdh)
                        slice.append(npysample[id + i])
                        list_label_visted[index+i] = True
                        print(id + i, end =' ')
                    id = id + 1
                    print()
                elif id == tempdf.index.size - 1 :
                    print('taking', end=' ')
                    for i in range(-2, 1, 1):
                        # if list_label_visted[index + i] == True:
                        #     break
                        print(id+i,end=' ')
                        list_label_visted[index + i] = True
                        any, edh, iph, ivh, sah, sdh = any | tempdf['any'][base_index + index + i], edh | tempdf['edh'][base_index + index + i], iph | tempdf['iph'][base_index + index + i], ivh | tempdf['ivh'][base_index + index + i], sah | tempdf['sah'][base_index + index + i], sdh | tempdf['sdh'][base_index + index + i]
                        # print('any', any, 'edh', edh, 'iph', iph, 'ivh', ivh, 'sah', sah, 'sdh', sdh)
                        slice.append(npysample[id + i])
                    id = id + 1
                    print()
                else:
                    print('taking', end=' ')
                    for i in range(-1,2,1):
                        if index + i  > -1 and index + i < len(list_label_visted) and list_label_visted[index + i] == True:
                            break
                        print(id+i, end=' ')

                        any, edh, iph, ivh, sah, sdh = any | tempdf['any'][base_index + index + i], edh | tempdf['edh'][base_index + index + i], iph | tempdf['iph'][base_index + index + i], ivh | tempdf['ivh'][base_index + index + i], sah | tempdf['sah'][base_index + index + i], sdh | tempdf['sdh'][base_index + index + i]
                        print('any', any, 'edh', edh, 'iph', iph, 'ivh', ivh, 'sah', sah, 'sdh', sdh)

                        if index + i  > -1 and index + i < len(list_label_visted):
                            list_label_visted[index + i] = True
                        slice.append(npysample[id+i])
                    # print()
                if slice != [] :
                    final_df['3Dslice'].append(slice)
                    final_df['any'].append(any)
                    final_df['edh'].append(edh)
                    final_df['iph'].append(iph)
                    final_df['ivh'].append(ivh)
                    final_df['sah'].append(sah)
                    final_df['sdh'].append(sdh)
            index = index + 1
            # print(index)
        final_slices = []
        # for j in range(0, len(final_df['3Dslice'])):
        #     print(final_df['StudyInstanceUID'],len(final_df['3Dslice'][j]),final_df['any'][j],final_df['edh'][j],final_df['iph'][j],final_df['ivh'][j],final_df['sah'][j],final_df['sdh'][j])
        for j in range(0, len(final_df['3Dslice'])):
            list1 = []
            for i in final_df:
                if i != 'StudyInstanceUID':
                    list1.append(final_df[i][j])
            final_slices.append(list1)

        # print(final_slices)
        # print(20 *'----')
        with open('/home/student/Akshat/Mini Project/' + '3DSampleICH/' + final_df['StudyInstanceUID'] + '.npy', 'wb') as a:
            np.save(a,np.array(final_slices),allow_pickle=True)
        slicedlist.append(final_df)

    # print(20*'---')

        #
        # begin = 0
        # end = npysample.shape[0]
        # # if ...
        #
        # slicedlist.append(npysample[1:4])
        # # slicedlabels.append()
        # slicedlist.append(npysample[7:10])


    # for i in slicedlist:
    #     with open(folder_dir+'3DSampleICH/'+i['StudyInstanceUID']+'.npy','wb') as a:
    #         np.save(a,np.array(i),allow_pickle=True)


        # break
        # for j in range(0,len(i['3Dslice'])):
        #     print(i['StudyInstanceUID'],len(i['3Dslice'][j]),i['any'][j],i['edh'][j],i['iph'][j],i['ivh'][j],i['sah'][j],i['sdh'][j])
    # for i in slicedlist:
    #     print(20*'-')
    #     for j in range(0,len(i['3Dslice'])):
    #         print(i['StudyInstanceUID'],len(i['3Dslice'][j]),i['any'][j],i['edh'][j],i['iph'][j],i['ivh'][j],i['sah'][j],i['sdh'][j])

    # slicedlist = np.array(slicedlist)
    # for i in slicedlist:
    #     print(20*'-')
    #     print(i['StudyInstanceUID'], i['3Dslice'][0],i['any'][0],i['edh'][0],i['iph'][0],i['ivh'][0],i['sah'][0],i['sdh'][0])
    # slicedlabels = np.array(slicedlabels)
    # print(slicedlist.shape)
    # print(slicedlist)
    # np.save('sudyinstid_x.npy',slicedlist)
    # np.save('sudyinstid_y.npy',slicedlabels)

def checkSlices(folder_dir):
    path = '/home/student/Akshat/Mini Project/'+  '3DSampleICH_1/'
    # slice_list = glob.glob(path + r'*.npy')
    # print(slice_list)
    # for i in slice_list:
    #     with open(i, 'rb') as n:
    #         npysample = np.load(n, allow_pickle=True)
    #     # np_object = np.load(i)
    #     for j in npysample:
    #         print(len(j))
    #     print(npysample)
    #     # print(npysample[1])
    #     print(40*'--')
    
    with open('/home/student/Akshat/Mini Project/'+  '3DSampleNormal/' +'ID_00047d6503_data.npy' , 'rb') as n:
        npysample = np.load(n, allow_pickle=True)       
    print(npysample[0].shape)




def correctingSliceFormat(folder_dir):
    path = folder_dir + '3DSampleICH/'
    slice_list = glob.glob(path + r'*.npy')
    # print(slice_list)
    count = 0
    for i in slice_list:
        slices = []
        labels = []
        # if count > 5:
        #     break
        count = count + 1
        filename = i.split('/')[-1].split('.npy')[0]
        print(filename)
        with open(i , 'rb') as n:
            npysample = np.load(n, allow_pickle=True)
        # part_labels[]
        for j in npysample:
            part_label = []
            for k in range(1,len(j)):
                part_label.append(j[k])
            labels.append(part_label)
            slices.append(j[0])
        with open('/home/student/Akshat/Mini Project/' + '3DSampleICH_1/' + filename+'_data' + '.npy', 'wb') as a:
            np.save(a,np.array(slices),allow_pickle=True)
        with open('/home/student/Akshat/Mini Project/' + '3DSampleICH_1/' + filename+'_label' + '.npy', 'wb') as a:
            np.save(a,np.array(labels),allow_pickle=True)
        # for x in range(len(slices)):
        #     print(len(slices[x]), labels[x])
        # print(20*'---')
        # slices.append()

# def prepareTrainData(folder_dir):
    

def verifypreprocess(folder_dir):
    npy_path = folder_dir + 'stage_2_preprocess_3/'
    npy_list = glob.glob(npy_path + r'/*.npy')
    df_csv = pd.read_csv('/home/student/Akshat/Mini Project/output/stage_2_train_with_metadata_sorted2.csv')
    # print(len(npy_list))
    print(npy_list[0])
    count = 0
    for npy_obj in npy_list:
        filename = npy_obj.split('/')[-1].split('.npy')[0]
        if count > 29:
            break
        count = count + 1
        print('filename')
        print(filename)
        print(npy_obj)
        # with open(npy_obj, 'rb') as n:
        npyrawcase = np.load(npy_obj)
        groups= df_csv.groupby(by= 'StudyInstanceUID')
        for name,group in groups:
            if filename == name:
                s = len(list(group['ID'].unique()))
                print(s)
        print(npyrawcase.shape)

        if npyrawcase.size == s:
            print('match')
        else :
            print('mismatch')

        # roilist = []
        # for rawob in npyrawcase:
        #     try:
        #         # winc,winw = getcentrewidth(rawob)
        #         # setattr(rawob,'WindowCenter',winc)
        #         # setattr(rawob, 'WindowWidth',winw)
        #         # print(winc,winw)
        #         # oc = rawob.pixel_array
        #         vximg = converttovoxel(rawob)
        #         # print('OC',oc.min(),oc.max())
        #         # print('VOX',vximg.min(),vximg.max())
        #         # focvox = focusvoxel(vximg, -1350.0, 150)
        #         # print('FVX',focvox.min(),focvox.max())
        #     except Exception as e:
        #         pass
        #
        # roilist.append(vximg)
        #
        # roisavepath = folder_dir + 'stage_2_train_preprocess_2/' + filename + '.npy'
        # with open(roisavepath,'wb') as a:
        #     np.save(a, np.array(roilist))


def create_csv_data_for_numpy(folder_dir):
    npy_path = folder_dir + 'stage_2_preprocess_3/'
    npy_list = glob.glob(npy_path + r'/*.npy')
    savepath = folder_dir
    print(len(npy_list))
    print(npy_list[0])
    count = 0
    df = pd.DataFrame()
    # df = pd.read_csv(folder_dir + 'numpy_normalize_data.csv' )
    # print(df.columns)
    # df2 = df.iloc[:,1:]
    # print(df2.columns)
    # df2.to_csv(savepath+'numpy_normalize_data1.csv',index=False)
    columns = {'StudyInstanceUID': [],'mean': [],'std': [],'min': [],'max': []}
    for npy_obj in npy_list:
        filename = npy_obj.split('/')[-1].split('.npy')[0]
        # if count > 10:
        #     break
        print(count)
        with open(npy_obj, 'rb') as n:
            npyrawcase = np.load(n, allow_pickle=True)
        columns['StudyInstanceUID'].append(filename)
        columns['mean'].append(npyrawcase.mean())
        # print('mean',npyrawcase.mean())
        columns['std'].append(npyrawcase.std(dtype=np.float32))
        # print('std', npyrawcase.std(dtype=np.float32))
        columns['min'].append(npyrawcase.min())
        # print('min',npyrawcase.min())
        columns['max'].append(npyrawcase.max())
        # print('max',npyrawcase.max())
        count = count + 1
    for col in columns:
        df[col] = columns[col]
    # print(df)
    df.to_csv(savepath+'numpy_normalize_data.csv',index=False)



try:
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    folder_dir = "/home/student/Akshat/Mini Project/"


    #STEP 1

    #dicomtonumpy(folder_dir) OR
    # savedicomtonumpy()

    # caselist = glob.glob(folder_dir+ 'stage_2_train_numpy/' + r'/*.npy' )
    # print(len(caselist))


    # STEP 2
    # savepreprocess(folder_dir)

    #STEP 3
    # create_csv_data_for_numpy(folder_dir)

    # verifypreprocess(folder_dir)

    #STEP 4
    # sliceForICH(folder_dir)
    checkSlices(folder_dir)
    # correctingSliceFormat(folder_dir)
    # sliceforNormal(folder_dir)
    # shallowslicing()

    # train_df_merged = pd.read_csv(folder_dir + 'stage_2_train_with_metadata1.csv',index_col=False)
    # # print(train_df_merged)
    # max_slices = 60
    # df_grp = train_df_merged.sort_values(['StudyInstanceUID','z_pos']).groupby(by='StudyInstanceUID').head(60)
    #
    # # print(df_grp)
    # # for name, group in df_grp:
    # #     group = group.sort_values(by=['z_pos'])
    # #
    # df_grp.to_csv(folder_dir + 'stage_2_train_with_metadata_sorted2.csv', index=False)
    # # ROOTFOLDER = ''


    # for dirtype in caselist:
    #     dirname = dirtype.split('/')[-2]
    #     cases = glob.glob(dirtype + r'*/')
    #     # print(dirtype,cases,dirname)
    #     savepathraw = './DataStore/pydicomObj/'
    #
    #     if not os.path.isdir(savepathraw + dirname + '/'):
    #         os.mkdir(savepathraw + dirname + '/')
    #
    #     for case in cases:
    #         caseno = case.split('/')[-2].split('Pat')[-1]
    #         # print('\n',2*'- -',caseno,2*'- -')
    #         dcmlist = glob.glob(case + r'/*.dcm')
    #         rawdcimlist = [dicom.read_file(dcmfile) for dcmfile in dcmlist]
    #         # Ordering the DCM
    #         rawdcimlist.sort(key=lambda x: int(x.InstanceNumber))
    #         # print(len(rawdcimlist))
    #         # #UnComment To save
    #         with open(savepathraw + dirname + '/Case-' + caseno + '.npy', 'wb') as a:
    #             np.save(a, np.array(rawdcimlist), allow_pickle=True)

except Exception as e:
    print(e)
    print(traceback.format_exc())
    pass