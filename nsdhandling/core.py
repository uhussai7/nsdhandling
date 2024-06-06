import nibabel as nib
import numpy as np
from scipy.io import loadmat
import os
from .config import *
import h5py
import torch
from tqdm import tqdm 
from .utils.roi_utils import update_mask,update_roi_dic,combine_basic_rois
from .utils.nsd_orig.load_nsd import load_betas,data_split,image_feature_fn
from pathlib import Path
from torchvision.transforms import Resize, Compose

#check how imgs_per_subject is stored
#and confusion about subject indices

def load_exp_design(A):
    """
    Load the experimental design file and get some useful arrays
    """ 
    A.exp_design   = loadmat(EXP_DESIGN_FILE)
    A.basic_cnt    = A.exp_design['basiccnt']
    A.shared_idx   = A.exp_design['sharedix']
    A.subject_idx  = A.exp_design['subjectim']
    A.trial_order  = A.exp_design['masterordering']
    A.stim_pattern = A.exp_design['stimpattern']

class NsdData:
    """
    Class for pairwise data
    """
    def __init__(self,subj_list,upto=4):
        self.subj_list=subj_list
        self.upto=upto
        load_exp_design(self)

    def load_from_raw(self,stim_data): #go through subjects and get the brain data + other stuff
        """
        Loads from raw betas and populates self.data attribute with a dictionary for the data
        """
        self.stim_data=stim_data
        self.data=[]
        for s_ind,subj in enumerate(self.subj_list):
            this_subj_brain_data=BrainData(subj,upto=self.upto)
            this_subj_brain_data.load_betas_raw()

            ordering=self.trial_order.flatten()-1

            trn_stim_data, trn_voxel_data,\
            val_stim_single_trial_data, val_voxel_single_trial_data,\
            val_stim_multi_trial_data, val_voxel_multi_trial_data = \
            data_split(image_feature_fn(self.stim_data.images_per_subject[s_ind].detach().cpu().numpy()),
                       this_subj_brain_data.voxel_data, 
                       ordering, imagewise=False)

            this_output_dictionary={
                'subject':subj,
                'train_stim': trn_stim_data,
                'val_stim_single': val_stim_single_trial_data,
                'val_stim_multi': val_stim_multi_trial_data,
                'train_vox':trn_voxel_data,
                'val_vox_single':val_voxel_single_trial_data,
                'val_vox_multi':val_voxel_multi_trial_data,
                'x':this_subj_brain_data.x,'y':this_subj_brain_data.y,'z':this_subj_brain_data.z,
                'voxel_ncsnr':this_subj_brain_data.voxel_ncsnr,
                'roi_dic':this_subj_brain_data.roi_dic,
                'voxel_mask':this_subj_brain_data.voxel_mask,
                'upto': this_subj_brain_data.upto,
                'Nv': this_subj_brain_data.Nv,
                'roi_dic_combined':this_subj_brain_data.roi_dic_combined
             }
            self.data.append(this_output_dictionary)
    
    def save_preprocessed(self,save_path=NSD_PREPROC):
        """
        Save data from rois only

        Args:
            save_path: path to save preprocessed data, default is NSD_PREPROC
        """
        Path(save_path+'/').mkdir(parents=True,exist_ok=True)

        for d in self.data:
            subj=int(d['subject'])
            this_path=os.path.join(save_path,"subj%02d"%subj)
            Path(this_path).mkdir(parents=True,exist_ok=True)
            for key in d.keys():
                this_this_path=os.path.join(this_path,key+'.npy')
                np.save(this_this_path,d[key])

    def load_preprocessed(self,load_path=NSD_PREPROC):
        """
        Load preprocessed (roi only) data, populates self.data

        Args:
            load_path: path to preprocessed data, default is NSD_PREPROC
        """
        self.data=[]
        for subj in self.subj_list:
            keys=os.listdir(load_path+ '/subj%02d'%int(subj))
            keys=[k.split('.npy')[0] for k in keys]
            this_dic={}
            for key in keys:
                this_dic[key]=np.load(load_path + '/subj%02d/'%int(subj) +key + '.npy',allow_pickle=True)
            self.data.append(this_dic)


class StimData:
    """
    Operations for raw stimulus (image) data 
    """
    def __init__(self):
        load_exp_design(self)

    def load_stimuli_raw(self,subj_list,size=(256,256),xfms=[]):
        """
        Load the raw img data

        Args:
            subj_list: List of subjects (e.g. [1,2])
            size: Size of output images (h,w) default (256,256)
            xfms: Any torchvision transforms to apply on the images
        
        Returns:
            images_per_subject: List of images array per subject
        """

        #create a transform for the target size and add any other xfms
        self.xfms=Compose(xfms + [Resize(size),])

        self.subj_list=subj_list
        images_per_subject=[]
        image_data = h5py.File(STIM_FILE,'r')['imgBrick']
        for subj in subj_list:
            subj=int(subj)-1 #just to ensure these are ints
            img_ids=self.subject_idx[subj]-1

            #h5py wants increasing order for indices
            arg_img_ids=img_ids.argsort() #args for img_ids
            extracted_image_data_sorted=image_data[img_ids[arg_img_ids]] #sorted extracted images
            extracted_image_data=np.zeros_like(extracted_image_data_sorted) #some where to store original order
            extracted_image_data[arg_img_ids]=extracted_image_data_sorted[:] #put in original order
            
            #append the list
            images_per_subject.append(self.xfms(torch.from_numpy(extracted_image_data.transpose(0,3,1,2))))
        
        self.images_per_subject=images_per_subject

        #need something here to save subject and resolution specific stims like before
        #maybe we consider loading the data as neeeded before each training session?

class BrainData: #this is done per subject as it carries fields for masks, etc. Can be wrapped in another class if needed
    """
    Operations for raw brain data per subject
    """
    def __init__(self,subj,upto=4):
        self.subj=int(subj)
        self.upto=upto 

        #mask preparation
        self.voxel_mask, self.voxel_ncsnr, self.roi_dic, self.roi_dic_combined,\
        self.x, self.y, self.z = self.prepare_masks(self.subj)


    def prepare_masks(self,subj,roi_list=['floc-bodies','floc-faces',
                                       'floc-places','floc-words',
                                       'prf-eccrois']):
        """
        A series of preparation steps for masks and rois

        Args: 
            subj: Subject id
            roi_list: List of rois to add in addition to basic rois

        Returns:
            voxel_mask: Flattened voxel mask
            voxel_ncsnr: Flattened noise ceiling array
            roi_dic: Roi dictionary (contains flattened masks)
            roi_dic_combined: Roi dictonary with basic rois combined
            x: x-coordinate
            y: y-coordinate
            z: z-coordinate
        """         

        #load some niftis
        basic_rois=nib.load(os.path.join(MASK_ROOT,'subj%02d'%subj,FUNC_RES,'roi','Kastner2015.nii.gz'))
        affine=basic_rois.affine
        basic_rois=basic_rois.get_fdata()
        voxel_mask_full=nib.load(os.path.join(MASK_ROOT,'subj%02d'%subj,FUNC_RES,'brainmask_inflated_1.0.nii')).get_fdata()
        prf=nib.load(os.path.join(MASK_ROOT,'subj%02d'%subj,FUNC_RES,'roi','prf-visualrois.nii.gz')).get_fdata()
        ncsnr_full=nib.load(os.path.join(BETA_ROOT,'subj%02d'%subj,FUNC_RES,FUNC_PREPROC,'ncsnr.nii.gz')).get_fdata()
        
        #some assurance steps from original authors
        basic_rois[prf>0]=prf[prf>0] 

        #create an initial mask
        mask=np.zeros_like(basic_rois).astype(bool)
        mask[basic_rois>0]=1
        mask[voxel_mask_full==0]=0
        nib.save(nib.Nifti1Image(basic_rois,affine),os.path.join(MASK_ROOT+"/subj%02d"%subj,FUNC_RES,'roi/basic_rois.nii.gz'))

        #roi steps and update mask
        roi_list=['basic_rois'] + roi_list #basic rois are added by default
        mask=update_mask(mask,roi_list,subj) #this ensures rois are part of mask 
        roi_dic=update_roi_dic(mask,{},roi_list,subj)
        roi_dic_combined=combine_basic_rois(roi_dic,roi_list,subj)

        #make the voxel mask
        voxel_mask = np.nan_to_num(mask).flatten().astype(bool)

        #apply mask to noise ceiling
        voxel_ncsnr = ncsnr_full.flatten()[voxel_mask]

        #get the x,y,z coordinates
        x = np.arange(0,basic_rois.shape[0])
        y = np.arange(0,basic_rois.shape[1])
        z = np.arange(0,basic_rois.shape[2])
        xx,yy,zz = np.meshgrid(x,y,z,indexing='ij')
        x = xx.flatten()[voxel_mask]
        y = yy.flatten()[voxel_mask]
        z = zz.flatten()[voxel_mask]


        return voxel_mask, voxel_ncsnr, roi_dic, roi_dic_combined, x, y, z
    
    def load_betas_raw(self,zscore=True,upto=None):
        """
        Loads the beta sessions into self.voxel_data and number of voxels into self.Nv

        Args:
            zscore: Z-score data or not
            upto: Number of sessions to load (if passed, preference given to this over the class level variable, upto)
        """

        #handle upto variable
        if upto==None:
            upto=self.upto

        beta_subj = os.path.join(BETA_ROOT,"subj%02d/" % (self.subj,),FUNC_RES,FUNC_PREPROC) + '/'
        print('Loading raw beta files, upto %d sessions from path:%s'%(upto,beta_subj))
        self.voxel_data, filenames = load_betas(folder_name=beta_subj, zscore=zscore,voxel_mask=self.voxel_mask,
                                           up_to=upto,load_ext=LOAD_EXT)
        print('done loading, shape is:',self.voxel_data.shape)

        #get shape
        self.data_size, self.Nv = self.voxel_data.shape

