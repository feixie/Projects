
import os, time, sys

from scipy import *
from numpy import *
import Image

from random import random, shuffle
import scipy.signal
conv = scipy.signal.convolve
conv2 = scipy.signal.convolve2d

from scipy.misc import toimage, fromimage

from numpy.fft import fft, ifft, fft2, ifft2, fftshift

from v1s import V1S

from v1s_funcs import *
from v1s_math import *
from mathelp import *
import pickle

# -----------------------------------------------------------------------------
class v1NN(object):

    """
    Do v1 recognition using nearest neighbor on either clear or motion
    blurred probe images. 
    
    The main function is get_performance. Feeding get_performance a path to 
    a list of images(probes and gallery) and a path to a list parameters
    will print out recognition rate and a few diagnostic data
    

    """


    # -------------------------------------------------------------------------
    def __init__(self):
        """ Constructor
          
        """
       
        #initialize global variables:
        
        #number of elements in each category
        self.element = 4
        
        self.face_list = None
        self.category_list = None
        self.filelists_dict = None
        self.probe_blength_bangle = None
        
        #randomize gallery toggle
        self._rand_gallery = True
        
        #take gallery set radius into account toggle
        self._adjust_radius = False
    
        #apply blur toggle
        self._apply_blur = True
        #randomize or pre-set blur angle toggle
        self._rand_bangle = False
        self.bangle = 45
        #randomize or pre-set blur length toggle
        self._rand_blength = False
        self.blength = 15
        
        #test controls: 
        #noweight = 0, preweight = 1, postweight = 2
        self._pre_post_weight = 0
        #binary = 0, precision = 1
        self._bin_prec_weight = 0
       
        
        
        #dimension of images 
        self.face_x = None
        self.face_y = None
        self.blur_resp = None
        
        self.gabor_resp_number = None
        
        #might not need:
        self.gabor_resp_length = None
       
   
       
    def get_performance(self, image_list, param_path):
    
        """ Compute recognition performance after running a trial for each
        category (takes first three of a category as gallery and the fourth 
        as probe)
        
        Inputs:
            image_list -- path to list of images (path)
            param_path -- path to parameter dictionary (path)
        
        Outputs:
            
            performance -- the percentage of trials that computed category 
                           correctly
        
        """
        
    
        face_list = []
        cate = []
        
        #get relevant parameters from param_path
        v1s_params = {}
        execfile(param_path, {}, v1s_params)
        
        model = v1s_params['model']
        rep = v1s_params['representation']
        filt_param = rep['filter']
        pca_threshold = v1s_params['pca_threshold']
        
        params = model[0][0]
        featsel = model[0][1]
        filt_params = params['filter']
        
        
        #parse image list from image_path
        
        file = open(image_list)
        while 1:
            line = file.readline()
            if not line:
                break
            line = line[:-1]
            print line
            face_list += [ line ]
            pass # do something
            
        file.close()
        self.face_list = face_list
        list_size = size(face_list)
        
        #populate a filelist dictionary and a list of 
        #feature vectors for each image in the filelist
        
        filelists_dict = self.get_filelist_dict(face_list)
        self.filelists_dict = filelists_dict
       
        
        #get category list
        cate_list01 = filelists_dict.keys()
        cate_list01.sort()
        
        cate_list = []
        for i in cate_list01:
            cate = i[-5:]
            cate_list += [cate]
      
        self.category_list = cate_list
        
        #apply probe blur and change filelists_dict to point to blurred probe
        if(self._apply_blur):
            filelists_dict = self.apply_blur_to_probe(filelists_dict, params)
            
            
        
        #compute feature vectors
        v1s = V1S()
        fvectors, flabels, fnames = \
                            v1s.get_fvectors_flabels_fnames(filelists_dict,
                                                              model, 0, list_size)
        
        #break out into wavelets each with pca applied
        #have pca keep more if blur is applied
        n_freq = len(filt_params['freqs'])
        n_orient = len(filt_params['orients'])
        n_wavelet = n_freq*n_orient

        #if self._apply_blur:
        #    pca_threshold = 0.9
        #else:
        #    pca_threshold = 0.7
        
        #NOTE: change pca_threshold to reflect number of images
        pca_threshold = 1   
        
        fvectors = self.into_reduced_wavelets(fvectors, n_wavelet, pca_threshold)
        #fvectors is now a list of matrices
        #each matrix is an fvector segment corresponding to a particular wavelet
        

        #take 1st fvector of each category as probe and the rest as gallery
        probe = []
        gallery = []
        
        for one_wave in fvectors:   
            probe_image_wave = []
            gallery_image_wave = []
            for j in xrange(list_size):
                this_image_wave = one_wave[j,:]
                
                if j%self.element == 0:
                    probe_image_wave += [this_image_wave]
                   
                else:
                    gallery_image_wave += [this_image_wave]
                   
            probe += [probe_image_wave]
            gallery += [gallery_image_wave]
        
  
        
        #get performance score
        score = self.compute_score(probe, gallery, filt_params, pca_threshold)
        
        print "blurs are:"
        print self.probe_blength_bangle
        
        
  
        return score   
        
        
    def compute_score(self, probe, gallery, params, pca_threshold):
        """ Compute the recognition score of a list of probes 
        against a list of gallery images
         
        Inputs:
            probe -- a list of probe fvectors (list)
            gallery -- a list of gallery fvectors (list)
            params -- a dictionary of parameters (dict)
        
        Outputs:
            score -- the recognition score of this probe and gallery lists
        """
        
        #get the number of probes and gallery
        n_probe = len(probe[1])
        n_gallery = len(gallery[1])
        
        
        #matrix to store comparison data in:
        
        #computed_cate = []
        computed_dist = []
        for i in xrange(n_probe):
            
            #weight probe and gallery
            if self._apply_blur and self._pre_post_weight > 0:
                scale = self.get_blur_based_weight(self.probe_blength_bangle[i], params)  
                
                #pre weight
                if self._pre_post_weight == 1:
                    probe, gallery = self.pre_weight(probe, gallery, scale)     
                
            
            
            n_wavelet = len(probe)
            print "num of wavelets"
            print n_wavelet
            
            
            
            #for each item in gallery
            for j in xrange(n_gallery):
                
                gal_dist = []
                #for each wavelet compute distance and add to dist_sum
                for k in xrange(n_wavelet):
                
                    one_dist = self.compute_euclidean_d(probe[k][i], gallery[k][j])
                    gal_dist += [one_dist]
                    
                  
        
                computed_dist += [gal_dist]
        
            
        #print "size of computed dist is"
        #print shape(computed_dist)
       
        scores = divide(1., computed_dist)
        
        
        norm_scores = scores
        
        
        #post weight
        if self._apply_blur and self._pre_post_weight == 2:
            
            weighted_scores = self.post_weight(norm_scores, scale)
            
        else:
            #just sum score for each gallery item
            weighted_scores = norm_scores
            
          
            
        
        sum_score = []    
        for i in weighted_scores:
            w_sum = sum(i)
            sum_score += [w_sum]
            
        print "sum score are"
        print sum_score
             
        
        computed_cate = []
        for i in xrange(n_probe):
        
            p_scores = sum_score[i*n_gallery:i*n_gallery+n_gallery]
            print "p scores are"
            print p_scores
          
            max_score = max(p_scores)
            max_index = p_scores.index(max_score)
            
            
            print "max score is"
            print max_score
            print "max index is"
            print max_index
        
       
            cate_index = max_index/(self.element-1)
           
            category = self.category_list[cate_index]
            computed_cate += [category]
            
  
            
        #write diagnostic text to file
        f = open('test_results.txt','w')

        f.write('blur lengths and angles are \n')
        f.write(str(self.probe_blength_bangle))
        
        f.write('correct categories are \n')
        f.write(str(self.category_list))
        f.write('\n')
        f.write('calculated categories are \n')
        f.write(str(computed_cate))
        f.write('\n')
        print "correct categories are"
        print self.category_list
        print "calculated categories are"
        print computed_cate
        
        #compute performance
        num_correct = 0.
        n_cate = size(computed_cate)
        for i in xrange(n_cate):
            if self.category_list[i] == computed_cate[i]:
                num_correct += 1.
                
        perf = num_correct/n_cate
        
        f.write('performance is \n')
        f.write(str(perf))
        f.write('\n')
        print "performance is"
        print perf
        
        f.close()
        
        #return performance
        
        return perf
        
        
    def znorm(self, set):
        set_mean = mean(set)
        set_std = std(set)
        
        norm_set = divide(subtract(set, set_mean), set_std)
        
       
        return norm_set
   
    def post_weight(self, score, scale):
    
        w_score = []
  
        for compare in score:
            weighted = multiply(compare, scale)
            w_score += [weighted]
        print "shape of w_score is"
        print shape(w_score)
        
        return w_score
        
    def compute_radius(self, set):
        """ Compute the radius of a category of images
         
        Inputs:
            set -- a list of gallery vectors (list)
        
        Outputs:
            avg -- the average distance between each gallery vector 
        """
    
        #compute the distance between each gallery vector and every other
        #find the average and use that as radius
        set_n, set_size = shape(set)
        sum = 0.
        
        for i in xrange(set_n):
            for j in range(i+1,set_n):
              
                distance = self.compute_euclidean_d(set[i],set[j])
                sum += distance
                
        avg = sum/set_n
        
        return avg
        
        
    def pre_weight(self, probe, gallery, scale):
    
        """
        """
        
        scaled_probe = [item for i, item in enumerate(probe) if scale[i] == 1]
        scaled_gallery = [item for i, item in enumerate(gallery) if scale[i] == 1]
       
                
        weighted = (scaled_probe, scaled_gallery)
        
        
        #what number would pca most ignore? 1? 0?
            
            
        
        return weighted
        
   
        
    def get_filelist_dict(self, face_list):
        """ Parse list of face images into a filelist dictionary
         
        Inputs:
            face_list -- a list of paths to face images (list) 
        
        Outputs:
            filelists_dict -- a dictionary of image paths organized by category (dict)
        """
    # -- Organize images into the appropriate categories
        cats = {}
        for f in face_list:
            cat = "/".join(f.split('/')[:-1])
            name = f.split('/')[-1]
            if cat not in cats:
                cats[cat] = [name]
            else:
                cats[cat] += [name]

                
        # -- Shuffle the images into a new random order
        filelists_dict = {}
        seed = 1
        for cat in cats:
            filelist = cats[cat]
            if self._rand_gallery:
                random.seed(seed)
                random.shuffle(filelist)
                seed += 1
            filelist = [ cat + '/' + f for f in filelist ]
            filelists_dict[cat] = filelist
            
        return filelists_dict
        
    def into_reduced_wavelets(self, fvector, n_wavelet, pca_threshold):
        """
        """
        
        n_fvector, size_fvector = shape(fvector)
        w_size = size_fvector / n_wavelet
        w_fvector = []
        
        
        #for one_fvec in fvector:
        #    sep_fvec = []
        #    for i in xrange(n_wavelet):
        #        one_wave = one_fvec[i*w_size:i*w_size+w_size]
        #        sep_fvec += [one_wave]
        #    w_fvector += []    
            
        Fvector = matrix(fvector)
        for i in xrange(n_wavelet):
            one_wave = Fvector[:, i*w_size:i*w_size+w_size]
            pca_wave = self.pca(one_wave, pca_threshold)
            w_fvector += [pca_wave]
            
       
        #w_fvector is now a list of matrices
       
        return w_fvector
        
    def pca(self, fvectors, pca_threshold):
        """ Reduce dimensionality using a pca / eigen subspace projection
         
        Inputs:
            fvectors -- list of a list of feature vector for each image (list)
            pca_threshold -- pca threshold 
        
        Outputs:
            fvectors -- list of a reduced list of feature vector for each image (list)
        """
    
        fvectors = squeeze(fvectors)
        nvectors, vsize = fvectors.shape    
        if nvectors < vsize:
            print "pca...", 
            print fvectors.shape, "=>", 
            U,S,V = fastsvd(fvectors)
            
            eigvectors = V.T
            
            i = tot = 0
            S **= 2.
            
           
            while (tot <= pca_threshold) and (i < S.size):
                tot += S[i]/S.sum()
                i += 1
                
                
            eigvectors = eigvectors[:, :i+1]
            fvectors = dot(fvectors, eigvectors)
            print fvectors.shape
        
        return fvectors
    
        
    def compute_euclidean_d(self, probe_vec, gallery_vec, radius = 1.):
    
        """ Computes the euclidean distance between probe_vec and gallery
         
        Inputs:
            probe_vec: feature vector of the probe (list)
            gallery_vec: feature vector of the gallery image (list)
        
        Outputs:
            sum_sqrt: distance between probe_vec and gallery_vec 
        
        """
        #check that the two vectors are the same size
        p_vec_n = size(probe_vec)
        g_vec_n = size(gallery_vec)
        sum = 0.
       
        probe_vec = array(probe_vec)
        gallery_vec = array(gallery_vec)
        probe_vec = squeeze(probe_vec)
        gallery_vec = squeeze(gallery_vec)
        
        if p_vec_n != g_vec_n:
            print "ERROR: probe and gallery vectors are not the same size"
        
       
        else:
            for i in xrange(p_vec_n):
                diff_sq = (probe_vec[i] - gallery_vec[i])**2
                sum = sum + diff_sq
                
            sum_sqrt = sqrt(sum)
            distance = sum_sqrt/radius
        
        return distance
    
    def get_blur_based_weight(self, length_angle, params):
        
        g_freq = params['freqs']
        g_orient = params['orients']
        x = self.face_x
        y = self.face_y
        
        center_freq = []
        
        
        b_length, b_theta = length_angle
        h = filt_mblur(b_length, b_theta)
        H = psf2otf(h, (y,x))
        
        
        blur_resp_at_center = []
        scale = []
        
       
        #calculate center frequencies for each gabor wavelet
        #and blur response at corresponding centers
        for f in g_freq:
            for o in g_orient:
                
                if o <= pi/2:
                    y1 = round(f*x*sin(o))
                    x1 = round(f*y*cos(o))
                    y2 = round(x - f*x*sin(o))
                    x2 = round(y - f*y*cos(o))
                    
                else:
                    o = o - pi/2
                    y1 = round(x - f*x*cos(o))
                    x1 = round(f*y*sin(o))
                    y2 = round(f*x*cos(o))
                    x2 = round(y - f*y*sin(o))
                    
                center = (x1,y1,x2,y2)
                # get blur response at centers
                
                if x1<y and y1<x:
                    center_resp1 = H[x1][y1]  
                    
                else:
                    center_resp1 = 1
                    
                 
                    
                if x2<y and y2<x:
                    center_resp2 = H[x2][y2]
                  
                else:
                    center_resp2 = 1
                   
                this_cen = [real(center_resp1),real(center_resp2)]
                blur_resp_at_center += [this_cen]
                
            
                #binary weighting:
                if self._bin_prec_weight == 0:
                    if center_resp1 > 0.9 and center_resp2 > 0.9:
                        scale += [1]
                    else:
                        scale += [0]
        
        #precision weighting
        if self._bin_prec_weight == 1:
            
          
            avg = [sum(item)/2. for item in blur_resp_at_center]
         
            scale = avg
          
        
        return scale
        
    def apply_blur_to_probe(self, filelists_dict, params):
        """ Applies blur to probe image then replace path to clear probe 
            in filelists_dict with path to blurred probe.
        
        Inputs:
        filelists_dict -- a dictionary of image paths organized by category (dict)
        params -- parameter dictionary (dict)
        
        Outputs:
        filelists_dict -- a dictionary of image paths organized by category (dict)
        
        """
        
        #take first image in each category as probe 
        keys = filelists_dict.keys()
        keys.sort()
        
        lengths = [10, 15, 20]
        angles = [i*11.25 for i in xrange(16)]
        
        probe_blength_bangle = []
        
        for i in keys:
            probe_image = filelists_dict[i][0]
            # -- get image as an array
            orig_imga = get_image(probe_image, params['preproc']['max_edge'])

            # -- 0. preprocessing
            imga0 = orig_imga.astype('f') / 255.0        
            if imga0.ndim == 3:
                # grayscale conversion
                imga0 = 0.2989*imga0[:,:,0] + \
                        0.5870*imga0[:,:,1] + \
                        0.1140*imga0[:,:,2]
                        
            #generate random blur length and angle
            if self._rand_blength:
                shuffle(lengths)
                b_length = lengths[0]
            else:
                b_length = self.blength
            
            if self._rand_bangle:
                shuffle(angles)
                b_angle = angles[0]
            else:
                b_angle = self.bangle
            
                
            blength_bangle = (b_length,b_angle)
            probe_blength_bangle += [blength_bangle]
            
            #blur image
            blurred = self.blur_image(imga0, b_length, b_angle)
            
            #save blurred image 
            path = probe_image[:-4]
            str1 = '_blur'
            str2 = '.pgm'
            new_path =path+str1+str2
            toimage(real(blurred)).save(new_path)
            
            #modify filelists_dict to list blurred probes
            filelists_dict[i][0] = new_path
        
        self.probe_blength_bangle = probe_blength_bangle
        
        return filelists_dict
        
    def blur_image(self, image, b_length, b_theta):
    
        """ Return a blurred image
        
        Inputs:
        image -- x by y image (gabor wavelet in this case) (list)
        b_length -- length of blur
        b_theta -- angle of blur
        
        Outputs:
        blurred -- blurred x by y image (list)
        
        """
        
        x,y = shape(image)
        #x and y switched because the result of shape refers to 
        #columns and rows 
        self.face_y = x
        self.face_x = y
        h = filt_mblur(b_length, b_theta)
        H = psf2otf(h, (x,y))
        self.blur_resp = H
        
        Image = fft2(image)
        Blurred = multiply(H,Image)
        blurred = ifft2(Blurred)
        
        return blurred
        
    
        
        
    
        
   
    
  

        
        
        
        
        
   
       

        

