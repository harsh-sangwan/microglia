import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors 
from mpl_toolkits.mplot3d import Axes3D # necessary to use ax = fig.gca(projection='3d')


#%% 
def voxelintensity(vol, threshold=0.01, color='g', mask=None, maskedcolor='y', scaling='sinoid', signalcap='auto', ahandle=None):
    ''' 
    voxelintensity(vol, threshold=0.01, color='g', mask=None, maskedcolor='y', scaling='sinoid')
    
    Takes a volume and plots all voxels by representing their signal intensity as opaqueness.
    
    Optional inputs:
        * threshold - Given as fraction of peak signal (i.e. between 0 and 1). Voxels below the threshold will not be plotted to decrease computational load unless they are masked.
        * color - Color of the voxel faces & edges. Please choose from r, g, b, y, c, m, w, and k.
        *  mask - Volume with boolean values with shape identical to volume. Voxels in the volume that correspond to a TRUE value in mask will be visually highlighted.
        * maskedcolor - Same as color, but for masked voxels.
        * signalcap - Signal value (abs. number) that corresponds to full opacity; if at 'auto', the signal peak will be chosen
        * scaling - Choose between 'linear' and 'sinoid' (recommended) scaling of signal to opaqueness
        * ahandle - handle to 3D axes of figure onto which to plot; if None, a new figure will be created
    '''

    if(len(vol.shape) != 3): raise ValueError('Please provide 3D volume')
    
    # Scale signal according to definition of full opacity (signalcap)
    if(signalcap=='auto'):
        vol = vol/np.max(vol) # normalize to [0,1]
    else:
        vol = vol/signalcap
    # Transform signal to transparency
    if(scaling=='sinoid'):
        # The sinoid transformation is crucial to ensure that visual impression matches intuitive interpretation
        # Otherwise, voxels with very small signal value would appear overproportionally bright
        alpha = 0.5*(1+np.sin(np.pi*(vol-0.5))) # apply scaled sine to enhance contrast and scale back to [0,1]
    else:
        alpha = vol
    
    # Determine colors for voxels
    C = np.zeros([3])
    if(color=='r' or color=='w'): C[0] = 1
    if(color=='g' or color=='w'): C[1] = 1
    if(color=='b' or color=='w'): C[2] = 1
    if(color=='y'): C[0:2] = 1
    if(color=='c'): C[1:3] = 1
    if(color=='m'): C[0] = 1;  C[2] = 1; 
    fcolors = np.zeros(alpha.shape + (4,)) 
    fcolors[:,:,:,0] = C[0] # red value
    fcolors[:,:,:,1] = C[1] # green value
    fcolors[:,:,:,2] = C[2] # blue value
    fcolors[:,:,:,3] = alpha # alpha value between 0 and 1
    ecolors = 0.5*fcolors # make edge darker than face
    #    ecolors[:,:,:,3] = 2*ecolors[:,:,:,3]
    
    # Determine colors for masked voxels
    if(mask is not None):
        if(vol.shape != mask.shape): raise ValueError('Please make sure mask has same size as volume')
        points = np.asarray(np.where(mask==1))
        C = np.zeros([3])
        if(maskedcolor=='r'): C[0] = 1
        if(maskedcolor=='g'): C[1] = 1
        if(maskedcolor=='b'): C[2] = 1
        if(maskedcolor=='y'): C[0:2] = 1
        if(maskedcolor=='c'): C[1:3] = 1
        if(maskedcolor=='m'): C[0] = 1;  C[2] = 1; 
        # adjust face colors for masked voxels
        fcolors[points[0,:],points[1,:],points[2,:],0] = C[0] # red value
        fcolors[points[0,:],points[1,:],points[2,:],1] = C[1] # green value
        fcolors[points[0,:],points[1,:],points[2,:],2] = C[2] # blue value
        # and give them solid, black edges
        ecolors[points[0,:],points[1,:],points[2,:],:] = [0,0,0,0.5]
    
    # Determine threshold of voxels that should be rendered
    if(type(threshold)==str and 'std' in threshold):
        thr = np.mean(alpha) + int(threshold.replace('std','')) * np.std(alpha)
    else:
        thr = threshold
    canvas = np.zeros(vol.shape,np.bool)
    # show points where signal is above threshold
    points = np.asarray(np.where(alpha>thr)) 
    canvas[points[0,:],points[1,:],points[2,:]] = True
    # but also those points that were masked - regardless of their signal level
    if(mask is not None):
        points = np.asarray(np.where(mask))
        canvas[points[0,:],points[1,:],points[2,:]] = True
    
    # Create plot
    if(ahandle is None):
        fhandle = plt.figure()
        ahandle = fhandle.gca(projection='3d')
    elif(hasattr(ahandle, 'get_zlim')):
        plt.sca(ahandle)
    else:
        raise ValueError('Please provide handle to 3D axis')
        
    voxelplot(ahandle, canvas, fcolors, ecolors)
    ahandle.set_ylabel('Y-axis')
    ahandle.set_xlabel('X-axis')
    ahandle.set_zlabel('Z-axis')
    plt.show()


#%%
def voxelplot(ahandle, canvas, fcolors, ecolors):
    ''' 
    voxelplot(ahandle, canvas, fcolors, ecolors)
    
    Works just like matplotlib's voxel() function, but without the bug that internal faces
    of voxels are not being rendered. This is achieved by first plotting one half of voxels,
    then the other. The split is chosen such that in each plotting, no voxel shares a face
    with any other voxel.
    
    IMPORTANT: matplotlib usually works in x,y,z order. To be consistent with numpy that works
    in y,x,z order, this function takes y,x,z volumes and plots them correctly as x,y,z.
    
    Use as in following example:
        * fig = plt.figure()
        * ax = fig.gca(projection='3d')
        * plotting.voxelplot(ax, canvas, fcolors, ecolors) 
        * #normally: ax.voxels(canvas, facecolors=fcolors, edgecolors=ecolors)
        * plt.show()
        
    Input values:
        * ahandle - handle to axes of figure
        * canvas - 3D volume with boolean values. TRUE voxels will be plotted
        * fcolors - Defines face color of voxels. Either 1D array in format [r, g, b, alpha] or a 4D volume with one such array per voxel
        * ecolors - The same as fcolors, but for the edges.
    '''
    # flip X and Y axes of all inputs
    canvas = np.swapaxes(canvas, 0, 1)
    fcolors = np.swapaxes(fcolors, 0, 1)
    ecolors = np.swapaxes(ecolors, 0, 1)
    # build checkered box to be able to plot all voxels in 2 steps
    x,y,z = canvas.shape
    x2 = x + 1 - np.remainder(x,2) # make uneven
    y2 = y + 1 - np.remainder(y,2)
    z2 = z + 1 - np.remainder(z,2)
    checkeredbox = np.reshape(np.remainder(np.arange(x2*y2*z2),2),[x2,y2,z2])
    checkeredbox = checkeredbox[0:x,0:y,0:z] #crop to orignal size
    # Plot first half of voxels
    points = np.asarray(np.where(checkeredbox==0))
    checkeredcanvas = np.zeros(canvas.shape,np.bool)
    checkeredcanvas[points[0,:],points[1,:],points[2,:]] = canvas[points[0,:],points[1,:],points[2,:]]
    ahandle.voxels(checkeredcanvas, facecolors=fcolors, edgecolors=ecolors)
    # Plot second half of voxels
    points = np.asarray(np.where(checkeredbox==1))
    checkeredcanvas = np.zeros(canvas.shape,np.bool)
    checkeredcanvas[points[0,:],points[1,:],points[2,:]] = canvas[points[0,:],points[1,:],points[2,:]]
    ahandle.voxels(checkeredcanvas, facecolors=fcolors, edgecolors=ecolors)
    ahandle.set_aspect('equal')


#%%
def projections(vol, mask=None, ahandles=None, cap=None, scaling='sinoid', mode='max', outline=False):
    '''     
    projections(vol, mask=None, ahandles=None, cap=None, scaling='sinoid', mode='max', outline=False)
    
    Plots 3 projections, one per dimension, in a 2x2 subplot (green colorscale). By default,
    this function shows maximum intensity projections. The volume should be within [0,1]
    
    Optional inputs:
        * mask - Masked voxels will be shown as yellow; must be boolean volume
        * ahandles - A list of 3 axis handels on which to plot
        * cap - Either as [minval, maxval] or as 'Xstd' (where X is an int) if signal is to be capped
        * scaling - Scaling of signal to available dynamic range (either 'sinoid' or 'linear')
        * mode - Set to 'cum' for cummulative projections instead of max intensity
        * outline - Set to True if background should be shaded in blue to highlight outline of mask
    '''
    if(len(vol.shape) != 3): raise ValueError('Please provide 3D volume')
    
    # Define parameters for plot
    if(cap == None):
        pass
    elif('std' in cap):
        maxsignal = int(cap.replace('std','')) * np.std(vol)
        minsignal = np.min([0, np.min(vol)])
        vol = np.clip(vol,minsignal,maxsignal)
    elif(len(cap) == 2):
        vol = np.clip(vol,cap[0],cap[1])
    if(scaling=='sinoid'):
        # The sinoid transformation is crucial to ensure that visual impression matches intuitive interpretation
        # Otherwise, voxels with very small signal value would appear overproportionally bright
        vol = 0.5*(1+np.sin(np.pi*(vol-0.5))) # apply scaled sine to enhance contrast and scale back to [0,1]
    
    # Plot projections
    if(ahandles is None): 
        fhandle = plt.figure()
        ax1 = fhandle.add_subplot(2,2,1)
        ax3 = fhandle.add_subplot(2,2,3)
        ax4 = fhandle.add_subplot(2,2,4)
        ahandles = [ax1, ax3, ax4]
    #plt.figure(fhandle.number)
    dims = ['Z','Y','X']
    for d in dims:
        if(d=='Z'): 
            if(mode=='cum'):
                volprojection = np.clip(np.sum(vol,2),0,1)
            else:
                volprojection = np.max(vol,2)
            if(mask is not None):
                maskprojection = np.max(mask,2)
            current_ax = ahandles[0]
            current_ax.set_xlabel('X-axis')
            current_ax.set_ylabel('Y-axis')
        elif(d=='Y'): 
            if(mode=='cum'):
                volprojection = np.clip(np.sum(vol,0),0,1)
            else:
                volprojection = np.max(vol,0)
            if(mask is not None):
                maskprojection = np.max(mask,0)
            current_ax = ahandles[1]
            current_ax.set_xlabel('Z-axis')
            current_ax.set_ylabel('X-axis')
        elif(d=='X'): 
            if(mode=='cum'):
                volprojection = np.clip(np.sum(vol,1),0,1)
            else:
                volprojection = np.max(vol,1)
            if(mask is not None):
                maskprojection = np.max(mask,1)
            current_ax = ahandles[2]
            current_ax.set_xlabel('Z-axis')
            current_ax.set_ylabel('Y-axis')
        rgb = np.zeros([volprojection.shape[0],volprojection.shape[1],3])
        rgb[:,:,1] = volprojection
        if(mask is not None):
            rgb[:,:,0] = maskprojection * volprojection
            if(outline):
                rgb[:,:,2] = 1-maskprojection  # --> Make background blue to inspect oversegmentation
        current_ax.imshow(rgb)
        current_ax.set_title(d + '-projection')

#%%
def orthoslices(vol, mask=None, ahandles=None, intersection='center', cap=None, scaling='linear', outline=False):
    '''     
    orthoslices(vol, mask=None, ahandles=None, intersection='center', cap=None, scaling='linear', outline=False)
    
    Plots 3 orthogonal slices, one per dimension, in a 2x2 subplot (green colorscale). By default,
    the slices intersect at the center. The volume should be within [0,1]
    
    Optional inputs:
        * mask - Masked voxels will be shown as yellow; must be boolean volume
        * ahandles - A list of 3 axis handels on which to plot
        * intersection - Set to 'peak' if slices should intersect at voxel of peak signal
        * cap - Either as [minval, maxval] or as 'Xstd' (where X is an int) if signal is to be capped
        * scaling - Scaling of signal to available dynamic range (either 'sinoid' or 'linear')
        * outline - Set to True if background should be shaded in blue to highlight outline of mask
    '''
    if(len(vol.shape) != 3): raise ValueError('Please provide 3D volume')
    
    # Define parameters for plot
    if(intersection == 'peak'):
        point = np.where(vol[:,:,:] == np.max(vol[:,:,:]))
    else:
        point = [int(np.round(vol.shape[0]/2)), int(np.round(vol.shape[1]/2)), int(np.round(vol.shape[2]/2))]
    if(cap == None):
        pass
    elif('std' in cap):
        maxsignal = int(cap.replace('std','')) * np.std(vol)
        minsignal = np.min([0, np.min(vol)])
        vol = np.clip(vol,minsignal,maxsignal)
    elif(len(cap) == 2):
        vol = np.clip(vol,cap[0],cap[1])
    if(scaling=='sinoid'):
        # The sinoid transformation is crucial to ensure that visual impression matches intuitive interpretation
        # Otherwise, voxels with very small signal value would appear overproportionally bright
        vol = 0.5*(1+np.sin(np.pi*(vol-0.5))) # apply scaled sine to enhance contrast and scale back to [0,1]
        
    # Plot orthogonal slices 
    if(ahandles is None): 
        fhandle = plt.figure()
        ax1 = fhandle.add_subplot(2,2,1)
        ax3 = fhandle.add_subplot(2,2,3)
        ax4 = fhandle.add_subplot(2,2,4)
        ahandles = [ax1, ax3, ax4]
    dims = ['Z','Y','X']
    for d in dims:
        if(d=='Z'): 
            volslice = np.squeeze(vol[:,:,point[2]])
            if(mask is not None):
                maskslice = np.squeeze(mask[:,:,point[2]])
            current_ax = ahandles[0]
            current_ax.set_title(d + '-slice through ' + intersection + ' at ' + d + '=' + str(point[2]))
            current_ax.set_xlabel('X-axis')
            current_ax.set_ylabel('Y-axis')
        elif(d=='Y'): 
            volslice = np.squeeze(vol[point[0],:,:])
            if(mask is not None):
                maskslice = np.squeeze(mask[point[0],:,:])
            current_ax = ahandles[1]
            current_ax.set_title(d + '-slice through ' + intersection + ' at ' + d + '=' + str(point[0]))
            current_ax.set_xlabel('Z-axis')
            current_ax.set_ylabel('X-axis')
        elif(d=='X'): 
            volslice = np.squeeze(vol[:,point[1],:])
            if(mask is not None):
                maskslice = np.squeeze(mask[:,point[1],:])
            current_ax = ahandles[2]
            current_ax.set_title(d + '-slice through ' + intersection + ' at ' + d + '=' + str(point[1]))
            current_ax.set_xlabel('Z-axis')
            current_ax.set_ylabel('Y-axis')
        rgb = np.zeros([volslice.shape[0],volslice.shape[1],3])
        rgb[:,:,1] = volslice
        if(mask is not None):
            rgb[:,:,0] = maskslice * volslice
            if(outline):
                rgb[:,:,2] = 1-maskslice  # --> Make background blue to inspect oversegmentation
        current_ax.imshow(rgb)


#%%
def maskoverlay(img, mask, ahandle=None, outline=False):
    '''     
    maskoverlay(img, mask, ahandle=None)
    
    Takes grayscale img and boolean mask of equal size and plots the overlay. The img is treated as 
    normalized between 0 and 1. If not normalized, the img will be normalized by setting its maximum
    value to 1
    
    Optional inputs:
        * ahandles - An axis handels on which to plot
    '''
    if(ahandle is None): 
        fhandle = plt.figure()
        ahandle = fhandle.gca()
    
    if(np.max(img) > 1):
        img = img / np.max(img)
    
    rgb = np.zeros([img.shape[0],img.shape[1],3])
    rgb[:,:,1] = img
    rgb[:,:,0] = mask * img
    if(outline):
        rgb[:,:,2] = 1-mask  # --> Make background blue to inspect oversegmentation
    ahandle.imshow(rgb)


#%%
def maskcomparison(truemask, predictedmask, ahandle=None):
    '''     
    maskcomparison(truemask, predictedmask, ahandle=None)
    
    Takes two binary 2D arrays (masks) as input and shows an RGB image to visualize the comparison of
    both masks. For this, the first mask is treated as the truth and the second as a prediction. Color
    shadings visualize true positives (green), false positives (red) and false negatives (blue).
    
    Optional inputs:
        * ahandles - An axis handels on which to plot
    '''
    if(ahandle is None): 
        fhandle = plt.figure()
        ahandle = fhandle.gca()
        
    assert truemask.size == len((np.where(truemask==1))[0]) + len((np.where(truemask==0))[0])
    assert predictedmask.size == len((np.where(predictedmask==1))[0]) + len((np.where(predictedmask==0))[0])
    
    rgb = np.zeros([truemask.shape[0],truemask.shape[1],3])
    truepositives = np.zeros([truemask.shape[0],truemask.shape[1]])
    falsepositives = np.zeros([truemask.shape[0],truemask.shape[1]])
    falsenegatives = np.zeros([truemask.shape[0],truemask.shape[1]])
    truepositives[np.where(truemask+predictedmask==2)] = 1
    falsepositives[np.where(predictedmask-truemask==1)] = 1
    falsenegatives[np.where(truemask-predictedmask==1)] = 1
    rgb[:,:,0] = falsepositives
    rgb[:,:,1] = truepositives
    rgb[:,:,2] = falsenegatives
    ahandle.imshow(rgb)


#%% 
def intensity(array, ahandle=None, color='green', cap=None):
    '''
    intensity(array, ahandle=None, color='green', cap=None)
    
    Plots 2D numpy array based on intensity with monochromatic plot. If no signal cap is provided 
    and the maximum value is above 1, the image will be normalized by its maximum value.
    '''
    if(ahandle is None):
        plt.figure()
        ahandle = plt.gca()
    if(cap is not None):
        array = np.clip(array,0,cap) / cap
    elif(np.max(array)>1):
        array = array / np.max(array)
    ahandle.imshow(array,cmap=cmap_intensity(color),vmin=0,vmax=1)


#%% 
def cmap_intensity(color):
    C = np.zeros((256,3))
    if(color=='red'   or color=='yellow'  or color=='magenta' or color=='white'): C[:,0] = np.linspace(0,255,num=256)
    if(color=='green' or color=='yellow'  or color=='cyan'    or color=='white'): C[:,1] = np.linspace(0,255,num=256)
    if(color=='blue'  or color=='magenta' or color=='cyan'    or color=='white'): C[:,2] = np.linspace(0,255,num=256)
    return matplotlib.colors.ListedColormap(C/255.0)


#%%
def print_dict(dictionary,filterlist=None,printit=True):
    ''' 
    textstring = print_dict(dictionary)
    
    This takes a dictonary as an input and returns its keys and values as text,
    such that overly long values are not shown. If a string 'filterlist' is 
    provided, only entries of that list will be shown
    '''
    text = ''
    keys = list(dictionary.keys())
    maxlen = len(max(keys, key=len)) + 1
    for key in keys:
        value = dictionary[key]
        infilter = True if filterlist is None else (key in filterlist)
        if(infilter and type(value)==list):
            if(0<len(value)<=3 and type(value[0])==int):
                text = text + key + ' '*(maxlen-len(key))+ ': '+str(value)+'\n'
            else:
                text = text + key + ' '*(maxlen-len(key))+ ': (list with '+str(len(value))+' elements)\n'
        elif(infilter and 'numpy.ndarray' in str(type(value))):
            if(len(value.shape)==1 and value.shape[0]<5):
                text = text + key + ' '*(maxlen-len(key))+ ': '+str(value)+'\n'
            else:
                text = text + key + ' '*(maxlen-len(key))+ ': (array of size '+str(value.shape)+')\n'
        elif(infilter and type(value)==dict):
            text = text + key + ' '*(maxlen-len(key))+ ':\n'
            deeptext = print_dict(dictionary[key],filterlist=filterlist,printit=False)
            for line in iter(deeptext.splitlines()):
                text = text + ' |-> ' + line + '\n'
        elif(infilter and('float' in str(type(value)) or 'int' in str(type(value)))):
            text = text + key + ' '*(maxlen-len(key))+ ': '  + str(round(value,2)) + '\n' # scalar metrics
        elif(infilter):
            text = text + key + ' '*(maxlen-len(key))+ ': '  + str(value)  + '\n' # vector metrics or text
    if(printit):
        print(text)
    else:
        return text

#%%
def progress_bar(pars,toto,prefixmsg="", postfixmsg=""):
    percent = 10 * (pars / toto) + 1
    bars = int(np.floor(percent)) * "█"
    print(prefixmsg + "\t | Overall: %d%%\t" % percent + bars + postfixmsg, end="\r", flush=True)
