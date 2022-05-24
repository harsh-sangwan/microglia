# microglia

training.py (mini U-Net) is the main file taking input cx3 and iba1 datasets in the shape (200, 200, 200)

  On line 33 in the beginning of the code, one can initialize the path for cx3 and iba1 patch directories and below that initialize the num of epochs and     batch size on line 42.

  On line 87 test set patch sequence can be initiatilized to take as many cx3cr1 and/or iba1 patch slices
  Similarly, on line 99 validation patch sequence can initialized.
  Rest of the patches are used for training.

  On line 165, initialize model loss learning rate, learning rate decay, optimizer and metrics to be used during training. Default loss function is dice     coefficient loss
  
  Later, model is fit with training and validation dataset, performance metrics (dice coef, sensitivitiy, specificitiy and accuracy are computed for each patch til 5-folds and the model with best validation set score is chosen as the final model.
  
  On line 300, threshold for f1-score is determined for sigmoid output and then blobanalysis script is run to get the blob dice metrics for the test set and results are saved in the output/ directory for the segmentation output and overlays.
  
  best weights are saved for the cx3 dataset as "cx3_best.h5"
  
  
training_input_test.py (large U-Net with different size input)

  On line 31, initialize the input path for raw and ground truth data.
  On line 35, initialize the num of epochs and batch size
  On line 63, initialize the test patch sequence
  On line 69, set the splice by parameter, by which size to divide the input patch by (splice by = 2 will divide the 200 cube into 100 cube and so on)
  On line 76, set the validation patch sequence, i.e. how many patches to take in validation set
  On line 114, get_splcied_array function returns the spliced list of arrays input for the patches.
