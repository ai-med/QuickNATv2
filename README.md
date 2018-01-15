# QuickNATv2

Tool: QuickNAT: Segmenting MRI Neuroanatomy in 20 seconds

Authors: Abhijit Guha Roy, Sailesh Conjeti, Nassir Navab and Christian Wachinger

Prerequisite:

1. A nvidia GPU with cuda compatibility
2. Installed Cuda Toolkit 8
3. CuDNN v5.1
4. Installed Matlab R2015a or later

If you use this code please cite:

Guha Roy, A., Conjeti, S., Navab, N., and Wachinger, C. 2018. QuickNAT: Segmenting MRI Neuroanatomy in 20 seconds. arXiv preprint arXiv:1801.04161.
 
 Link to paper: https://arxiv.org/abs/1801.04161 
 
 Enjoy!!! :)

------------------------------------
Components and Steps to run
------------------------------------

1. sudo git-clone https://github.com/abhi4ssj/QuickNATv2/

2. Download MatConvNet (http://www.vlfeat.org/matconvnet/). Paste the extra layers 'DagNN_Layers/..' to 'MatConvNet/matlab/+dagnn/' and 'SimpleNN_Layers/..' to 'MatConvNet/matlab/'

3. Compile the extracted MatConvNet with (enablegpu = true) (Tested on Beta 24) (Refer: http://www.vlfeat.org/matconvnet/install/)

5. Go To folder '/RunFile/' and open the file RunFile.m

6. Enter the path and name of the MRI volume. (Make sure the Data has iso-tropic resolution of 256 x 256 x 256, if not use freesurfer to do it using the command: mri-convert --conform InputVol OutputVol. It takes less than a second to do this.). The code uses the 'MRIread' routine from FreeSurfer for read/write operation. You may use your own customized version. 

7. Run the Code and Get Segmentations saved in the same folder within 20secs!!! (Please note: 20sec is when deployed on Titan X Pascal 12GB GPU in Linux Ubuntu 16.04 OS. In the routine SegmentVol, the 'NumFrames' is passed as 70 to utilize this memory effectively. If you are using other GPU, please modify this to accomodate. 'NumFrames' of 10 works on a 4GB GTX 960M Laptop with segmentation time around 1 minute.)

List of Classes with IDs

0 - 'Background'

1 - 'WM left'

2 - 'GM left'

3 - 'WM right'

4 - 'GM right' 

5 - 'Ventricle left'

6 - 'Cereb WM left'

7 - 'Cereb GM left'

8 - 'Thalamus left'

9 - 'Caudate left'

10 -'Putamen left'

11 -'Pallidum left'

12 -'3rd ventricle'

13 -'4th ventricle'

14 -'Brainstem'

15 -'Hippo left'

16 -'Amygdala left'

17 -'VentralDC left'

18 -'Ventricle right'

19 -'Cereb WM right'

20 -'Cereb GM right'

21 -'Thalamus right'

22 -'Caudate right'

23 -'Putamen right'

24 -'Pallidum right'

25 -'Hippo right'

26 -'Amygdala right'

27 -'VentralDC right
