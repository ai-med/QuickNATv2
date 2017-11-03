# QuickNATv2

Tool: QuickNAT: Segmenting MRI Neuroanatomy in 20 seconds

Authors: Abhijit Guha Roy, Sailesh Conjeti, Nassir Navab and Christian Wachinger

Prerequisits:

1. A nvidia GPU with cuda compatibility
2. Installed Cuda Toolkit 8
3. CuDNN v5.1
4. Installed Matlab R2015a or later

------------------------------------
Components and Steps to run
------------------------------------

1. Extract the zip file 'BrainSegQuickNAT.zip'.

2. MatConvNet version beta 24 in the folder (with extra layers and files required for QuickNAT)

3. Open Matlab and add all the contents of the un-zipped folder to the path.

4. Compile the extracted MatConvNet (Refer: http://www.vlfeat.org/matconvnet/install/)

5. Go To folder '/RunFile/' and open the file RunFile.m

6. Enter the path and name of the MRI volume. (Make sure the Data has iso-tropic resolution of 256 x 256 x 256, if not use freesurfer to do it using the command: mri-convert --conform InputVol OutputVol. It takes less than a second to do this.)

7. Run the Code and Get Segmentations saved in the same folder within 20secs!!!

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
