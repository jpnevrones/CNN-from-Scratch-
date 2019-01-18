=============================================================================================================================
												DeepNet Project Overview
=============================================================================================================================

This file contains a summary of what you will find in each of the files that make up your DeepNet Project.

Summary
DeepNet project is complete Artiffical neural network package created in C++, inorder to solve computer vision problem.
As of now with version 0.5, it can be used create any kind of deep neural structure right from complex deep CNN based 
architecture to simple one layer neural network and so on. 

Different kind of network layer support and there required input are provided below:
Input layer				: 
Convolution layer		:
Fully Connected layer	:
Pooling layer			:
Softmax layer			:
Dropout layer			:


Description		DeepNet: Artificial Neural Network 
				Developed using :	Microsoft .NET Framework version 4.6.01055
									Microsoft Visual Studio Community 2017 
									(VisualStudio/15.0.0-RTW+26228.4)
				Source language :   Microsoft Visual C++ 2017
				Dependency		:   Opencv v3.0 
									How to install opencv on your machine x86/x64/ARM?
									Link : https://developer.microsoft.com/en-us/windows/iot/samples/opencv

				
version 0.4                                            
				Start Date :		03/31/2017
				Developers :		JP - Jithin Pradeep(jxp161430), 
									(Add name as you update the code) 
				Change Deatils :	version 0.1 - Intial Draft: DeepNet: Convolutional Neural Network
												  -- 03/31/2017 -- JP 	
									version 0.2 - Implememtation of Network logic and utility function with support of 
												  opencv on Matrix and scalar dataype
												  -- 04/10/2017 -- JP 
									version 0.3 - Implmentation of various network layer and support function like 
													- Convolution kernel and other convolution related function
													- Max and stochastic pooling with overlap support
												  -- 04/15/2017 -- JP 	
									version 0.4 - Input processing for image based data
				Test Dataset used : MINST, CIFAR10 
			    Performance stats : Still to be tested

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Folder Structure and Description as per visual studio
(any new file added changing the structure must be note along with version number)
Project root
|_____
	|---Header Files						--Header file folders
		|_____
			|----utilities.h				v0.1.2
			|----targetver.h				v0.1.0
			|----stdafx.h					v0.1.3
			|----Network
			|_________
		 			|----network.h			v0.1.5
					|----Layers				--Specific to network layer folder
					|_________
							|----input.h				v0.1.2
							|----convolution.h			v0.1.6
							|----fullyConnected.h		v0.1.2
							|----pooling.h				v0.1.5
							|----softmax.h				v0.1.3
							|----dropout.h				v0.1.2
							|----branch.h				v0.1.1
							|----combine.h				v0.1.1
							|----normalization.h		v0.1.3
							|----activation.h			v0.1.3
	|---Resource Files						--Resource file folder
	|---Source Files						--Source cpp file folder
		|_____
			|----deepNetMain.cpp			v0.1.0
			|----network.cpp			    v0.1.7        -- contains the core modules for network operation
			|----utilities.cpp			    v0.1.2
			|----stdafx.cpp                 v0.1.0  
	below are the opencv3 compiled dll file used within the project, are placed under the root directory
	|---opencv_core300.dll					v3.0
	|---opencv_imgcodecs300.dll				v3.0
	|---opencv_imgproc300.dll				v3.0
	|---opencv_ml300.dll					v3.0
	|---opencv_objdetect300.dll				v3.0


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Tips required to setup the DeepNet to perform further devlopment using Visual studio, followng file are important.

DeepNet.vcxproj
    This is the main project file for VC++ projects generated using an Application Wizard.
    It contains information about the version of Visual C++ that generated the file, and
    information about the platforms, configurations, and project features selected with the
    Application Wizard.

DeepNet.vcxproj.filters (filter are required for recreating back the folder structure in visual studio)
    This is the filters file for VC++ projects generated using an Application Wizard. 
    It contains information about the association between the files in your project 
    and the filters. This association is used in the IDE to show grouping of files with
    similar extensions under a specific node (for e.g. ".cpp" files are associated with the
    "Source Files" filter).

deepNetMain.cpp
    This is the main application source file.

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Other standard files:

StdAfx.h, StdAfx.cpp
    These files are used to build a precompiled header (PCH) file
    named convolutionNN.pch and a precompiled types file named StdAfx.obj.

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

How to install opencv on your machine x86/x64/ARM?
Link : https://developer.microsoft.com/en-us/windows/iot/samples/opencv

