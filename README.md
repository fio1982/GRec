# GRec
GNN based third-party library recommendation framework for mobile app development


## Usage ###

Each pair of randomly generated training set and testing set is stored in sub 
folder of "Data". Simply start Python in COMMAND LINE mode, then use the following
statement (one line in the COMMAND Prompt window) to execute GRec on one dataset:

```
python main.py --dataset qiaoji_5_30 --alg_type ngcf --regs [1e-5] --lr 0.0001 
--save_flag 1 --pretrain 0 --batch_size 4096 --epoch 800 --verbose 1
```

Then, you may receive the results as follows:

```
Best Iter=[79]@[7756.1]	recall=[0.59502	0.73338], precision=[0.58724	0.36238],
map=[0.84104	0.78629], cov=[0.65841	0.75393], f1=[0.59070	0.48482]
Benchmarking time consuming: average 8.1111s per epoch
```

For the subfolder name: 

```
"***_5_30" means for each mobile app, 5 randomly selected TPLs were removed to form the testing set and the remaining TPLs formed the training set. 
Besides, this pair of dataset is used for the 30th experiment run.
```

For the parameters:

```
dataset: specifies the corresponding testing set and training set. 
epoch: maximum epochs during the training process. GRec may stop early if it cannot further improve the performance in 5 consecutive epochs.
```


## Environment Settup ###

Our code has been tested under Python 3.6.9. The experiment was conducted via 
PyTorch, and thus the following packages are required:

	pytorch == 1.3.1
	numpy == 1.18.1
	scipy == 1.3.2
	sklearn == 0.21.3

Updated version of each package is acceptable. A GPU accelerator NVIDIA Tesla P100
12GB is needed when running the experiments. 


## Description of Folders and Files 

Name |Type |	Description
---|---|---
main.py		|	File	|	main python file of GRec
Models.py	|	File	|	NGCF modules used by GRec
utility		|	Folder	|	tools and essential libraries used by GRec
Training Dataset		|	Folder	|	Training set and testing set, one pair in each sub folder. Only part of the used dataset is provided due to the size limitation of the uploaded file.
Output		|	File	|	A sample of the TPL recommendation results. By comparing the TPLs in the testing set with the recommended ones, the effectiveness of GRec can be evaluated. Please note, only part of the results are provided due to the size limitation.
Original Dataset |	Folder	|	The original public MALib dataset proposed in [9] (TSE,2020)



## Essential Parameter Setup for GRec 

All essential parameters of GRec can be set via file "utility/parser.py". Specifically:

Line|Description
---|---
8|	Setup dataset path, only the parent folder "Data" is needed, GRec can find out each sub folder through parameter "--dataset" in Python commands.
20	|	Setup the default maximum epoch of each experiment run, it will be  overwritten if the corresponding parameter is set through command line.
25	|	Setup the total number of GNN layers and the size of each layer. For example, a value "[128,128,128,128]" means GRec has 4 GNN layers in total and each layer has the same layer size of 128 nodes, i.e., the size of each latent factor vector is 128.
46 	|	Setup how much information will be randomly discarded during the training, the number of elements is equal to the size of parameter array in line 25, and 0.1 means 10% of information will be discarded.
49	|	Specify the size of each recommendation list. An array "[5,10]" means the first TPL recommendation list has 5 TPLs and the second recommendation list has 10 TPLs. This way, we can evaluate GRec's performance with different \textit{nr} in one batch.
others	|	Other paramenter can also be found in this file. Please refer to the comments of each parameter in this file. 
				
## Citation
It would be very much appreciated if you cite the following papers:

>Li, B., He, Q., Chen, F., Xia, X., Li, L., Grundy, J.,and Yang, Y., 2021. Embedding App-Library Graph for Neural Third Party Library Recommendation. In proceddings of FSE 2021. DOI:10.1145/3468264.3468552
