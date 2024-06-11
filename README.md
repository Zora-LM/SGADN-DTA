# SGADN
This is the code necessary to run experiments on SGADN described in the paper [Structure-Aware Graph Attention Diffusion Network for Protein–Ligand Binding Affinity Prediction](https://ieeexplore.ieee.org/abstract/document/10264137).



# Environments

python==3.10.6  
rdkit==2023.03.1  
dgl==1.0.2+cu102  
pytorch==1.12.1  
openbabel==3.1.0  


##################################

0. download PDBbind-2020 dataset from http://www.pdbbind.org.cn/, and CSAR-HiQ dataset from http://www.csardock.org/.
Notice: You need to use the UCSF Chimera tool to convert the PDB-format files into MOL2-format files for feature extraction at first.
The SMILES of ligands and sequences of proteins in dataset/smiles_sequence.csv are obtained from http://www.pdbbind.org.cn/.

1. run 'prepare_pdbbind.py' to prepare dataset for training

2. run 'run_train.py' to train SGADN and test on the core set.


If you make use of this code or the GraIL algorithm in your work, please cite the following paper:

@article{li2023structure,  
  title={Structure-Aware Graph Attention Diffusion Network for Protein--Ligand Binding Affinity Prediction},  
  author={Li, Mei and Cao, Ye and Liu, Xiaoguang and Ji, Hua},  
  journal={IEEE Transactions on Neural Networks and Learning Systems},  
  year={2023},  
  publisher={IEEE}    
}
