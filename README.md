# PHYTOMap

[PHYTOMap](Link to preprint) is a multiplexed fluorescence in situ hybridization method that enables single-cell and spatial analysis of gene expression in whole-mount plant tissue in a transgene-free manner and at low cost. 

<!---![alt text]("https://github.com/tnobori/PhytoMap/blob/220707/resources/phytomap_principles.png" width="60%" height="50%")--->
<img src="https://github.com/tnobori/PhytoMap/blob/220707/resources/phytomap_principles.png" width="60%" height="60%">

## PHYTOMap analysis pipeline
![alt text](https://github.com/tnobori/PhytoMap/blob/220707/resources/phytomap_analysis_fig.png)

This pipeline processes images acquired in sequential rounds of FISH in PHYTOMap experiments. 

**Image registration**: This step corrects shifts in a field-of-view during sample handling and imaging by a global affine alignment using random sample consensus (RANSAC)-based feature matching.  
**Spot detection and decoding**: [starfish package](https://github.com/spacetx/starfish) is used to detect single RNA molecule-derived spots in registered images, and the spots are decoded based on imaging rounds and channels they are detected. Deteiled tuorial on starfish is available [here](https://spacetx-starfish.readthedocs.io/en/latest/)  
**Segmentation**: Cells need to be segmented to allow single-cell analysis. This step is not covered here.  
**Cell by gene matrix**: Decoded spots are assigned to segmented cells and counted to obtain a cell by gene matrix for downstream analysis.  
