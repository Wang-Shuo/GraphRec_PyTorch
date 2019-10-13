# GraphRec_PyTorch
A PyTorch implementation of the GraphRec model in [Graph Neural Networks for Social Recommendation](https://arxiv.org/pdf/1902.07243.pdf) (Fan, Wenqi, et al. "Graph Neural Networks for Social Recommendation." The World Wide Web Conference. ACM, 2019.).

# Usage

1. Install required packages from requirements.txt file.
```bash
pip install -r requirements.txt
```

2. Preprocess dataset. Two pkl files named dataset and list should be generated in the respective folders of the dataset.
```bash
python preprocess.py --dataset Ciao
python preprocess.py --dataset Epinions
``` 

3. Run main.py file to train the model. You can configure some training parameters through the command line. 
```bash
python main.py
```