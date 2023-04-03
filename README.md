## BETA-free

This repository contians the source code and datasets for the paper: **BETA-free**.



## Usage

1. **Environment**
   ```shell
   conda create -n your_env_name python=3.8
   conda activate your_env_name
   cd OneRel
   pip install -r requirements.txt
   ```

2. **The pre-trained BERT**

    The pre-trained BERT (bert-base-cased) will be downloaded automatically after running the code. Also, you can manually download the pre-trained BERT model and decompress it under `./pre_trained_bert`.


3. **Train the model (take NYT as an example)**

    Modify the second dim of `batch_triple_matrix` in `data_loader.py` to the number of relations, and run

    ```shell
    python train.py --dataset=NYT --batch_size=8 --rel_num=24 
    ```
    The model weights with best performance on dev dataset will be stored in `checkpoint/NYT/`

4. **Evaluate on the test set (take NYT as an example)**

    Modify the `model_name` (line 48) to the name of the saved model, and run 
    ```shell
    python test.py --dataset=NYT --rel_num=24
    ```

    The extracted results will be save in `result/NYT`.




## **Acknowledgment**

I followed the previous work of [OneRel](https://github.com/ssnvxia/OneRel)

So, I would like to express my sincere thanks to them. 



