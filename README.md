# bottom-up-attention-vqa
To compat higher version of python and Pytorch, I changed some part of the hengyuan-hu's implementation: https://github.com/hengyuan-hu/bottom-up-attention-vqa
. Thanks for him generously sharing the code. 

Most importantly, the results of this code is able to get 63.60 now. The accuracy is calculated using the [VQA evaluation metric](http://www.visualqa.org/evaluation.html).
Due to time restrictions, I just trained few times. If you are interested, I believe it's able to get the higher performance.

>By the way, I find a problem exist in weight_norm of Pytorch v1.0, which leads to the performance decline in Python3. And there's the link about this problem in Pytorch forum: https://discuss.pytorch.org/t/is-weight-norm-different-in-different-python-versions/71108. 
If someone know or interested about this, please don't hesitate to contact me.
>
>Thanks for ptrblck's answer, this issue is solved in higher version of Pytorch, like v1.4. 
>So weight_norm is safe to be used in this model. And now, whatever you train with Python 3.7 or 2.7, 
>you can get the same results.
>

There are some changes in this code:
 - All codes are reformatted and some unnecessary part are trimmed. 
 - The ways to process and load images are edited.
 - To keep succinct, some functions are changed in dataset.
 - This code is compatible with 4 dataset: VQA-v1, VQA-v2, CP-v1, CP-v2.
 - Small fraction is changed in model to compatible with Pytorch v1.0.0 (or higher version, e.g., 1.8.1) and python3.
 - The ways to train and eval are edited, refer to this code: [https://github.com/guoyang9/vqa-prior](https://github.com/guoyang9/vqa-prior). Also great thanks!
 - Add test support and save_results function.
 
  Prerequisites 
 - 
 - tqdm, h5py
 - python 3.7
 - pytorch (v1.4, or higher version)
 
 
 Preprocessing 
-
 (Please edit the config.py to make sure the directory is correct)
1. Extract pre-trained image features.
    ```
    python preprocess-image-rcnn.py
    ```
2. Preprocess questions and answers.
    ```
    bash tools/process.sh
    ```

 Train
 - 
```
python main.py --output baseline --gpu 0
```

 Model Test only
 - 
 ```
python main.py --test --output baseline --gpu 0
```
