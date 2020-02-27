# bottom-up-attention-vqa
To compat higher version of python and Pytorch, I changed some part of the hengyuan-hu's implementation: https://github.com/hengyuan-hu/bottom-up-attention-vqa
. Thanks for him generously sharing the code. 

Most importantly, the results of this code is able to get 63.50 now.
Due to time restrictions, I just trained few times. If you are interested, I believe it's able to get the reported performance in original version.

By the way, I find a problem exist in weight_norm of Pytorch, and there's the link about this in Pytorch forum: https://discuss.pytorch.org/t/is-weight-norm-different-in-different-python-versions/71108. 
If someone is know or interested about this, please don't hesitate to contact me.
Thanks! So before this problem is solved, train with python2.7 env is suggested.

There are some changes in this code:
 - All codes are reformatted and some unnecessary part are trimmed. 
 - The ways to process and load images are edited.
 - To keep succinct, some functions are changed in dataset.
 - This code is compatible with 4 dataset: VQA-v1, VQA-v1, CP-v1, CP-v2.
 - Small fraction is changed in model to compatible with Pytorch v1.0.1 and python3.
 - The ways to train and eval are edited, refer to this code: [https://github.com/guoyang9/vqa-prior](https://github.com/guoyang9/vqa-prior). Also great thanks!
 - Add test support and save_results function.
 
 
 Preprocessing
 - 
1.Extract pre-trained image features.
```
python preprocess-image-rcnn.py
```
2.Preprocess questions and answers.
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