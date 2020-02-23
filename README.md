# AI & Cloud Assignment 4

<ul>
  <li> Contributors: Yongyi(Nikki)Zhao, Hewei Yang, Zhigeng Liu</li>
</ul>

## Overview
<p> 
  CNN Tweets sentiment model training
</p >


## Details
#### enviorment you need to change:
1. train directory 
2. validation directory 
3. test directory. 
4. dictionary directory.

#### Our enviroenment is as following: 
  ```sh
   os.environ["SM_CHANNEL_TRAIN"] =  r"C:\Users\hewei\Desktop\ai_hw4\noon_0222\Archive\SM_CHANNEL_TRAIN" 
  ```
  
   ```sh
   os.environ["SM_CHANNEL_VALIDATION"] = r"C:\Users\hewei\Desktop\ai_hw4\noon_0222\Archive\SM_CHANNEL_VALIDATION"
  ```
  ```sh
   os.environ["SM_CHANNEL_EVAL"] = r"C:\Users\hewei\Desktop\ai_hw4\noon_0222\Archive\SM_CHANNEL_EVAL"
  ```
  ```sh
   os.environ["SM_MODEL_DIR"] = r"C:\Users\hewei\Desktop\ai_hw4\noon_0222\Archive\SM_MODEL_DIR"
  ```
  
<span style="color:blue"> Due to git capacity issue, we didn't upload glove dictionary, but you can click the link in Reference Matrial Section to download the dictionary and run locally </span> 


## Requirements
<ul>
  <li> Python (3.6, 3.7) </li>
  <li> TensorFlow v1.13 </li>
</ul>

## Reference Matrial
- [Twitter GloVe Dict](https://twitter-text.s3.amazonaws.com/training/data/glove.twitter.27B.25d.txt) 


## Discussion and Development

<p> Most development discussion is taking place on github in this repo.</p >

## Contributing to Tweets Preprocessing Library
<p>
Any contributions, bug reports, bug fixes, documentation improvements, enhancements to make this project better are warmly welcomed.
</p >
