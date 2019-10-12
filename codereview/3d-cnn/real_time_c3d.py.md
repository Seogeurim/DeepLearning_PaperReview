---
description: '#Real-Time #C3D'
---

# real\_time\_c3d.py

#### 실시간으로 Video Classification을 실행하는 코드이다.

#### 코드는 맨 밑의 main 함수부터 위로 올라가면서 보는 것을 추천한다. 

### Import

```python
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import c3d_model # 여기서 나중에 inference_c3d() 함수 쓸거임 
from real_time_input_data import * # real_time_input_data 함수 다 쓸꺼임!! 
import numpy as np
import cv2
import heapq
```

```python
# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 1 , 'Batch size.')
FLAGS = flags.FLAGS

images_placeholder = tf.placeholder(tf.float32, shape=(1, 16, 112, 112, 3))
labels_placeholder = tf.placeholder(tf.int64, shape=1)
```

```python
def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(1, 16,112,112,3))
  labels_placeholder = tf.placeholder(tf.int64, shape=1)
  return images_placeholder, labels_placeholder
```

```python
def _variable_on_cpu(name, shape, initializer):
    #with tf.device('/cpu:%d' % cpu_id):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var
```

```python
def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('losses', weight_decay)
    return var
```

```python
def run_one_sample(norm_score, sess, video_imgs): # predict 값 도출해내는 함수 
    """
    run_one_sample and get the classification result
    :param norm_score:
    :param sess:
    :param video_imgs:
    :return:
    """
# images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size * gpu_num)
#    start_time = time.time()
#    video_imgs = np.random.rand(1, 16, 112, 112, 3).astype(np.float32)
    predict_score = norm_score.eval(
            session=sess,
            feed_dict={images_placeholder: video_imgs}
            )
    top1_predicted_label = np.argmax(predict_score)
    predict_score = np.reshape(predict_score,101)
    #print(predict_score)
    top5_predicted_value = heapq.nlargest(5, predict_score)
    top5_predicted_label = predict_score.argsort()[-5:][::-1]
    return top1_predicted_label, top5_predicted_label, top5_predicted_value
```

### build\_c3d\_model

pre-train된 model을 돌려서 그 결과 \(norm\_score, sess\)를 리턴하는 함수이다. 

```python
def build_c3d_model():
    #model_name = "pretrained_model/c3d_ucf101_finetune_whole_iter_20000_TF.model.mdlp"
    #model_name = "pretrained_model/conv3d_deepnetA_sport1m_iter_1900000_TF.model"
    model_name = "pretrained_model/sports1m_finetuning_ucf101.model"
    # Get the sets of images and labels for training, validation, and
    with tf.variable_scope('var_name') as var_scope:
        weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
            'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
            'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
        }
        biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
            'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
            'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
            'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
        }
    # logit 이 무엇인지? 자꾸 나오네 
    logits = []
    for gpu_index in range(0, gpu_num):
        with tf.device('/gpu:%d' % gpu_index):
            # c3d_model.py 의 inference_c3d 함수에 weight랑 bias 넣어서 돌렸어. 그 결과가 logits 
            logit = c3d_model.inference_c3d(
                images_placeholder[0 * FLAGS.batch_size:(0 + 1) * FLAGS.batch_size,:,:,:,:], 0.6,
                FLAGS.batch_size, weights, biases)
            logits.append(logit)
    logits = tf.concat(logits, 0)
    norm_score = tf.nn.softmax(logits) # logits softmax 돌려서 score 갖고 오고 
    saver = tf.train.Saver() # 이건 무언지.. 
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) # 이것도 무언지 .. 
    init = tf.global_variables_initializer()
    sess.run(init)
    # Create a saver for writing training checkpoints.
    saver.restore(sess, model_name) # 무언지? 
    return norm_score, sess # score 계산한거랑 session 리턴 
```

* model\_name pretrained model은 _pretrained\_model_ 디렉토리 안의 _sports1m\_finetuning\_ucf101.model_ 이름으로 저장되어 있다. 
* weights / biases

### real\_time\_recognition

실시간으로 비디오를 받아 classification 을 진행하는 함수이다.   
param video\_path : the origin video\_path  
return : x

```python
def real_time_recognition(video_path):
    norm_score, sess = build_c3d_model() # pretrained를 c3d 모델 돌려서 나온 score랑 세션 
    video_src = video_path if video_path else 0 # 비디오 패쓰 받음 
    cap = cv2.VideoCapture(video_src) 
    count = 0
    video_imgs = []
    predicted_label_top5 = []
    top5_predicted_value = []
    predicted_label = 0
    classes = {}
    flag = False
    with open('./list/classInd.txt', 'r') as f: # classlist 열어따 
        for line in f:
            content = line.strip('\r\n').split(' ')
            classes[content[0]] = content[1] # classes라는 배열에 저장 
   # print(classes)
    while True:
        ret, img = cap.read() # cap.read()는 무슨 함수? 
        if type(img) == type(None): # type None이 무엇? 
            break
        float_img = img.astype(np.float32) # 이미지를 float로 바꿨네용 
        video_imgs.append(float_img) # video_imgs 배열에 넣었네용 
        count += 1
        if count == 16: # 16 프레임을 다 받으면 
            video_imgs_tensor = clip_images_to_tensor(video_imgs, 16, 112) 
            # real_time_input_data.py 에 있는 함수 : 이미지 크롭하고 전처리 해서 리턴 
            predicted_label, predicted_label_top5, top5_predicted_value = run_one_sample(norm_score, sess, video_imgs_tensor)
            # top5를 도출 : run_one_sample 
            count = 0 # 다시 초기화 
            video_imgs = [] # 다시 초기화 
            flag = True # 액션 분류 해냈다 ~~~ 근데 flag 다시 false 해주는 코드는 어디..? 
          # channel_1, channel_2, channel_3 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        if flag:
            for i in range(5):
                cv2.putText(img, str(top5_predicted_value[i])+ ' : ' + classes[str(predicted_label_top5[i] + 1)], (10, 15*(i+1)),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                            1, False) # 라벨 보일 수 있도록 putText 

        cv2.imshow('video', img) # 자꾸 무슨 img인지 몰라서 ,,, 

        if cv2.waitKey(33) == 27: # 이건 무엉ㅅ이지?? 
            break

    cv2.destroyAllWindows()
```

### main 함수

```python
def main(_):
    video_path = input("please input the video path to be classification:")
    real_time_recognition(video_path)

if __name__ == '__main__':
    tf.app.run()
```

video\_path를 input으로 받아 실시간으로 classification을 진행하는 방식이다.  
video\_path를 받으면, real\_time\_recognition 함수를 실행시켜 classification을 진행한다.



