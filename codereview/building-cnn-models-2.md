---
description: This is main code for building cnn models.
---

# Building CNN Models 2

#### This is Code Review of [https://github.com/taspinar/sidl](https://github.com/taspinar/sidl) that contains Deep Learning code to understand DL concepts. 

**1\_Building\_CNN\_Models\_in\_Tensorflow.ipynb** 코드를 이어서 리뷰해보도록 하겠다.   


```python
#learning_rate = 0.1
learning_rates = [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001]
display_step = 1000
batch_size = 64

#set the image dimensions
image_width, image_height, image_depth, num_labels = mnist_image_width, mnist_image_height, mnist_image_depth, mnist_num_labels
#image_width, image_height, image_depth, num_labels = c10_image_width, c10_image_height, c10_image_depth, c10_num_labels

#Define some variables in order to avoid the use of magic variables
MODEL_KEY = "adam_lenet5"
USED_DATASET= "CIFAR-10"

#save the accuracy at each step in these dictionaries
dict_train_accuracies = { MODEL_KEY: defaultdict(list) }
dict_test_accuracies = {  MODEL_KEY: defaultdict(list) }


for learning_rate in learning_rates:
    graph = tf.Graph()
    with graph.as_default():
        #1) First we put the input data in a tensorflow friendly form. 
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, image_depth))
        tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
        tf_test_dataset = tf.constant(test_dataset, tf.float32)

        #2) 
        # Choose the 'variables' containing the weights and biases
        # You can choose from:
        # variables_lenet5() | variables_lenet5_like() | variables_alexnet() | variables_vggnet16()
        variables = variables_lenet5(image_width = image_width, image_height=image_height, image_depth = image_depth, num_labels = num_labels)

        #3.
        # Choose the model you will use to calculate the logits (predicted labels)
        # You can choose from:
        # model_lenet5       |  model_lenet5_like      | model_alexnet       | model_vggnet16
        model = model_lenet5
        logits = model(tf_train_dataset, variables)

        #4. 
        # Then we compute the softmax cross entropy between the logits and the (actual) labels
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

        #5. 
        # The optimizer is used to calculate the gradients of the loss function 
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        #optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.0).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(model(tf_test_dataset, variables))


    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized with learning_rate', learning_rate)
        for step in range(num_steps):
            #Since we are using stochastic gradient descent, we are selecting  small batches from the training dataset,
            #and training the convolutional neural network each time with a batch. 
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            train_accuracy = accuracy(predictions, batch_labels)
            dict_train_accuracies[MODEL_KEY][learning_rate].append(train_accuracy)

            if step % display_step == 0:
                test_accuracy = accuracy(test_prediction.eval(), test_labels)
                dict_test_accuracies[MODEL_KEY][learning_rate].append(test_accuracy)
                message = "step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on test set {:02.2f} %".format(step, l, train_accuracy, test_accuracy)
                print(message)
```

#### 



