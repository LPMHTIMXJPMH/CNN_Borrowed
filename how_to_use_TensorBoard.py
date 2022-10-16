# 20220807
# 
# <- Set log directory & Create model and summarywriter at the begining ->

# <- While traing, using summarywriter() to save traing data into log file  ->

# <- Open TensorBoard at the broswer and show ->

# Tensorflow TensorBoard VS Pytorch TensorBoard Run on [Google colab], [Jupyter notebook], [VS code]
# use % sign in front of command line in the jupyter notebook or Google colab
# use command directly in linux terminal or VS Code terminal


# Install TensorBoard:

# $ pip install -U tensorboard # -U, --upgrade Upgrade all packages to the newest available version

# Clear any logs from previous runs (if there is previous runs)
$ rm -rf ./logs/



# https://www.tensorflow.org/tensorboard/get_started#:~:text=TensorBoard%20is%20a%20tool%20for,dimensional%20space%2C%20and%20much%20more.
# < TENSORFLOW > #
import tensorflow as tf
# 1 create log directory that will instore data
import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# 2 build your NetWork model
model = model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3 instore data in the directory pre-defined
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 4 train your model
# Have to load x_train, y_train, x_test, y_test at the begining
model.fit(x=x_train, 
          y=y_train, 
          epochs=5, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])

# 5 Open TensorBoard
$ tensorboard --logdir logs/fit
# use %tensorboard --logdir logs/fit if using jupyter notebook



# < TensorFlow using other Method with TensorBoard > #
# By using GradinetTape() mdthod: Make TensorFlow track gradients 
# So the optimizer will calculate the gradient and give you access to those values. 
# Then you can double them, square them, triple them

# 1 set log directory
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'

summary_train = tf.summary.create_file_writer(train_log_dir)
summary_test = tf.summary.create_file_writer(test_log_dir)

# 2 define loss and gradient optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# 2 Define metrics
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

# 3 load dataset
# Have to load x_train, y_train, x_test, y_test at the begining
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.shuffle(60000).batch(64)
test_dataset = test_dataset.batch(64)

# 4 create model
model = model()


# 5 write data information to log file (to show on the TensorBoard) while training
EPOCHS
for epoch in range(EPOCHS):

  for (x_train, y_train) in train_dataset:

    with tf.GradientTape() as tape:
      predictions = model(x_train, training=True)
      loss = loss_object(y_train, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_accuracy(y_train, predictions)

  with summary_train.as_default():
    tf.summary.scalar('loss', train_loss.result(), step=epoch)
    tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)


  for (x_test, y_test) in test_dataset:

    predictions = model(x_test)
    loss = loss_object(y_test, predictions)

    test_loss(loss)
    test_accuracy(y_test, predictions)

  with summary_test.as_default():
    tf.summary.scalar('loss', test_loss.result(), step=epoch)
    tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)


  # # python string
  # template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  # print (template.format(epoch+1,
  #                        train_loss.result(), 
  #                        train_accuracy.result()*100,
  #                        test_loss.result(), 
  #                        test_accuracy.result()*100))


# 6 oepn tensorboard
$ tensorboard --logdir logs/gradient_tape




# https://pytorch.org/docs/stable/tensorboard.html
# < PYTORCH > #
import torch
import torchvision
# 1 build SummaryWriter to show data on the TensorBoard
from torch.utils.tensorboard import SummaryWriter
torch.utils.tensorboard.writer.SummaryWriter(log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='') # log_dir: Default is runs/CURRENT_DATETIME_HOSTNAME
# 1 build SummaryWriter
write_on_Board = SummaryWriter()

# 2 load dataset and build model
trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True)
images, labels = next(iter(trainloader))
grid = torchvision.utils.make_grid(images)
model = model() # build your model



# 3 add data infomation to SummaryWriter to show
write_on_Board.add_scalar('Title', value, step_index)
# value: example: sqrt of training loss
# step_index: example: index in epochs loop;

# SummaryWriter show: 1 image 2 graph 3 scalar value
# SummaryWriter.add_image('Title', grid, 0)
# SummaryWriter.add_graph(model, images)
# SummaryWriter.add_scalar('Title', value, step_index) # Mostly used one.

write_on_Board.close()

# 4 launch TensorBoard
# Launch tensorboard in the broswer
$ tensorboard --logdir = runs