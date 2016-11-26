import tensorflow as tf
import numpy as np
import os
import scipy.io
import glob
import random
import BatchDatsetReader as dataset
import scipy.misc as misc
from six.moves import cPickle as pickle

# reset the graph
tf.reset_default_graph()

# reset tf.flags.FLAGS
import argparse
tf.reset_default_graph()
tf.flags.FLAGS = tf.python.platform.flags._FlagValues()
tf.flags._global_parser = argparse.ArgumentParser()

# set tf.flags.FLAGS
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size","2","batch size for training")
tf.flags.DEFINE_string("logs_dir","logs/","path to logs directory")
tf.flags.DEFINE_string("data_dir","Data_zoo/MIT_SceneParsing/","path to dataset")
tf.flags.DEFINE_string("pickle_name","MITSceneParsing.pickle","pickle file of the data")
tf.flags.DEFINE_string("data_url","http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip","url of the data")
tf.flags.DEFINE_float("learning_rate","1e-4","learning rate for the optimizier")
tf.flags.DEFINE_string("model_dir","Model_zoo/","path to vgg model mat")
tf.flags.DEFINE_bool("debug","True","Debug model: True/False")
tf.flags.DEFINE_string("mode","train","Mode: train/ valid")
tf.flags.DEFINE_integer("max_iters","100001","max training iterations of batches")
tf.flags.DEFINE_integer("num_classes","151","mit_sceneparsing with (150+1) classes")
tf.flags.DEFINE_integer("image_size","224","can be variable in deed")
tf.flags.DEFINE_string("model_weights","http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat","pretrained weights of the CNN in use")
tf.flags.DEFINE_string("full_model","full_model/","trained parameters of the whole network")
tf.flags.DEFINE_string("full_model_file"," ","pretrained parameters of the whole network")
tf.flags.DEFINE_bool("load","False","load in pretrained parameters")

# check if the CNN weights folder exist
if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)

# check if the CNN weights file exist
weights_file = os.path.join(FLAGS.model_dir,FLAGS.model_weights.split('/')[-1])
if not os.path.exists(weights_file):
    print("\ndownloading "+weights_file+" ...")
    os.system("wget "+FLAGS.model_weights+" -P "+FLAGS.model_dir)
    print("download finished!\n")
else:
    print("\n"+weights_file+" has already been downloaded.\n")

# load the weights file
print("\nloading pretrained weights from: "+weights_file)
pretrain_weights = scipy.io.loadmat(weights_file)
print("loading finished!\n")

# the mean RGB
mean = pretrain_weights['normalization'][0][0][0] # shape(224,224,3)
mean_pixel = np.mean(mean,axis=(0,1)) # average on (height,width) to compute the mean RGB   

# the weights and biases
weights_biases = np.squeeze(pretrain_weights['layers'])

# network input data
dropout_prob = tf.placeholder(tf.float32,name="dropout_probability")
images = tf.placeholder(tf.float32,shape=[None,FLAGS.image_size,FLAGS.image_size,3],name="input_images")
annotations = tf.placeholder(tf.uint8,shape=[None,FLAGS.image_size,FLAGS.image_size,1],name="input_annotations")

# subtract the mean image
processed_image = images - mean_pixel

# construct the semantic_seg network
with tf.variable_scope("semantic_seg"):
    # convs of the vgg net
    net = {}
    layers = [
        'conv1_1','relu1_1','conv1_2','relu1_2','pool1',
        'conv2_1','relu2_1','conv2_2','relu2_2','pool2',
        'conv3_1','relu3_1','conv3_2','relu3_2','conv3_3','relu3_3','conv3_4','relu3_4','pool3',
        'conv4_1','relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','conv4_4','relu4_4','pool4',
        'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3' #,'relu5_3','conv5_4','relu5_4','pool5'
    ]
    current = processed_image

    # sanity check
    print("processed_image: {}".format(processed_image.get_shape()))

    for i,name in enumerate(layers):
        type = name[:4]
        if type == 'conv':
            # matconvnet weights: (width, height, in_channels, out_channels)
            # tensorflow weights: (height, width, in_channels, out_channels)
            weights, biases = weights_biases[i][0][0][0][0]

            weights = np.transpose(weights,(1,0,2,3)) 
            biases = np.squeeze(biases)
            
            init = tf.constant_initializer(weights,dtype=tf.float32)
            weights = tf.get_variable(initializer=init,shape=weights.shape,name=name+"_w")
            
            init = tf.constant_initializer(biases,dtype=tf.float32)
            biases = tf.get_variable(initializer=init,shape=biases.shape,name=name+"_b")
            
            current = tf.nn.conv2d(current,weights,strides=[1,1,1,1],padding="SAME")
            current = tf.nn.bias_add(current,biases,name=name)

            # sanity check
            print("{}: {}".format(name,current.get_shape()))
        elif type == 'relu':
            current = tf.nn.relu(current,name=name)
            if FLAGS.debug:
                tf.histogram_summary(current.op.name+"/activation",current)
                tf.scalar_summary(current.op.name+"/sparsity",tf.nn.zero_fraction(current))
            # sanity check
            print("{}: {}".format(name,current.get_shape()))        
        elif type == 'pool':
            if name == 'pool5':
                current = tf.nn.max_pool(current,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME",name=name)
            else:
                current = tf.nn.avg_pool(current,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME",name=name)
            # sanity check
            print("{}: {}".format(name,current.get_shape()))
        net[name] = current
             
    net['pool5'] = tf.nn.max_pool(net['conv5_3'],ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME",name=name)
    # sanity check
    print("pool5: {}".format(net['pool5'].get_shape()))
    
     # fcn6
    init = tf.truncated_normal(shape=[7,7,512,4096],stddev=0.02)
    fcn6_w = tf.get_variable(initializer=init,name="fcn6_w")

    init = tf.constant(0.0,shape=[4096])
    fcn6_b = tf.get_variable(initializer=init,name="fcn6_b")

    fcn6 = tf.nn.conv2d(current,fcn6_w,strides=[1,1,1,1],padding="SAME")
    fcn6 = tf.nn.bias_add(fcn6,fcn6_b,name="fcn6")

    relu6 = tf.nn.relu(fcn6,name="relu6")
    if FLAGS.debug:
        tf.histogram_summary("relu6/activation", relu6, collections=None, name=None)
        tf.scalar_summary("relu6/sparsity", tf.nn.zero_fraction(relu6), collections=None, name=None)
    dropout6 = tf.nn.dropout(relu6, keep_prob=dropout_prob, noise_shape=None, seed=None, name="dropout6")
    # sanity check
    print("dropout6: {}".format(dropout6.get_shape()))

     # fcn7
    init = tf.truncated_normal(shape=[1,1,4096,4096],stddev=0.02)
    fcn7_w = tf.get_variable(initializer=init,name="fcn7_w")

    init = tf.constant(0.0,shape=[4096])
    fcn7_b = tf.get_variable(initializer=init,name="fcn7_b")

    fcn7 = tf.nn.conv2d(dropout6, fcn7_w, strides=[1,1,1,1], padding="SAME", use_cudnn_on_gpu=None, data_format=None, name=None)
    fcn7 = tf.nn.bias_add(fcn7, fcn7_b, data_format=None, name="fcn7")

    relu7 = tf.nn.relu(fcn7,name="relu7")
    if FLAGS.debug:
        tf.histogram_summary("relu7/activation", relu7, collections=None, name=None)
        tf.scalar_summary("relu7/sparsity", tf.nn.zero_fraction(relu7), collections=None, name=None)
    dropout7 = tf.nn.dropout(relu7, keep_prob=dropout_prob, noise_shape=None, seed=None, name="dropout7")
    # sanity check
    print("dropout7: {}".format(dropout7.get_shape()))

    # fcn8
    init = tf.truncated_normal(shape=[1,1,4096,FLAGS.num_classes],stddev=0.02)
    fcn8_w = tf.get_variable(initializer=init,name="fcn8_w")

    init = tf.constant(0.0,shape=[FLAGS.num_classes])
    fcn8_b = tf.get_variable(initializer=init,name="fcn8_b")

    fcn8 = tf.nn.conv2d(dropout7, fcn8_w, strides=[1,1,1,1], padding="SAME", use_cudnn_on_gpu=None, data_format=None, name=None)
    fcn8 = tf.nn.bias_add(fcn8, fcn8_b, data_format=None, name="fcn8")
    # sanity check
    print("fcn8: {}".format(fcn8.get_shape()))

    # deconv1 + net['pool4']: x32 -> x16
    s = 2
    k = 2*s
    in_channel = FLAGS.num_classes
    out_channel = net['pool3'].get_shape()[3].value
    out_shape = tf.shape(net['pool3'])

    init = tf.truncated_normal(shape=[k,k,out_channel,in_channel],stddev=0.02)
    deconv1_w = tf.get_variable(initializer=init,name="deconv1_w")

    init = tf.constant(0.0,shape=[out_channel])
    deconv1_b = tf.get_variable(initializer=init,name="deconv1_b")

    # sanity check
    print("deconv1 output_shape: {}".format(net['pool3'].get_shape()))

    deconv1 = tf.nn.conv2d_transpose(fcn8, deconv1_w, output_shape=out_shape, strides=[1,s,s,1], padding='SAME', name=None)
    deconv1 = tf.nn.bias_add(deconv1, deconv1_b, data_format=None, name="deconv1")

    fuse1 = tf.add(deconv1, net['pool4'], name="fuse1")
            
    # deconv2 + net['pool3']: x16 -> x8
    s = 2
    k = 2*s
    in_channel = out_channel
    out_channel = net['pool3'].get_shape()[3].value
    out_shape = tf.shape(net['pool3'])

    init = tf.truncated_normal(shape=[k,k,out_channel,in_channel],stddev=0.02)
    deconv2_w = tf.get_variable(initializer=init,name="deconv2_w")

    init = tf.constant(0.0,shape=[out_channel])
    deconv2_b = tf.get_variable(initializer=init,name="deconv2_b")

    deconv2 = tf.nn.conv2d_transpose(fuse1, deconv2_w, output_shape=out_shape, strides=[1,s,s,1], padding='SAME', name=None)
    deconv2 = tf.nn.bias_add(deconv2, deconv2_b, data_format=None, name="deconv2")

    fuse2 = tf.add(deconv2,net['pool3'],name="fuse2")

    # deconv3: x8 -> image_size
    s = 8
    k = 2*s
    in_channel = out_channel
    out_channel = FLAGS.num_classes
    out_shape = tf.pack([tf.shape(processed_image)[0],tf.shape(processed_image)[1],tf.shape(processed_image)[2],out_channel])
            
    init = tf.truncated_normal(shape=[k,k,out_channel,in_channel],stddev=0.02)
    deconv3_w = tf.get_variable(initializer=init,name="deconv3_w")

    init = tf.constant(0.0,shape=[out_channel])
    deconv3_b = tf.get_variable(initializer=init,name="deconv3_b")

    deconv3 = tf.nn.conv2d_transpose(fuse2, deconv3_w, output_shape=out_shape, strides=[1,s,s,1], padding='SAME', name=None)
    deconv3 = tf.nn.bias_add(deconv3, deconv3_b, data_format=None, name="deconv3")

    # per-pixel prediction
    annotations_pred = tf.argmax(deconv3, dimension=3, name=None)
    annotations_pred = tf.expand_dims(annotations_pred, dim=3, name="prediction")

# log images, annotations, annotations_pred
tf.image_summary("images", images, max_images=2, collections=None, name=None)
tf.image_summary("annotations", tf.cast(annotations,tf.uint8), max_images=2, collections=None, name=None)
tf.image_summary("annotations_pred", tf.cast(annotations_pred,tf.uint8), max_images=2, collections=None, name=None)

# construct the loss
loss = tf.nn.softmax_cross_entropy_with_logits(deconv3, annotations, dim=3, name=None)
loss = tf.reduce_mean(loss, reduction_indices=None, keep_dims=False, name="pixel-wise_cross-entropy_loss")

# log the loss
tf.scalar_summary("pixel-wise_cross-entropy_loss", loss, collections=None, name=None)

# log all the trainable variables
trainabel_vars = tf.trainable_variables()
if FLAGS.debug:
    for var in trainabel_vars:
        tf.histogram_summary(var.op.name+"/values", var, collections=None, name=None)
        tf.add_to_collection("sum(t ** 2) / 2 of all trainable_vars", tf.nn.l2_loss(var))
        
# construct the optimizier
optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
gradients = optimizer.compute_gradients(loss,trainabel_vars)
if FLAGS.debug:
    # log the gradients
    for grad, var in gradients:
        tf.histogram_summary(var.op.name+"/gradients", grad, collections=None, name=None)
train_op = optimizer.apply_gradients(gradients)

# initialize the variables
print("\nInitializing the variables ...\n")
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

# set up the saver
print("\nSetting up the Saver ...\n")
saver = tf.train.Saver()
if FLAGS.load:
    print("\nLoading pretrain parameters of the whole network ...\n")
    saver.restore(sess, FLAGS.full_model_file)

# set the summary writer
print("\nSetting the summary writers ...\n")
summary_op = tf.merge_all_summaries()
if not os.path.exists(FLAGS.logs_dir):
    os.system("mkdir "+FLAGS.logs_dir)
if FLAGS.mode == 'train':
    if os.path.exists(FLAGS.logs_dir+"/train"):
        os.system("rm -r "+FLAGS.logs_dir+"/train")
    if os.path.exists(FLAGS.logs_dir+"/valid"):
        os.system("rm -r "+FLAGS.logs_dir+"/valid")
    train_writer = tf.train.SummaryWriter(FLAGS.logs_dir+"/train",sess.graph)
    valid_writer = tf.train.SummaryWriter(FLAGS.logs_dir+"/valid")
elif FLAGS.mode == 'valid':
    if os.path.exists(FLAGS.logs_dir+"/complete_valid"):
        os.system("rm -r "+FLAGS.logs_dir+"/complete_valid")
    valid_writer = tf.train.SummaryWriter(FLAGS.logs_dir+"/complete_valid")    

# read data_records from *.pickle
print("\nReading in and reprocessing all images ...\n")
# check if FLAGS.data_dir folder exist
if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)
# check if the *.pickle file exist
pickle_file = os.path.join(FLAGS.data_dir,FLAGS.pickle_name)
if not os.path.exists(pickle_file):
    # check if the *.zip exist
    zip_file = os.path.join(FLAGS.data_dir,FLAGS.data_url.split('/')[-1])
    if not os.path.exists(zip_file):
        # download the *.zip
        print("downloading "+zip_file+" ..")
        os.system("wget "+FLAGS.data_url+" -P "+FLAGS.data_dir)
        print("download finished!")
        # unzip the file
        print("unzipping "+zip_file+" ..")
        os.system("unzip "+zip_file+" -d "+FLAGS.data_dir)
        print("unzipping finished!")
    # pack data into *.pickle
    source_datadir =  zip_file.split('.')[0]
    if not os.path.exists(source_datadir):
        print("Error: source_datadir not found!!!")
        exit()
    else:
        data_types = ['training','validation']
        data_list = {}
        for data_type in data_types:
            image_list = []
            data_list[data_type] = []
            # find all images
            image_names = os.path.join(source_datadir,"images",data_type,'*.jpg')
            print("\nimage_names: %s\n"%(image_names))
            image_list.extend(glob.glob(image_names))
            if not image_list:
                print("Error: no images found for "+data_type+"!!!")
                exit()
            else:
                # find corresponding annotations
                for i in image_list:
                    image_name = (i.split('/')[-1]).split('.')[0]
                    annotation_name = os.path.join(source_datadir,"annotations",data_type,image_name+".png")
                    if os.path.exists(annotation_name):
                        # record this data tuple
                        record = {'image':i,'annotation':annotation_name,'filename':image_name}
                        data_list[data_type].append(record)
            # shuffle all tuples
            random.shuffle(data_list[data_type])
            print("Number of %s tuples: %d"%(data_type,len(data_list[data_type])))
    print("Packing data into "+pickle_file+" ...")
    with open(pickle_file,'wb') as f:
        pickle.dump(data_list,f,pickle.HIGHEST_PROTOCOL)
    print("pickle finished!!!")
# load data_records from *.pickle
with open(pickle_file,'rb') as f:
    pickle_records = pickle.load(f)
    train_records = pickle_records['training']
    valid_records = pickle_records['validation']
    del pickle_records
    
# initialize the data reader
print("Initializing the data reader...")
reader_options = {'resize':True,'resize_size':FLAGS.image_size}
if FLAGS.mode == 'train':
    train_reader = dataset.BatchDatset(train_records[:10],reader_options)
valid_reader = dataset.BatchDatset(valid_records[:10],reader_options)

# check if FLAGS.full_model exist
if not os.path.exists(FLAGS.full_model):
    os.makedirs(FLAGS.full_model)

# start training/ validation
print("\nStarting training/ validation...\n")
if FLAGS.mode == 'train':
    for itr in xrange(FLAGS.max_iters):
        # read next batch
        train_images, train_annotations = train_reader.next_batch(FLAGS.batch_size)
        feed_dict = {images:train_images,annotations:train_annotations,dropout_prob:0.85}
        # training
        sess.run(train_op,feed_dict=feed_dict)
        # log training info
        if itr % 10 == 0:
            train_loss, train_summary = sess.run([loss,summary_op],feed_dict=feed_dict)
            train_writer.add_summary(train_summary,itr)
            print("Step: %d, train_loss: %f"%(itr,train_loss))
        # log valid info
        if itr % 100 == 0:
            valid_images, valid_annotations = valid_reader.get_random_batch(FLAGS.batch_size)
            feed_dict = {images:valid_images,annotations:valid_annotations,dropout_prob:1.0}
            valid_loss, valid_summary = sess.run([loss,summary_op],feed_dict=feed_dict)
            valid_writer.add_summary(valid_summary,itr)
            print("==============================")
            print("Step: %d, valid_loss: %f"%(itr,valid_loss))
            print("==============================")
        # save snapshot
        if itr % 500 == 0:
            snapshot_name = os.path.join(FLAGS.full_model,str(itr)+"_model.ckpt")
            saver.save(sess,snapshot_name)
elif FLAGS.mode == 'valid':
    # quantitative results
    valid_images,valid_annotations=valid_reader.get_records()
    feed_dict = {images:valid_images,annotations:valid_annotations,dropout_prob:1.0}
    valid_loss,valid_summary = sess.run([loss,summary_op],feed_dict=feed_dict)
    valid_writer.add_summary(valid_summary,FLAGS.max_iters)
    print("==============================")
    print("Step: %d, valid_loss: %f"%(FLAGS.max_iters,valid_loss))
    print("==============================")
    # qualitative results
    valid_images,valid_annotations=valid_reader.get_random_batch(FLAGS.batch_size)
    feed_dict = {images:valid_images,annotations:valid_annotations,dropout_prob:1.0}
    annotations_pred_results = sess.run(annotations_pred,feed_dict=feed_dict)
    
    valid_annotations = np.squeeze(valid_annotations,axis=3)
    annotations_pred_results = np.squeeze(annotations_pred_results,axis=3)
    
    for n in xrange(FLAGS.batch_size):
        print("Saving %d valid tuples for qualitative comparisons...")
        misc.imsave(FLAGS.logs_dir+"/complete_valid/"+str(n)+"_image.png",valid_images[n].astype(np.uint8))
        misc.imsave(FLAGS.logs_dir+"/complete_valid/"+str(n)+"_annotation.png",valid_annotations[n].astype(np.uint8))
        misc.imsave(FLAGS.logs_dir+"/complete_valid/"+str(n)+"_prediction.png",annotations_pred_results[n].astype(np.uint8))
        print("saving finished!!!")