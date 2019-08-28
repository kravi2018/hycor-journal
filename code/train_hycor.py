import tensorflow as tf 
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True
import numpy as np
import os
import json
os.chdir(os.environ['USERPROFILE'] +'\\Downloads\\hycor-journal-master\\code')
import data_helper
import pre_trained_glove
from learn_metrics import calcMetric
from hycor_model import model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

#load training flags
tf.app.flags.DEFINE_integer('GLOVE', 300, 'pre-trained global vectors size.')
tf.app.flags.DEFINE_integer('test_eval_batch', 500, 'the size of test data to evaluate iteratively in the test phase to prevent memory overload')
tf.app.flags.DEFINE_string('filename', 'sample.xlsx', 'set the filename to train the model.')
tf.app.flags.DEFINE_boolean('rmv_stop_wrds', False, 'remove stop words from dataset.')
tf.app.flags.DEFINE_boolean('trained_emb', True, 'include trained embeddings.')
tf.app.flags.DEFINE_string('Input_Size', 'avg', 'set max or avg value for documents length.')
tf.app.flags.DEFINE_integer('overfit_threshold', 200, 'set the number of iterations for overfit threshold.(early stop)')
tf.app.flags.DEFINE_boolean('preset_dataset', False, 'true if train/dev/test are preset in the dataset.')
tf.app.flags.DEFINE_string('training_config', 'training_config.json', "load the model's parameters.")
tf.app.flags.DEFINE_integer('n_classes', 2, 'the number of classes to train the model.')
FLAGS = tf.app.flags.FLAGS

# load the model's hyper-parameters
params = json.loads(open(FLAGS.training_config).read())

# load base dropout
dyn_dropout, _ = calcMetric.calcDropout(0,0,params['dropout_keep_prob'],0)

print('train file:', (FLAGS.filename[0:len(FLAGS.filename)-5]))

# preprocess the dataset to calculate avg/max document's/sentence's length
_oplen,_seqlen,_sentences,_vocab,vocab_size,_vocab_R = data_helper._run_sentence_document_mode_pre_stage(FLAGS.filename, FLAGS.rmv_stop_wrds,FLAGS.n_classes,FLAGS.Input_Size)

print(FLAGS.Input_Size + ' Sentences/Sequences: ' + str(_oplen) + ' / ' + str(_seqlen))

x_,y_,sentence_size,opinion_size,seqlengths,op_lengths =  data_helper._run_sentence_document_mode(FLAGS.filename,_seqlen,_oplen,FLAGS.rmv_stop_wrds,FLAGS.n_classes,_vocab_R) 

# convert to numpy
x_ = np.array(x_,dtype=np.float32)
y_ = np.eye(int(np.max(y_) + 1))[np.int32(y_)]
seqlengths = np.array(seqlengths,dtype=np.int32)
op_lengths = np.array(op_lengths,dtype=np.int32)

# load trained word embeddings
if FLAGS.trained_emb:
    print('loading pre-trained glove embeddings...')
    embedding_mat = np.float32(pre_trained_glove.getPretrainedWordVextors(_vocab))
    params['embedding_size'] = FLAGS.GLOVE
else :
    embedding_mat=tf.constant(0.0)
    
# monitor test accuracy for every experiment
metric_list=[]

# set cross validation parameters
kfold = KFold(5, True)

# enumerate splits
for train_idx,test_idx in kfold.split(x_):
    
    print('creating train/test datasets...')
    # preset train/dev/test values
    if FLAGS.preset_dataset:
        ids_train,ids_dev,ids_test = data_helper.read_preset_dataset_idxs(FLAGS.filename,FLAGS.n_classes)
        # train dataset
        x_train =x_[:ids_train]
        y_train=y_[:ids_train]
        seqlen_train = seqlengths[:ids_train]
        op_lengths_train= op_lengths[:ids_train]
        # dev dataset
        x_dev = x_[ids_train:ids_train+ids_dev]
        y_dev =y_[ids_train:ids_train+ids_dev]
        seqlen_dev = seqlengths[ids_train:ids_train+ids_dev]
        op_lengths_dev =op_lengths[ids_train:ids_train+ids_dev]
        # test dataset
        x_test = x_[-ids_test:]
        y_test = y_[-ids_test:]
        seqlen_test=seqlengths[-ids_test:]
        op_lengths_test = op_lengths[-ids_test:]
    
    # cross validation    
    elif not FLAGS.preset_dataset:  
        x_train = x_[train_idx]
        y_train = y_[train_idx]
        seqlen_train = seqlengths[train_idx]
        op_lengths_train= op_lengths[train_idx]
        x_test = x_[test_idx]
        y_test = y_[test_idx]
        seqlen_test=seqlengths[test_idx]
        op_lengths_test = op_lengths[test_idx]
        x_train, x_dev,y_train,y_dev,seqlen_train,seqlen_dev,op_lengths_train,op_lengths_dev, =train_test_split(x_train,y_train,seqlen_train,op_lengths_train,test_size=0.1)
    
    print('dataset: ' + str(len(x_))  + ' train/dev/test ' + str(len(x_train)) + '/' +str(len(x_dev)) +'/' + str(len(x_test)))
    
    # load batch size  
    batch_size = params['batch_size']
    
    # calculate the training iterations
    training_iters = int(params['n_epochs']*(1/0.75) * (int(len(x_train))/params['batch_size']))
    
    print()
    print('Model Parameters')
    print('-------------------')
    print('training classes: ' + str(FLAGS.n_classes))
    print('n_hidden: ' + str(params['n_hidden']))
    print('embedding_size: ' + str(params['embedding_size']))
    print('base_dropout: ' + str(params['dropout_keep_prob']))
    print('filter_sizes: ' + str(params['filter_sizes']))
    print('num_feature_maps: ' + str(params['num_feature_maps']))
    print('rnn_out_window: ' + str(params['rnn_out_window']))
    print('n_epochs: ' + str(params['n_epochs']))
    print('batch_size: ' + str(params['batch_size']))
    print('-------------------')
    print()
    print('training iterations: ' + str(training_iters))
    print('training the HyCoR model...')
    
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(config=config)
        with sess.as_default():
            cnn_rnn = model(
                n_steps=x_train.shape[1],
                n_input=x_train.shape[2],
                n_classes = y_train.shape[1],
                n_hidden=params['n_hidden'],
                vocab_size=vocab_size,
                embedding_size=params['embedding_size'],
                filter_sizes=[int(i) for i in params['filter_sizes'].split(',')],
                n_feature_maps = params['num_feature_maps'],
                trained_emb=FLAGS.trained_emb,
                rnn_out_window=params['rnn_out_window'])
            
            # set the optimization algorithm
            optimizer=tf.train.RMSPropOptimizer(learning_rate=params['learning_rate'],epsilon=1e-6).minimize(cnn_rnn.loss)
            
            sess.run(tf.global_variables_initializer())
            
            if FLAGS.trained_emb:
                sess.run(cnn_rnn.embedding_init,feed_dict={cnn_rnn.embedding_placeholder: embedding_mat})
                
            step = 1
            merged = tf.summary.merge_all()
            # monitor train/dev scalar accuracy values 
            train_writer = tf.summary.FileWriter('/tmp/tensorflowlogs' + '/train', graph=tf.get_default_graph())
            dev_writer = tf.summary.FileWriter('/tmp/tensorflowlogs' + '/dev',graph=tf.get_default_graph())
            
            # initiate train/dev accuracies
            acc=0
            _acc=0
            _count=0
            
            # keep training until max iterations
            while step <= training_iters:
                # get train batches
                batch_x, batch_y, batch_lengths,batch_oplengths =  data_helper.next_batch(batch_size, x_train,y_train, seqlen_train, True,op_lengths_train)
                # monitor train accuracy information
                summary,_ = sess.run([merged,optimizer], feed_dict={cnn_rnn.input_x: batch_x, cnn_rnn.input_y: batch_y,cnn_rnn.seqlen: batch_lengths, cnn_rnn.dropout_keep_prob: dyn_dropout, cnn_rnn.oplens:batch_oplengths})
                
                # add to summaries
                train_writer.add_summary(summary, step)
                
                # run optimization operation (backprop)
                sess.run(optimizer, feed_dict={cnn_rnn.input_x: batch_x, cnn_rnn.input_y: batch_y, cnn_rnn.seqlen: batch_lengths,  cnn_rnn.dropout_keep_prob: dyn_dropout, cnn_rnn.oplens:batch_oplengths})
                
                # print train accuracy information in python window
                if step % params['display_step'] == 0:
                    # calculate batch accuracy and print
                    acc = sess.run(cnn_rnn.accuracy, feed_dict={cnn_rnn.input_x: batch_x, cnn_rnn.input_y: batch_y ,cnn_rnn.seqlen: batch_lengths,  cnn_rnn.dropout_keep_prob: 1.0, cnn_rnn.oplens:batch_oplengths})
                    
                    # calculate batch loss
                    loss = sess.run(cnn_rnn.loss, feed_dict={cnn_rnn.input_x: batch_x, cnn_rnn.input_y: batch_y ,cnn_rnn.seqlen: batch_lengths, cnn_rnn.dropout_keep_prob: dyn_dropout, cnn_rnn.oplens:batch_oplengths})
                    
                    print("Iter " + str(step) + ", Minibatch Loss= " + \
                        "{:.6f}".format(loss) + ", Accuracy= " + \
                        "{:.5f}".format(acc) + ", Dropout= " +\
                        "{:.2f}".format(dyn_dropout) +", Overfit:" +\
                        "{:.2f}".format((_count/FLAGS.overfit_threshold) *100)  + "%") 
                
                # monitor dev accuracy information
                if step % 5 == 0:
                    # get dev batch
                    batch_x, batch_y, batch_lengths,batch_oplens  =  data_helper.next_batch((batch_size, int(x_dev.shape[0]))[batch_size > int(x_dev.shape[0])], x_dev, y_dev, seqlen_dev, True,op_lengths_dev)
                    
                    summary, _acc = sess.run([merged,cnn_rnn.accuracy], feed_dict={cnn_rnn.input_x: batch_x, cnn_rnn.input_y: batch_y,cnn_rnn.seqlen: batch_lengths,  cnn_rnn.dropout_keep_prob: 1.0,cnn_rnn.oplens:batch_oplens})
                    
                    # calculate new dropout
                    dyn_dropout, _count = calcMetric.calcDropout(acc,_acc,params['dropout_keep_prob'],_count)
            
                    # monitor overfit early-stop
                    if _count > FLAGS.overfit_threshold: 
                        print('Overfit Identidied')
                        step = training_iters
                    
                    dev_writer.add_summary(summary, step )
                
                step += 1
            
            print("Optimization Finished!")
            # calculate test size to evaluate
            test_len = int(x_test.shape[0])
            # init partial lists metrices for evaluating test dataset in parts
            list_partial_acc = []
            list_partial_cm = []
            partial_eval = False
            
            if test_len > FLAGS.test_eval_batch:
                # mark the index of the data
                eval_idx_start = 0
                eval_idx_end = FLAGS.test_eval_batch - 1
                
                # mark partial evaluation
                partial_eval = True
                
                # partition the test data
                eval_range = (test_len//FLAGS.test_eval_batch) + 1
                for _ in range(eval_range):
                    tmp_test_data = x_test[eval_idx_start:eval_idx_end]
                    tmp_test_label = y_test[eval_idx_start:eval_idx_end]
                    tmp_test_seqs = seqlen_test[eval_idx_start:eval_idx_end]
                    tmp_op_lengths_test =op_lengths_test[eval_idx_start:eval_idx_end]
                    
                    partial_acc = sess.run(cnn_rnn.accuracy, feed_dict={cnn_rnn.input_x: tmp_test_data, cnn_rnn.input_y: tmp_test_label,cnn_rnn.seqlen: tmp_test_seqs, cnn_rnn.dropout_keep_prob: 1.0,cnn_rnn.oplens:tmp_op_lengths_test})
                    
                    # store partial accuracy
                    list_partial_acc.append(partial_acc)
                    
                    print("Partial Testing Accuracy:", partial_acc)
                    
                    tmp_actual = np.array([np.where(r==1)[0][0] for r in tmp_test_label])
                    tmp_predicted = cnn_rnn.logits.eval(feed_dict={cnn_rnn.input_x: tmp_test_data, cnn_rnn.dropout_keep_prob: 1.0,cnn_rnn.seqlen: tmp_test_seqs,cnn_rnn.oplens:tmp_op_lengths_test})
                    
                    cm = tf.confusion_matrix(tmp_actual,tmp_predicted,num_classes=n_classes)
                    
                    # get confusion matrix values / store partial confusion matrix
                    list_partial_cm.append(sess.run(cm))
                    
                    # feed test evaluation with new values
                    eval_idx_start+=FLAGS.test_eval_batch
                    eval_idx_end +=FLAGS.test_eval_batch
                    if eval_idx_end > test_len:
                        eval_idx_end = test_len
            
            # evaluate all test dataset    
            else :
                test_data = x_test[:test_len]
                test_label = y_test[:test_len]
                test_seqs = seqlen_test[:test_len]
                
                # calculate overall accuracy
                overall_acc = sess.run(cnn_rnn.accuracy, feed_dict={cnn_rnn.input_x: test_data, cnn_rnn.input_y: test_label,cnn_rnn.seqlen: test_seqs, cnn_rnn.dropout_keep_prob: 1.0,cnn_rnn.oplens:op_lengths_test})
                
                # get actual labels
                actual = np.array([np.where(r==1)[0][0] for r in test_label])
                predicted = cnn_rnn.logits.eval(feed_dict={cnn_rnn.input_x: test_data, cnn_rnn.dropout_keep_prob: 1.0,cnn_rnn.seqlen: test_seqs,cnn_rnn.oplens:op_lengths_test})
                
                cm = tf.confusion_matrix(actual,predicted,num_classes=FLAGS.n_classes)
                
                # get confusion matrix values
                tf_cm = sess.run(cm)
                #print(tf_cm)
                accuracy = np.sum([tf_cm[i,i] for i in range(tf_cm.shape[1])])/np.sum(tf_cm)
            
            # present the test results    
            if partial_eval:
                 accuracy = np.ma.average(list_partial_acc) 
                 tf_cm = np.ma.sum(list_partial_cm,axis=0)
            
            print("\nOverall Testing Accuracy: ", accuracy)    
            
            print('Confusion Matrix: (H:labels, V:Predictions)')
            print('Precision | Recall | Fscore')
            if(y_train.shape[1]==2):
                print(calcMetric.pre_rec_fs2(tf_cm))
            elif (y_train.shape[1]==3):
                print(calcMetric.pre_rec_fs3(tf_cm))
            elif (y_train.shape[1]==4):
                print(calcMetric.pre_rec_fs4(tf_cm))
            elif (y_train.shape[1]==5):
                print(calcMetric.pre_rec_fs5(tf_cm))
            elif (y_train.shape[1]==6):
                print(calcMetric.pre_rec_fs6(tf_cm))
            
            metric_list.append(accuracy)
    
    print("")
    print("reseting default graph")
    tf.reset_default_graph()
    
# tensorboard --logdir=/tmp/tensorflowlogs  
print('average acurracy: ' + "{:.2f}".format(np.average(metric_list)*100))
print(metric_list)
    
