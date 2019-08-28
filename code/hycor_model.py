import tensorflow as tf

class model(object):
    def __init__(self, n_steps, n_input, n_classes, n_hidden, vocab_size, embedding_size, filter_sizes, n_feature_maps, trained_emb, rnn_out_window):
        self.input_x = tf.placeholder(tf.int32, [None, n_steps, n_input], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, n_classes], name='input_y')
        self.seqlen = tf.placeholder(tf.float32 , name='sentences_lengths')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.oplens = tf.placeholder(tf.int32, shape=[None])
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size+1,embedding_size])
        
        # keeping track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)
        self.lambda_term = tf.constant(0.0001)
        self.lambda_term_conv = tf.constant(0.00001)
        
        self.L = 1 # the number of convolution layers
        self.k_top = 5 # the number of top-k max pooling
        self.transpose_perm = [0, 3, 2, 1]
        
        def convlayer(s,w):
            return tf.nn.conv2d(s,w,strides=[1, 1, 1, 1],padding="VALID",name="conv")
        
        def non_linearity(C,bias):
            return tf.nn.relu(tf.nn.bias_add(C,bias) ,name="relu")
            
        def KMaxPooling(layer,l):
            k=max(self.k_top,int(((self.L-l)/self.L)*(int(layer.shape[1]))))
            top_k = tf.nn.top_k(tf.transpose(layer, perm=self.transpose_perm),k=k, sorted=True, name=None)[0]
            return tf.transpose(top_k, perm=self.transpose_perm)
        
        # define weights for seq to seq
        self.weights_f_w=[]
        for i in range(n_steps):
            self.weights_f_w.append(tf.Variable(tf.random_normal([n_hidden, n_classes])))
        
        self.weights_b_w=[]
        for i in range(n_steps):
            self.weights_b_w.append(tf.Variable(tf.random_normal([n_hidden, n_classes])))
        
        self.biases_f_w =[]
        for i in range(n_steps):
            self.biases_f_w.append(tf.Variable(tf.random_normal([n_classes])))
        
        self.biases_b_w =[]
        for i in range(n_steps):
            self.biases_b_w.append(tf.Variable(tf.random_normal([n_classes])))
        
        
        # define embeddings layer
        with tf.device('/cpu:0'), tf.name_scope('embedding_layer'):
            if trained_emb:
                self.W = tf.Variable(tf.constant(0.0, shape=[vocab_size+1, embedding_size]),trainable=True,name='W')
                self.embedding_init = self.W.assign(self.embedding_placeholder)
            else:
                self.W = tf.Variable(tf.random_uniform([vocab_size+1 , embedding_size], -1.0, 1.0),name='W')
            
        def sentence_embedding(conv_x):
            
            self.embedded_chars = tf.nn.embedding_lookup(self.W , conv_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope('conv-maxpool-%s' % filter_size):
                    filter_shape_1 = [filter_size, embedding_size, 1, n_feature_maps]
                    self.W_1 = tf.Variable(tf.truncated_normal(filter_shape_1, stddev=0.1), name='W_1')
                    self.b_1 = tf.Variable(tf.constant(0.1, shape=[n_feature_maps]), name='b_1')
                    # start convolve
                    conv_1 = convlayer(self.embedded_chars_expanded,self.W_1) 
                    
                    # get k-max-pool from first layer
                    kmax_pool = KMaxPooling(conv_1,1)
                    # apply non-linearity
                    c_1 = non_linearity(kmax_pool,self.b_1)
                    
                    pooled_outputs.append(c_1)
            
            num_feature_maps_total = sum([int(fm.shape[1])* int(fm.shape[3]) for fm in pooled_outputs])
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_feature_maps_total])
            out = tf.nn.dropout(h_pool_flat,
            tf.minimum(1.0, tf.add(self.dropout_keep_prob,0.0)))
            return out 
        
        # store the sentence embeddings
        _cnn_stack = []
        sentences = tf.unstack(self.input_x,n_steps,1)
        for i in range(n_steps):
            _cnn_stack.append(sentence_embedding( sentences[i]))
        
        # store the sentences embeddings
        with tf.name_scope('sentence_embeddings'):
            sent_embeddings = tf.tuple(_cnn_stack)
        
        # create bi-direction LSTM network
        def bi_rnn(x):
            # forward direction cell
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1,name='basic_lstm_cell')
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=tf.minimum(1.0, tf.add(self.dropout_keep_prob,0.0)))
            # backward direction cell
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1,name='basic_lstm_cell')
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=tf.minimum(1.0, tf.add(self.dropout_keep_prob,0.0)))
            
            # get bi-lstm cell output
            try:
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, tf.reshape(tf.expand_dims(x, axis = 3),[len(x),-1, int((x[1].shape)[1])]),self.oplens, dtype=tf.float32,time_major=True,scope="bi_lstm")
                outputs= tf.unstack(tf.concat(outputs,2),axis=0)
                
            except Exception: # old version static states
                outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,dtype=tf.float32)
            # provide rnn training information
            tf.summary.histogram('blstm_outputs', outputs)   
            
            return outputs
            
        # create the blstm layer
        with tf.name_scope('blstm_layer'):
            blstm_outputs = bi_rnn(sent_embeddings)
        
        # calculate the output window size
        window = 1 if (int(n_steps*rnn_out_window)==0) else int(n_steps*rnn_out_window)
        
        # create the classical layer
        with tf.name_scope('classical_layer'):
            classical_layer = tf.concat([tf.matmul(tf.slice(blstm_outputs[-1-i], [0, 0], [-1, n_hidden]),self.weights_f_w[-1-i]) + self.biases_f_w[-1-i] for i in range(window)]  + [tf.matmul(tf.slice(blstm_outputs[i], [0, n_hidden], [-1,n_hidden]),self.weights_b_w[i])+ self.biases_b_w[i] for i in range(window)],1)
            
            
        self.W_3 = tf.Variable(tf.truncated_normal([int(classical_layer.shape[1]), n_classes], stddev=0.1), name='W_3')
        self.b_3 = tf.Variable(tf.constant(0.1, shape=[n_classes]), name='b_3')
        
        # calculate the unormalized prediction layer
        with tf.name_scope('prediction'):
            self.pred = tf.add(tf.matmul(classical_layer,self.W_3), self.b_3,name='preds')
            self.logits = tf.argmax(tf.nn.softmax(self.pred),1) 
        
        with tf.name_scope('loss'):
            # define loss 
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pred, labels= self.input_y))
            self.loss = tf.reduce_mean(self.loss + 
            self.lambda_term_conv * tf.nn.l2_loss(self.W_1) + 
            self.lambda_term * tf.nn.l2_loss(self.weights_f_w) +
            self.lambda_term * tf.nn.l2_loss(self.weights_b_w) +
            self.lambda_term * tf.nn.l2_loss(self.W_3))
        
        with tf.name_scope('accuracy'):
            # evaluate model
            self.correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name='num_correct')
        
        # provide accuracy information
        tf.summary.scalar('accuracy', self.accuracy)    
        
           
        
        
        
        
        
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
            
            
            
            
            