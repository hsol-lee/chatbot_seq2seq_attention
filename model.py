import tensorflow as tf
import sys

from configs import DEFINES

# Bagdanau Attention class 정의
class BahdanauAttention(tf.keras.Model):
    # 파라미터 세팅
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

    # Attention 계산
  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

# LSTM + dropout 네트워크 설정
def lstm_cell(mode, hiddenSize, index):
    cell = tf.nn.rnn_cell.BasicLSTMCell(hiddenSize, name = "lstm"+str(index), state_is_tuple=False)
    if mode == tf.estimator.ModeKeys.TRAIN:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=DEFINES.dropout_width)
    return cell

# Estimator Model Setting
def model(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT
    
    # 임베딩 초기화
    initializer = tf.contrib.layers.xavier_initializer()
    embedding = tf.get_variable(name = "embedding", # 이름
                                shape=[params['vocabulary_length'], params['embedding_size']], #  모양
                                dtype=tf.float32, # 타입
                                initializer=initializer, # 초기화 값
                                trainable=True) # 학습 유무
   

    # 임베딩된 인코딩 배치 생성
    embedding_encoder = tf.nn.embedding_lookup(params = embedding, ids = features['input'])

    # 인코더용 RNN 모델 생성
    with tf.variable_scope('encoder_scope', reuse=tf.AUTO_REUSE):
       
        encoder_cell_list = [lstm_cell(mode, params['hidden_size'], i) for i in range(params['layer_size'])]
        rnn_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell_list, state_is_tuple=False)
        
        # rnn_cell에 의해 지정된 dynamic_rnn 반복적인 생성
        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(cell=rnn_cell, 
                                                                inputs=embedding_encoder, 
                                                                dtype=tf.float32) 

    # 디코더용 RNN 모델 생성
    with tf.variable_scope('decoder_scope', reuse=tf.AUTO_REUSE):
        
        decoder_cell_list = [make_lstm_cell(mode, params['hidden_size'], i) for i in range(params['layer_size'])]
        rnn_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell_list, state_is_tuple=False)
        
        #예측 결과용 임시 파라미터
        predict_temp = list()
        logits_temp = list()
        output_token = tf.ones(shape=(tf.shape(encoder_outputs)[0],), dtype=tf.int32) * 1
        
        for i in range(DEFINES.max_sequence_length):
            # 학습시 디코더용 임베딩
            if TRAIN:
                if i > 0:
                    input_embedding_decoder = tf.nn.embedding_lookup(embedding, labels[:, i-1])  
                else:
                    input_embedding_decoder = tf.nn.embedding_lookup(embedding, output_token) 
            else: 
                input_embedding_decoder = tf.nn.embedding_lookup(embedding, output_token)

            # attention class 적용
            attentions = BahdanauAttention(params['hidden_size'])
            context_vector, attention_weights = attentions(encoder_states, encoder_outputs)
            input_embedding_decoder = tf.concat([context_vector,input_embedding_decoder], axis=-1)
        
            input_embedding_decoder = tf.keras.layers.Dropout(0.5)(input_embedding_decoder)
            decoder_outputs, decoder_state = rnn_cell(input_embedding_decoder, encoder_states)
            decoder_outputs = tf.keras.layers.Dropout(0.5)(decoder_outputs)
            
            # output에 대한 logit 계산
            output_logits = tf.layers.dense(decoder_outputs, params['vocabulary_length'], activation=None)

            # softmax를 통해 단어에 대한 예측 probability를 구현
            output_probs = tf.nn.softmax(output_logits)
            output_token = tf.argmax(output_probs, axis=-1)

            # 한 스텝에 나온 output_token, output_logits 확장
            predict_temp.append(output_token)
            logits_temp.append(output_logits)

        # predict_temp, logits_temp 매트릭스 조절
        predict = tf.transpose(tf.stack(predict_temp, axis=0), [1, 0])
        logits = tf.transpose(tf.stack(logits_temp, axis=0), [1, 0, 2])

        
    # 예측 딕셔너리 파일 생성
    if PREDICT:
        predictions = {
            'indexs': predict, 
            'logits': logits, 
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    

    # Loss 계산 수식 정의
    labels_ = tf.one_hot(labels, params['vocabulary_length'])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_))
    
    # Accuracy 계산
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predict, name='accOp')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    
    # 평가 모드 시 정확도 출력
    if EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert TRAIN

    
    optimizer = tf.train.AdamOptimizer(learning_rate=DEFINES.learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())  

    # Loss, train_op 값 출력
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
