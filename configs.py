import tensorflow as tf

tf.app.flags.DEFINE_integer('batch_size', 200, 'batch size') 
tf.app.flags.DEFINE_integer('Epoch', 10000, 'Epoch') 
tf.app.flags.DEFINE_integer('layer_size', 1, 'layer size') 
tf.app.flags.DEFINE_integer('hidden_size', 128, 'weights size') 
tf.app.flags.DEFINE_integer('shuffle_seek', 1000, 'shuffle random seek') 
tf.app.flags.DEFINE_integer('max_seq_len', 20, 'max sequence length')
tf.app.flags.DEFINE_integer('emb_size', 128, 'embedding size') 
tf.app.flags.DEFINE_float('dropout_width', 0.8, 'dropout width') 
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate') 
tf.app.flags.DEFINE_string('data_path', './data/ChatBotData.csv', 'data path') 
tf.app.flags.DEFINE_string('voca_path', './vocaData.voc', 'voca path') 
tf.app.flags.DEFINE_string('check_point_path', './check_point', 'check point path') 
tf.app.flags.DEFINE_string('f', '', 'kernel') 


# Define FLAGS
DEFINES = tf.app.flags.FLAGS
