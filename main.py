from seq2seq_attention_model import seq2seqModel
from datasets import DataSet, DataConvert
import numpy as np
import tensorflow as tf


def train():
    buckets = [(30, 20)]
    content_word_lengths = [30]
    batch_size = 1
    dataset = DataSet("data/contents.txt", "data/titles.txt", batch_size, False, buckets, content_word_lengths, content_max_length=20)
    vocab_size = dataset.vocab_size
    size = 128
    num_layers = 1
    num_softmax_samples = int(dataset.vocab_size / 3)  # python3需要转换成int
    do_decode = False
    model = seq2seqModel(vocab_size, buckets, size, num_layers, batch_size, num_softmax_samples, do_decode)
    model.fit(dataset)

def predict():
    buckets = [(30, 20)]
    content_word_lengths = [30]
    batch_size = 1
    dataset = DataSet("data/contents.txt", "data/titles.txt", batch_size, True, buckets, content_word_lengths, content_max_length=20)  # 必须是True啊 防止出错
    vocab_size = dataset.vocab_size
    size = 128
    num_layers = 1
    num_softmax_samples = int(dataset.vocab_size / 3)  # python3需要转换成int
    do_decode = True  # 解码过程
    convert = DataConvert(buckets, content_word_lengths, content_max_length=30, title_max_length=20)
    bucket_id, encoder_length, encoder_inputs_ids, decoder_inputs_ids = convert.convert("我爱你中国")
    checkpointfile = "C:\\Users\\John.zhang\\PycharmProjects\\mycode\\model\\model-999"
    encoder_inputs_ids = np.array(encoder_inputs_ids)
    decoder_inputs_ids = np.array(decoder_inputs_ids)
    encoder_inputs_ids = np.transpose(encoder_inputs_ids)
    decoder_inputs_ids = np.transpose(decoder_inputs_ids)
    with tf.Session() as sess:
        model = seq2seqModel(vocab_size, buckets, size, num_layers, batch_size, num_softmax_samples, do_decode, initial_state_attention=True)
        saver = tf.train.Saver()
        encoder_inputs_bucket_dict = dict(zip(model.encoder_inputs_buckets[bucket_id], encoder_inputs_ids))
        # decoder_inputs_bucket_dict = dict(zip(model.decoder_inputs_buckets[bucket_id], decoder_inputs_ids))
        decoder_inputs_bucket_dict = {model.decoder_inputs_buckets[bucket_id][0]: decoder_inputs_ids[0]}
        print(decoder_inputs_bucket_dict)
        length_dict = {model.squence_length: encoder_length}
        feed_dict = dict(list(encoder_inputs_bucket_dict.items()) + list(decoder_inputs_bucket_dict.items()) + list(length_dict.items()))
        saver.restore(sess, checkpointfile)
        result = sess.run([model.model_output_predict_merger_buckets[bucket_id]], feed_dict=feed_dict)
    print(result)
    print(", ".join(list(map(lambda x: dataset.all_words[x], result[0][0]))))

if __name__ == '__main__':
    train()
    # predict()