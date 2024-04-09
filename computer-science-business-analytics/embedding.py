import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text # necessary for the BERT layers

class BERTEmbedding(tf.keras.layers.Layer):
    def __init__(self, output_type=-1, **kwargs) -> None:
        preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
        bert_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
        self.embedding_dim = 768

        self.preprocessor = hub.KerasLayer(preprocess_url)
        self.preprocessor = hub.load(preprocess_url)
        self.tokenize = hub.KerasLayer(self.preprocessor.tokenize)
        
        self.bert_pack_inputs = hub.KerasLayer(
        self.preprocessor.bert_pack_inputs,
        arguments=dict(seq_length=256))  # Optional argument.
        
        self.bert = hub.KerasLayer(bert_url)

        self.output_type = output_type
        super().__init__(**kwargs)
    
    def call(self, inputs):
        preproc = self.preprocessor(inputs)
        
        embeddings = self.bert(preproc)
        # tokenized_inputs = self.tokenize(inputs)
        # encoder_inputs = self.bert_pack_inputs(tokenized_inputs)
        # embeddings = self.bert(encoder_inputs)

        # We use the average of the last four hidden layers.
        # Other methods may result in better performance
        if self.output_type == -1:
            return embeddings["pooled_output"]
        elif self.output_type < 12:
            return tf.math.reduce_mean(embeddings['encoder_outputs'][-self.output_type:], axis=0)
        # return average_last_four