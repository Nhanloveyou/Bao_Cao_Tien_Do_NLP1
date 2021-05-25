from transformers import RobertaConfig, TFRobertaModel

from Library import library as l
from LoadData.load_data import PATH, LABEL_SMOOTHING, MAX_LEN
from LoadData import load_data as load


class Model1():
    def __init__(self, PAD_ID = 1, MAX_LEN = 96):
        self.MAX_LEN=MAX_LEN
        self.PAD_ID = PAD_ID
        self.ids = l.tf.keras.layers.Input((MAX_LEN,), dtype=l.tf.int32)
        self.att = l.tf.keras.layers.Input((MAX_LEN,), dtype=l.tf.int32)
        self.tok = l.tf.keras.layers.Input((MAX_LEN,), dtype=l.tf.int32)
        self.padding = l.tf.cast(l.tf.equal(self.ids, PAD_ID), l.tf.int32)
        self.Dropout_new = 0.15     # originally 0.1
        self.n_split = 5            # originally 5
        self.lr = 3e-5   

        self.lens = MAX_LEN - l.tf.reduce_sum(self.padding, -1)
        self.max_len = l.tf.reduce_max(self.lens)
        self.ids_ = self.ids[:, :self.max_len]
        self.att_ = self.att[:, :self.max_len]
        self.tok_ = self.tok[:, :self.max_len]


    def save_weights(self, model, dst_fn):
        weights = model.get_weights()
        with open(dst_fn, 'wb') as f:
            l.pickle.dump(weights, f)


    def load_weights(self, model, weight_fn):
        with open(weight_fn, 'rb') as f:
            weights = l.pickle.load(f)
        model.set_weights(weights)
        return model

    def loss_fn(self, y_true, y_pred):
        # adjust the targets for sequence bucketing
        ll = l.tf.shape(y_pred)[1]
        y_true = y_true[:, :ll]
        loss = l.tf.keras.losses.categorical_crossentropy(y_true, y_pred,
            from_logits=False, label_smoothing=LABEL_SMOOTHING)
        loss = l.tf.reduce_mean(loss)
        return loss

    def build_model(self):
        config = RobertaConfig.from_pretrained(PATH + 'config-roberta-base.json')
        bert_model = TFRobertaModel.from_pretrained(PATH + 'pretrained-roberta-base.h5', config=config)
        x = bert_model(self.ids_, attention_mask=self.att_, token_type_ids=self.tok_)

        x1 = l.tf.keras.layers.Dropout(self.Dropout_new)(x[0])
        x1 = l.tf.keras.layers.Conv1D(768, 2,padding='same')(x1)
        x1 = l.tf.keras.layers.LeakyReLU()(x1)
        x1 = l.tf.keras.layers.Conv1D(64, 2,padding='same')(x1)
        x1 = l.tf.keras.layers.Dense(1)(x1)
        x1 = l.tf.keras.layers.Flatten()(x1)
        x1 = l.tf.keras.layers.Activation('softmax')(x1)
        
        x2 = l.tf.keras.layers.Dropout(self.Dropout_new)(x[0]) 
        x2 = l.tf.keras.layers.Conv1D(768, 2,padding='same')(x2)
        x2 = l.tf.keras.layers.LeakyReLU()(x2)
        x2 = l.tf.keras.layers.Conv1D(64, 2, padding='same')(x2)
        x2 = l.tf.keras.layers.Dense(1)(x2)
        x2 = l.tf.keras.layers.Flatten()(x2)
        x2 = l.tf.keras.layers.Activation('softmax')(x2)

        model = l.tf.keras.models.Model(inputs=[self.ids, self.att, self.tok], outputs=[x1,x2])
        optimizer = l.tf.keras.optimizers.Adam(learning_rate=self.lr) 
        model.compile(loss=self.loss_fn, optimizer=optimizer)
        
        # this is required as `model.predict` needs a fixed size!
        x1_padded = l.tf.pad(x1, [[0, 0], [0, self.MAX_LEN - self.max_len]], constant_values=0.)
        x2_padded = l.tf.pad(x2, [[0, 0], [0, self.MAX_LEN - self.max_len]], constant_values=0.)
        
        padded_model = l.tf.keras.models.Model(inputs=[self.ids, self.att, self.tok], outputs=[x1_padded,x2_padded])
        return model, padded_model
    

    @staticmethod
    def jaccard(str1, str2):
        a = set(str1.lower().split())
        b = set(str2.lower().split())
        if (len(a) == 0) & (len(b) == 0): return 0.5
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
