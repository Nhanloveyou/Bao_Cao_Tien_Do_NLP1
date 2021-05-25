# import sklearn.model_selection


from BuildModel import build
from BuildModel.build import Model1
from Library import library as l
from LoadData import load_data as load
# from transformers import tokenizer
import math
from LoadData.load_data import EPOCHS, BATCH_SIZE, PAD_ID, MAX_LEN, PATH
from LoadData import processing_data as p


# %%time

class Train(Model1):
    def __init__(self, MAX_LEN, PATH, input_ids, input_ids_t, attention_mask, attention_mask_t, token_type_ids, token_type_ids_t, start_tokens, end_tokens):
        super().__init__(PAD_ID, MAX_LEN)
        self.PATH = PATH
        self.DISPLAY = 1
        self.VER = 'v0'
        self.jac = []
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.input_ids_t = input_ids_t
        self.token_type_ids = token_type_ids
        self.token_type_ids_t = token_type_ids_t
        self.attention_mask_t = attention_mask_t
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens
        self.skf = l.StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
        self.oof_start = l.np.zeros((input_ids.shape[0], MAX_LEN))
        self.oof_end = l.np.zeros((input_ids.shape[0], MAX_LEN))
        self.preds_start_train = l.np.zeros((input_ids.shape[0],MAX_LEN))
        self.preds_end_train = l.np.zeros((input_ids.shape[0],MAX_LEN))
        self.preds_start = l.np.zeros((input_ids_t.shape[0], MAX_LEN))
        self.preds_end = l.np.zeros((input_ids_t.shape[0], MAX_LEN))

    def accuracy(self):
        print('>>>> OVERALL 5Fold CV Jaccard =', l.np.mean(self.jac))
        return l.np.mean(self.jac)

    def trainModel(self):
        for fold,(idxT,idxV) in enumerate(self.skf.split(self.input_ids, load.train.sentiment.values)):
            print('#'*25)
            print('### FOLD %i'%(fold+1))
            print('#'*25)
            
            l.K.clear_session()
            model, padded_model = super().build_model()
                
            inpT = [self.input_ids[idxT,], self.attention_mask[idxT,], self.token_type_ids[idxT,]]
            targetT = [self.start_tokens[idxT,], self.end_tokens[idxT,]]
            inpV = [self.input_ids[idxV,], self.attention_mask[idxV,],self.token_type_ids[idxV,]]
            targetV = [self.start_tokens[idxV,], self.end_tokens[idxV,]]
            
            shuffleV = l.np.int32(sorted(range(len(inpV[0])), key=lambda k: (inpV[0][k] == PAD_ID).sum(), reverse=True))
            inpV = [arr[shuffleV] for arr in inpV]
            targetV = [arr[shuffleV] for arr in targetV]

            for epoch in range(1, EPOCHS + 1):
                # sort and shuffle: We add random numbers to not have the same order in each epoch
                shuffleT = l.np.int32(sorted(range(len(inpT[0])), key=lambda k: (inpT[0][k] == PAD_ID).sum() + l.np.random.randint(-3, 3), reverse=True))
                # shuffle in batches, otherwise short batches will always come in the beginning of each epoch
                num_batches = math.ceil(len(shuffleT) / BATCH_SIZE)
                batch_inds = l.np.random.permutation(num_batches)
                shuffleT_ = []
                for batch_ind in batch_inds:
                    shuffleT_.append(shuffleT[batch_ind * BATCH_SIZE: (batch_ind + 1) * BATCH_SIZE])
                shuffleT = l.np.concatenate(shuffleT_)
                # reorder the input data
                inpT = [arr[shuffleT] for arr in inpT]
                targetT = [arr[shuffleT] for arr in targetT]
                model.fit(inpT, targetT, 
                    epochs=epoch, initial_epoch=epoch - 1, batch_size=BATCH_SIZE, verbose=self.DISPLAY, callbacks=[],
                    validation_data=(inpV, targetV), shuffle=False)  # don't shuffle in `fit`

            print('Predicting OOF...')
            self.oof_start[idxV,],self.oof_end[idxV,] = padded_model.predict([self.input_ids[idxV,],self.attention_mask[idxV,],self.token_type_ids[idxV,]],verbose=self.DISPLAY)
            
            print('Predicting all Train for Outlier analysis...')
            self.preds_train = padded_model.predict([self.input_ids, self.attention_mask, self.token_type_ids],verbose=self.DISPLAY)
            self.preds_start_train += self.preds_train[0]/self.skf.n_splits
            self.preds_end_train += self.preds_train[1]/self.skf.n_splits

            print('Predicting Test...')
            preds = padded_model.predict([self.input_ids_t,self.attention_mask_t, self.token_type_ids_t],verbose=self.DISPLAY)
            self.preds_start += preds[0]/self.skf.n_splits
            self.preds_end += preds[1]/self.skf.n_splits
            
            # DISPLAY FOLD JACCARD
            all = []
            for k in idxV:
                a = l.np.argmax(self.oof_start[k,])
                b = l.np.argmax(self.oof_end[k,])
                if a>b: 
                    st = load.train.loc[k,'text'] # IMPROVE CV/LB with better choice here
                else:
                    text1 = " "+" ".join(load.train.loc[k,'text'].split())
                    enc = load.tokenizer.encode(text1)
                    st = load.tokenizer.decode(enc.ids[a-2:b-1])
                all.append(super().jaccard(st,load.train.loc[k,'selected_text']))
            self.jac.append(l.np.mean(all))
            print('>>>> FOLD %i Jaccard ='%(fold+1),l.np.mean(all))
            print()

        print('Đã train xong')
        return self.input_ids, self.input_ids_t, self.preds_start_train, self.preds_end_train, self.preds_start, self.preds_end, self.start_tokens, self.end_tokens

# %%
