from LoadData import load_data as load
from Library import library as l

MAX_LEN = 96
class Process_data():
    def __init__(self, ct1, ct2, MAX_LEN):
        self.ct1 = ct1
        self.ct2 = ct2
        self.MAX_LEN = MAX_LEN
        self.input_ids = l.np.ones((ct1, MAX_LEN), dtype='int32')
        self.attention_mask = l.np.zeros((ct1, MAX_LEN), dtype='int32')
        self.token_type_ids = l.np.zeros((ct1, MAX_LEN), dtype='int32')
        self.start_tokens = l.np.zeros((ct1, MAX_LEN), dtype='int32')
        self.end_tokens = l.np.zeros((ct1, MAX_LEN), dtype='int32')
        self.input_ids_t = l.np.ones((ct2, MAX_LEN), dtype='int32')
        self.attention_mask_t = l.np.zeros((ct2, MAX_LEN), dtype='int32')
        self.token_type_ids_t = l.np.zeros((ct2, MAX_LEN), dtype='int32')

    @staticmethod
    def FIND_OVERLAP(k):
        text1 = " " + " ".join(load.train.loc[k, 'text'].split())
        text2 = " ".join(load.train.loc[k, 'selected_text'].split())
        idx = text1.find(text2)
        chars = l.np.zeros((len(text1)))
        chars[idx:idx + len(text2)] = 1
        if text1[idx - 1] == ' ':
            chars[idx - 1] = 1
        enc = load.tokenizer.encode(text1)
        return idx, chars, enc

    @staticmethod
    def ID_OFFSETS(enc=None):
        offsets = []
        idx = 0
        for t in enc.ids:
            w = load.tokenizer.decode([t])
            offsets.append((idx, idx + len(w)))
            idx += len(w)
        return offsets

    def START_END_TOKENS(self, offsets, chars=None, k=None, enc=None):
        toks = []
        for i, (a, b) in enumerate(offsets):
            sm = l.np.sum(chars[a:b])
            if sm > 0:
                toks.append(i)

        s_tok = load.sentiment_id[load.train.loc[k,'sentiment']]
        self.input_ids[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]
        self.attention_mask[k,:len(enc.ids)+3] = 1
        if len(toks)>0:
            self.start_tokens[k,toks[0]+2] = 1
            self.end_tokens[k,toks[-1]+2] = 1

    def INPUT_IDS(self, k):
        text1 = " "+" ".join(load.test.loc[k,'text'].split())
        enc = load.tokenizer.encode(text1)                
        s_tok = load.sentiment_id[load.test.loc[k,'sentiment']]
        self.input_ids_t[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]
        self.attention_mask_t[k,:len(enc.ids)+3] = 1

a = Process_data(load.train.shape[0], load.test.shape[0], MAX_LEN)

for k in range(load.train.shape[0]):
    idx, chars, enc = a.FIND_OVERLAP(k)
    offsets = a.ID_OFFSETS(enc)
    a.START_END_TOKENS(offsets, chars, k, enc)
for k in range(load.test.shape[0]):
    a.INPUT_IDS(k)