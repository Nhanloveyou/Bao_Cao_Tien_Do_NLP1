from Library import library as l

MAX_LEN = 96
PATH = '/content/drive/MyDrive/Do_An_Python/Dataset/'
tokenizer = l.tokenizers.ByteLevelBPETokenizer(
    vocab=PATH + 'vocab-roberta-base.json',
    merges=PATH + 'merges-roberta-base.txt',
    lowercase=True,
    add_prefix_space=True
)
EPOCHS = 3 # originally 3
BATCH_SIZE = 32 # originally 32
PAD_ID = 1
SEED = 88888
LABEL_SMOOTHING = 0.1
l.tf.random.set_seed(SEED)
l.np.random.seed(SEED)
sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
train = l.pd.read_csv('/content/drive/MyDrive/Do_An_Python/Dataset/train.csv').fillna('')
test = l.pd.read_csv('/content/drive/MyDrive/Do_An_Python/Dataset/test.csv').fillna('')
print(train.head())
