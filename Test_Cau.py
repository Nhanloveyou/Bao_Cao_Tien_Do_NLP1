import keras

from LoadData import load_data as load
from build_model import build
import pandas as pd
import random as r
import os
class Test(build.Model_RoBERTa):
    def __init__(self,MAX_LEN,PATH,path):
        super().__init__(MAX_LEN,PATH)
        self.path=path
        self.model=super().build_model()
        self.VER = 'v0';
        self.DISPLAY = 1;
        self.skf = load.library.Stra_Kfold(5,True,777)
        self.arr = {
            0: 'neutral',
            1: 'positive',
            2: 'negative'
        }
#Test model
    def TEST_MODEL(self,test):
        # INPUT_IDS
        ct=test.shape[0]
        input_ids_t = load.library.np.ones((ct, self.MAX_LEN), dtype='int32')
        attention_mask_t = load.library.np.zeros((ct, self.MAX_LEN), dtype='int32')
        token_type_ids_t = load.library.np.zeros((ct, self.MAX_LEN), dtype='int32')
        preds_start = load.library.np.zeros(( input_ids_t.shape[0], self.MAX_LEN))
        preds_end = load.library.np.zeros(( input_ids_t.shape[0], self.MAX_LEN))
        for k in range(test.shape[0]):
            text1 = " " + " ".join(test.loc[k,'text'].split())
            enc = load.tokenizer.encode(text1)
            s_tok = load.sentiment_id[test.loc[k,'sentiment']]
            input_ids_t[0, :len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
            attention_mask_t[0, :len(enc.ids) + 5] = 1
        self.model.load_weights(self.path)
        print('Predicting Test...')
        preds = self.model.predict([input_ids_t, attention_mask_t, token_type_ids_t], verbose=self.DISPLAY)
        preds_start += preds[0]
        preds_end += preds[1]
        all = []
        for k in range(input_ids_t.shape[0]):
            a = load.library.np.argmax(preds_start)
            b = load.library.np.argmax(preds_end)
            if a > b:
                print(a)
                st = test.loc[k,'text']
            else:
                text1 = " " + " ".join(test.loc[k,'text'].split())
                enc = load.tokenizer.encode(text1)
                st = load.tokenizer.decode(enc.ids[a - 1:b])
            all.append(st)
        return all
#Load d??? li???u ng?????i d??ng nh???p
    def TEXT(self):
        df = pd.DataFrame(columns=["textID", "text", "sentiment"])
        while True:
            name = input("Nh???p text c???m x??c : ")
            n = int(input("[0:'neutral',1:'positive',2:'negative']=  "))
            while n != 0 and n != 1 and n != 2:
                n = int(input("[0:'neutral',1:'positive',2:'negative']=  "))
            i = r.randint(1000, 9999)
            data = {
                "textID": i,
                "text": name,
                "sentiment": self.arr[n]
            }
            df = df.append(data, ignore_index=True)
            k = int(input("N???u b???n ko c???n text n???a th?? ch???n 0 = "))
            if k == 0:
                break
        return df
        # df[["textID", "text", "sentiment", "selected_text"]].to_csv('D:/UIT LEARN/N??m 3 K?? 2/Python/do_an/doAN/Dataset/submission.csv', index=False)
#????a file CSV ????? load
    @staticmethod
    def TEXT_CSV():
        k = input("Nh???p ???????ng d???n link = ")
        test = pd.read_csv(k).fillna('')
        return test,k
#Test 1 c??u kh??ng c???n l??u ch??? test ????? xem k???t qu???
    def Text_speed_1cau(self,str1,sentiment):
        input_ids_t = load.library.np.ones((1, self.MAX_LEN), dtype='int32')
        attention_mask_t = load.library.np.zeros((1, self.MAX_LEN), dtype='int32')
        token_type_ids_t = load.library.np.zeros((1, self.MAX_LEN), dtype='int32')
        preds_start = load.library.np.zeros((input_ids_t.shape[0], self.MAX_LEN))
        preds_end = load.library.np.zeros((input_ids_t.shape[0], self.MAX_LEN))
        text1 = " " + " ".join(str1.split())
        enc = load.tokenizer.encode(text1)
        s_tok = load.sentiment_id[sentiment]
        input_ids_t[0, :len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
        attention_mask_t[0, :len(enc.ids) + 5] = 1
        #N???u load 5 fold v???i 5 model ???? train th?? k???t qu??? t???t h??n
        # for fold in range(0,model_ct):
        #     self.model.load_weights('/content/drive/MyDrive/NLP/backup/%s-roberta-%i.h5'%(self.VER,fold))
        #     print('Predicting Test...')
        #     preds = self.model.predict([input_ids_t, attention_mask_t, token_type_ids_t],
        #                            verbose=self.DISPLAY)
        #     preds_start += preds[0] / self.skf.n_splits
        #     preds_end += preds[1] / self.skf.n_splits
        self.model.load_weights(self.path)
        preds = self.model.predict([input_ids_t, attention_mask_t, token_type_ids_t],verbose=self.DISPLAY)
        preds_start += preds[0] / self.skf.n_splits
        preds_end += preds[1] / self.skf.n_splits
        a = load.library.np.argmax(preds_start)
        b = load.library.np.argmax(preds_end)
        if a > b:
            st = str1
        else:
            text1 = " " + " ".join(str1.split())
            enc = load.tokenizer.encode(text1)
            st = load.tokenizer.decode(enc.ids[a - 1:b])
        print("text: {}   sentiment: {}   selected_text:  {}".format(str1,sentiment,st))
#K???t qu??? l??u v??o file csv minh t???o ????? l??u khi ng?????i d??ng nh???p
    @staticmethod
    def KQ(test,all):
        test['selected_text'] = all
        if not os.path.isfile('/content/drive/MyDrive/Ky?? thua????t la????p tri??nh Python/Code_Python/submission.csv'):
            test.to_csv('/content/drive/MyDrive/Ky?? thua????t la????p tri??nh Python/Code_Python/submission.csv', index=False)
        else:
            with open('/content/drive/MyDrive/Ky?? thua????t la????p tri??nh Python/Code_Python/submission.csv', 'a', encoding="utf-8",newline='') as f:
                test.to_csv(f, index=False, header=f.tell()==0)
        print(test[['text','selected_text']])
#K???t qu??? l??u v??o file CSV khi ????a v??o ????? test,sample_submission.csv l??u to??n b??? d??? li???u ???????c test t??? file csv (nhi???u file csv)
    @staticmethod
    def KQ_ADD_CSV(test,all,link_file):
        test['selected_text'] = all
        if not os.path.isfile('/content/drive/MyDrive/Ky?? thua????t la????p tri??nh Python/Code_Python/sample_submission.csv'):
            test.to_csv('/content/drive/MyDrive/Ky?? thua????t la????p tri??nh Python/Code_Python/sample_submission.csv', index=False)
        else:
            with open('/content/drive/MyDrive/Ky?? thua????t la????p tri??nh Python/Code_Python/sample_submission.csv', 'a', encoding="utf-8",newline='') as f:
                test.to_csv(f, index=False, header=f.tell()==0)
        test.to_csv(link_file, index=False)
        print(test.sample(2))
path='/content/drive/MyDrive/Ky?? thua????t la????p tri??nh Python/Code_Python/Code_Luong/Backup/v0-roberta-4.h5'
a=Test(load.MAX_LEN,load.PATH,path)
#Test v???i 1 c??u nhanh ko c???n l??u v?? csv:
name = input("Nh???p text c???m x??c : ")
arr = {
            0: 'neutral',
            1: 'positive',
             2: 'negative'
         }
n = int(input("[0:'neutral',1:'positive',2:'negative']=  "))
a.Text_speed_1cau(name,arr[n])


#Test v???i 1 ho???c nhi???u c??u l??u v?? file D:\UIT LEARN\N??m 3 K?? 2\Python\do_an\doAN\Dataset\submission.csv ????y l?? file ch??nh l??u d??? li???u ng?????i d??ng ????a v??o
df=a.TEXT()  #ng?????i d??ng nh???p d?? li???u t??? b??n ph??m (c???n x??? l?? try catch khi ng?????i d??ng nh???p sai ho???c r??ng bu???c)
all=a.TEST_MODEL(df)  # ????a d??? li???u v??o v?? b???t ?????u test xu???t ra kq
a.KQ(df,all)

#Test v???i 1 file csv b???t k?? nhung ph???i c?? header l?? "text", "sentiment" sai ?????nh d???ng c??t
df,link=a.TEXT_CSV()
all=a.TEST_MODEL(df)
a.KQ_ADD_CSV(df,all,link)