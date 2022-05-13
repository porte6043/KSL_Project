from datetime import datetime
import json
import os

class Json_Data():
    def __init__(self):
        self.dt = datetime.now()
        self.date = str(self.dt.date()).replace("-","")[2:]
        self.lenght = None

    def extract_data(self, path, lenght=None, save_word_label=True, save_label=True): # path : path of json
        self.word = []
        self.word_label = {}
        self.time = []
        self.label = []
        self.error_data = {}
        self.save_word_label = save_word_label
        self.save_label = save_label
        self.path_num = path.split('/')[-2]
        
        if lenght == None:
            pass
        else:
            self.lenght = lenght
        
        self.file_names = os.listdir(path)
    
        for idx , file_name in enumerate(self.file_names):
            self.error = False
            self.file_name = '_'.join(file_name.split('_')[:-1])

            with open(path + file_name, 'r', encoding= 'UTF8') as jsonfile:
                self.json_data = json.load(jsonfile)

            self.data_word(idx)
            self.data_time(idx)
            self.data_label(idx)
            self.data_dict(idx)
            
            if lenght == None:
                pass
            elif len(self.word) == self.lenght:
                break

        self.lenght = len(self.word)
        
    def data_word(self, idx):
        try:
            self.word.append( self.json_data['data'][0]['attributes'][0]['name'] )
        except IndexError:
            # print("{} : {}".format(idx, self.file_name))
            self.word.append( False )
            self.error_data[idx//5] = self.file_name
            self.error = True

    def data_time(self, idx):
        try:
            self.time.append( [ self.json_data['data'][0]['start'] , self.json_data['data'][0]['end'] ] )
        except IndexError:
            self.time.append( [-1,-1] )
            self.error_data[idx//5] = self.file_name
            self.error = True

    def data_label(self, idx):
        self.Q = idx // 5

        if self.error == False:
            self.label.append(self.Q)
        else:
            self.label.append(-1)

    def data_dict(self, idx):
        if self.error == False:
            self.word_label[self.Q] = self.word[idx]
        else:
            if self.word_label.get(self.Q) == None:
                self.word_label[self.Q] = False
            
            
    def data_save(self):
        try:
            if not os.path.exists('../saved_data/{}_{}'.format(self.date, self.lenght)):
                os.makedirs('../saved_data/{}_{}'.format(self.date, self.lenght))
        except OSError:
            pass
        if self.save_word_label == True:
            with open('../saved_data/{}_{}/word_labeling_{}'.format(self.date, self.lenght, self.path_num), 'w', encoding ='UTF8') as word_labeling:
                json.dump(self.word_label, word_labeling, indent=1)

        with open('../saved_data/{}_{}/time_{}'.format(self.date, self.lenght, self.path_num), 'w', encoding ='UTF8') as time:
            json.dump(self.time, time)

        if self.save_label == True:
            with open('../saved_data/{}_{}/label_{}'.format(self.date, self.lenght, self.path_num), 'w', encoding ='UTF8') as label:
                json.dump(self.label, label)

        with open('../saved_data/{}_{}/error_{}'.format(self.date, self.lenght, self.path_num), 'w', encoding ='UTF8') as err:
            json.dump(self.error_data, err, indent=1)

    def data_print(self):
        print('-'*50)
        print("word , len={}".format(len(self.word)))
        print(self.word)

        print('-'*50)
        print("time , len={}".format(len(self.time)))
        print(self.time)

        print('-'*50)
        print("label , len={}".format(len(self.label)))
        print(self.label)
        print('-'*50)

        print('-'*50)
        print("word_dict , len={}".format(len(self.label)))
        print(self.word_label)
        print('-'*50)

if __name__ == "__main__":
    path = 'E:/수어 영상/1.Training/[라벨]01_real_word_morpheme/morpheme/01/'
    JD = Json_Data()
    JD.extract_data(path, lenght=13)
    JD.data_print()
    JD.data_save()
    