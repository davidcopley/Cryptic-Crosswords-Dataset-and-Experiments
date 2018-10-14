import pandas as pd
import math
import fasttext
from keras.preprocessing.text import text_to_word_sequence,Tokenizer
import pickle
from model_trainer import cc_data
from model_trainer import confusion_matrix

cc_types = ['is_anagram', 'is_homophone', 'is_double', 'is_cryptic', 'is_contain', 'is_reverse', 'is_alternate', 'is_init', 'is_delete', 'is_&lit', 'is_hidden', 'is_spoonerism', 'is_palindrome']
upsampled_input_cc_types_df,val_cc_types_df,test_cc_types_df = cc_data.get_cc_data_without_charades()

tokenizer = Tokenizer(filters='"#$%&()*+-/:;<=>?@[\]^_`{|}~')
tokenizer.fit_on_texts(pd.concat([upsampled_input_cc_types_df,val_cc_types_df,test_cc_types_df])['clue'])

cc_input_df = upsampled_input_cc_types_df[['clue','category']]
cc_val_df = val_cc_types_df.drop_duplicates()[['clue','category']]
cc_test_df = test_cc_types_df.drop_duplicates()[['clue','category']]

cc_input_df['clue'] = cc_input_df['clue'].apply(lambda x:' '.join(text_to_word_sequence(x)))
cc_val_df['clue'] = cc_val_df['clue'].apply(lambda x:' '.join(text_to_word_sequence(x)))
cc_test_df['clue'] = cc_test_df['clue'].apply(lambda x:' '.join(text_to_word_sequence(x)))

cc_input_df['fasttext_input'] = cc_input_df['category'].apply(lambda x:'__label__'+x+' ,').map(str)+cc_input_df['clue']
cc_val_df['fasttext_input'] = cc_val_df['category'].apply(lambda x:'__label__'+x+' ,').map(str)+cc_val_df['clue']
cc_test_df['fasttext_input'] = cc_test_df['category'].apply(lambda x:'__label__'+x+' ,').map(str)+cc_test_df['clue']

with open("cc_input_df.txt",'w') as f:
    f.writelines([x+'\n' for x in cc_input_df['fasttext_input'].tolist()])

with open("cc_val_df.txt",'w') as f:
    f.writelines([x+'\n' for x in cc_val_df['fasttext_input'].tolist()])

epochs = [5,10,50,100]
dims = [5,10,50,100]

for epoch in epochs:
    for dim in dims:
        model = fasttext.supervised('cc_input_df.txt','temp_model', epoch=epoch, dim=dim)
        cc_types_dict = {k: v for v, k in enumerate(cc_types)}
        results = model.test('cc_val_df.txt')
        pres = str(round(results.precision, 2))
        print(pres)
        val_pred = model.predict(cc_val_df['clue'])
        val_pred = [cc_types_dict[pred[0]] for pred in val_pred]
        val_out = [cc_types_dict[pred] for pred in cc_val_df['category'].tolist()]
        cfm = confusion_matrix.generate_confusion_matrix(val_out,val_pred)
        confusion_matrix.plot_confusion_matrix_png(cfm,'fasttext - epochs:{} - dim:{} - pres:{}'.format(epoch,dim,pres))
