#coding = utf-8
import sys
import os

sen_dir = sys.argv[1]
lab_dir = sys.argv[2]
data_dir = sys.argv[3]

for sen_file_name in os.listdir(sen_dir):
    sen_tag = sen_file_name.split('.')[0]
    sen_File = open(os.path.join(sen_dir, sen_file_name), 'r')
    sen_dict = {}

    for sen in sen_File.readlines():
        sen_id = sen.split('\t')[0]
        sen_text = sen.split('\t')[4].strip('\n')
        sen_dict[sen_id] = sen_text

    for lab_file_name in os.listdir(lab_dir):
        if sen_tag in lab_file_name.split('.')[0]:
            lab_File = open(os.path.join(lab_dir, lab_file_name), 'r')
            data_File = open(os.path.join(data_dir, lab_file_name), 'w')
        for lab in lab_File.readlines():
            lab_id = lab.split('\t')[0]
            label = lab.split('\t')[1].strip('\n')
            if sen_dict.__contains__(lab_id):
                print ('%s\t%s' % (sen_dict[lab_id].strip('""'), label.strip('""')), file = data_File)
        
