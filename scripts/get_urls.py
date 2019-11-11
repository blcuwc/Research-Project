#coding = utf-8

import sys

con_file = sys.argv[1]
url_file = sys.argv[2]

con_File = open(con_file, 'r')
url_File = open(url_file, 'w')

for line in con_File.readlines()[1:]:
    url = line.strip().split('\t')[3]
    print (url.strip('\"\"'), file=url_File)

url_File.close()
