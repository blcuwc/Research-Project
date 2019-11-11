#coding: utf-8

from bs4 import BeautifulSoup
import leveldb
import re
import sys
import os
import spacy
from spacy.lang.en import English

def sentencizer_post(post):
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    doc = nlp(post)
#    for sent in doc.sents:
#        print(sent.text)
    return list(doc.sents)

def Link_conId_posts(Url_author_post_dict, dataset_dir):
    con_file_dir = os.path.join(dataset_dir, "conversations")

    con_id_url_dict = {}
    for con_file_name in os.listdir(con_file_dir):
        con_file_path = os.path.join(con_file_dir, con_file_name)
        conFile = open(con_file_path, 'r')
        con_infos = conFile.readlines()[1:]
        for con_info in con_infos:
            con_id = con_info.split('\t')[0].strip().strip('""')
            dis_id = con_info.split('\t')[1].strip().strip('""')
            con_url = con_info.split('\t')[3].strip().strip('""')
            con_id_url_dict[con_url] = (con_id, dis_id)

    conId_author_posts = {}
    for url, author_post_pair in Url_author_post_dict.items():
        con_id, dis_id = con_id_url_dict[url]
        conId_author_posts[(con_id, dis_id)] = author_post_pair

    return conId_author_posts

def Link_posId_posts(conId_author_posts, dataset_dir):
    post_file_dir = os.path.join(dataset_dir, "posts")
    for post_file_name in os.listdir(post_file_dir):
        if post_file_name.split('.')[0] in list(conId_author_posts.keys())[0]:
            break
    post_file_path = os.path.join(post_file_dir, post_file_name)

    postFile = open(post_file_path, 'r')
    post_infos = postFile.readlines()[1:]
    author_postId = {}
    for post_info in post_infos:
        post_id = post_info.split('\t')[0].strip().strip('""')
        con_id = post_info.split('\t')[1].strip().strip('""')
        dis_id = post_info.split('\t')[2].strip().strip('""')
        author = post_info.split('\t')[3].strip().strip('""')
        if (con_id, dis_id) in author_postId:
            author_postId[(con_id, dis_id)].append((author, post_id))
            continue
        author_postId[(con_id, dis_id)] = [(author, post_id)]
       
    conId_posts = {}
    #??????????
    for conId_disId, author_post_list in conId_author_posts.items():
        author_postId_list = author_postId[conId_disId]
        merge_list = []

        print ("post list:", len(author_post_list))
        author_list = []
        for author, post in author_post_list:
            author_list.append(author)
#        print (author_list)
        print ("Id list:",len(author_postId_list))
#        print (author_postId_list)
        
        for i in range(len(author_postId_list)):
            author1 = author_postId_list[i][0]
            post_id = author_postId_list[i][1]
            for author2, post in author_post_list:
                if author1 == author2:
                    merge_list.append((author1, post_id, post))
                    author_post_list = author_post_list[author_post_list.index((author2, post))+1:]
                    break
                else:
                    break
        print ("merge list:", len(merge_list))
        author_list = []
        for author, postid, post in merge_list:
            author_list.append(author)
#        print (author_list)
        conId_posts[conId_disId] = merge_list
    #>??????
    return conId_posts

def get_usernames(info):
    name_list = []
    if re.search(r'name\\\'\">([^&<>\\]+)&', str(info)):
        m = re.search(r'name\\\'\">([^&<>\\]+)&', str(info))
#        print (m.group(1))
        name_list.append(m.group(1))
    names = re.findall(r'name\\\'&gt;([^&<>\\]+)&', str(info))
    name_list.extend(names)
    return name_list

def get_resps(resp):
#    print (str(resp))
#    print (resp)
    resp_list = []
    if re.search(r'^<div[^>]*>(.+)<\/div>$', str(resp)):
        m = re.search(r'^<div[^>]*>(.+)<\/div>$', str(resp))
        resp = m.group(1)
#        resp = m.group(1).replace('<br/>', '').replace('\\n', '').replace('\t', '')
#        resp = re.sub(r'\xa0', '', resp)
#        resp_list.append(resp.strip())
        resp_list.append(resp)
        return resp_list

    if re.search(r'user_19210146.{6}([^\\]+)\\n', str(resp)):
        m = re.search(r'user_19210146.{6}([^\\]+)\\n', str(resp))
#        resp_list.append(m.group(1).strip())
        resp_list.append(m.group(1))

    if re.search(r'(Wow\. I read this.*ramble\.)', str(resp)):
        m = re.search(r'(Wow. I read this.*ramble.)', str(resp))
        resp_list.append(m.group(1))
#        resp_list.append(m.group(1).strip())

    if re.search(r'\\n&lt;br/&gt;', str(resp)):
        m = re.search(r'\\n&lt;br/&gt;', str(resp))
        resp = re.sub(r'\\n&lt;br/&gt;', '', str(resp))

    resps = re.findall(r'text.{4,6}\\n([^\\]+)\\n', str(resp))
    resp_list.extend(resps)
#    print (resp_list)
    return resp_list

def Clean_post(post):
 #   print (post)
    post = post.strip()
    post = post.replace('\\', '')
    post = re.sub(r'\t\t\t', ' ', post)
    post = re.sub(r'\xa0', ' ', post)
    post = re.sub(r'<br\/> <br\/>', '', post)
    post = re.sub(r'<br\/>', '', post)
#    print (post)
    return post

def parse(infile):
    posts = []
    usernames = []
    name_post_list = []
    url_author_post_dict = {}
    for line in open(infile, 'r'):
        frags = line.split('\t', 1)
        if len(frags) == 1:
            continue
        html = frags[1]
        soup = BeautifulSoup(html, "html5lib")
#	print (soup.prettify())

        subj_info = soup.find(class_=re.compile("subj_info"))
        subj_username = subj_info.find("span").string
        usernames.append(subj_username)
#        print (subj_username)
        subj_body = soup.find(class_=re.compile("subj_body"))
        posts.append(subj_body.get_text())

        resp_info = soup.find_all(class_=re.compile("resp_info"))
#        print (resp_info)
        for info in resp_info:
            if info.find("span").string:
                resp_username = info.find("span").string
#                print (resp_username)
                usernames.append(resp_username)
            else:
#                print (info)
                resp_username_list = get_usernames(info)
                usernames.extend(resp_username_list)

        resps_body = soup.find_all(class_=re.compile("resp_body"))
        for resp_body in resps_body:
#            print (resp_body)
            resp_text_list = get_resps(resp_body)
            posts.extend(resp_text_list)

        if len(usernames) != len(posts):
            print (frags[0])
            print (len(usernames))
            print (usernames)
            print (len(posts))
            for post in posts:
                print (post)
        
        for i in range(len(usernames)):
            cleaned_post = Clean_post(posts[i])
#            print (cleaned_post)
            name_post_list.append((usernames[i], cleaned_post))
#        print (name_post_list)

        url_author_post_dict[frags[0]] = name_post_list
        posts = []
        usernames = []
        name_post_list = []

    return url_author_post_dict

def Link_senId_sen(author_postId_post, dataset_dir):
    sen_file_dir = os.path.join(dataset_dir, "sentences")
    for sen_file_name in os.listdir(sen_file_dir):
        if sen_file_name.split('.')[0] in list(author_postId_post.keys())[0]:
            break
    sen_file_path = os.path.join(sen_file_dir, sen_file_name)
    senFile = open(sen_file_path, 'r')
    sen_infos = senFile.readlines()[1:]

    senId_index_dict = {}
    post_list = []
    sen_index_list = []
    postId_count_dict = {}
    for sen_info in sen_infos:
        sen_id = sen_info.split('\t')[0].strip()
        post_id = sen_info.split('\t')[1].strip()
        con_id = sen_info.split('\t')[2].strip()
        start_index = sen_info.split('\t')[4].strip()
        end_index = sen_info.split('\t')[5].strip()
        if post_id in postId_count_dict:
            postId_count_dict[post_id].append(sen_id)
            continue
        postId_count_dict[post_id] = [sen_id]

    Id_sen_dict = {}
    Dis_Id_sen_dict = {}
    for conId_disId, con_list in author_postId_post.items():
#        print ("sen_list:", len(post_list))
#        print ("con_list:", len(con_list))
        conId, disId = conId_disId
        for author, post_id, post in con_list:
#            author, post_id, post = con_list[i]
            splited_post = sentencizer_post(post)
            if len(splited_post) != len(postId_count_dict[post_id]):
                print (post_id)
                print (len(splited_post))
                print (splited_post)
                print (len(postId_count_dict[post_id]))
                print (postId_count_dict[post_id])
                continue
            #splited_len = min(len(splited_post), len(postId_count_dict[post_id]))
            #splited_post = splited_post[:splited_len]
            #sentId_list = postId_count_dict[post_id][:splited_len]
            for i, (sentence, sent_id) in enumerate(zip(splited_post, postId_count_dict[post_id])):
                Id_sen_dict[sent_id] = sentence

#    print (author_postId_post)
    Dis_Id_sen_dict[disId] = Id_sen_dict
    return Dis_Id_sen_dict
    '''
#        print (sen_info)
        if sen_infos.index(sen_info) == 0:
            start_post_id = int(post_id[1][-3:])
#            print (start_post_id)
        if sen_infos.index(sen_info) == len(sen_infos) - 1:
            post_list.append(sen_index_list)
        if int(start_index) not in [0, 1]:
            sen_index_list.append((sen_id, start_index, end_index))
            continue
        if sen_index_list:
            post_list.append(sen_index_list)
        sen_index_list = [(sen_id, start_index, end_index)]

#    print (sen_list)
    Id_sen_dict = {}
    Dis_Id_sen_dict = {}
    for conId_disId, con_list in author_postId_post.items():
        print ("sen_list:", len(post_list))
        print ("con_list:", len(con_list))
        conId, disId = conId_disId
        for i in range(len(con_list)):
            author, post_id, post = con_list[i]
            if int(post_id[-3:]) - start_post_id < len(post_list):
#                print (post_id)
                index_list = post_list[int(post_id[-3:]) - start_post_id]
                con_list[i] = con_list[i] + (index_list, )
#                index_minus = False
                
                for i in range(len(index_list)):
                    sen_id, start_index, end_index = index_list[i]
#                    if i > 0:
#                        if start_index != index_list[i-1][2]:
#                            start_index = index_list[i-1][2]
                    
                    if start_index == "1":
                        Id_sen_dict[sen_id] = post[int(start_index) - 1: int(end_index)]
                    elif start_index == "0":
                        Id_sen_dict[sen_id] = post[int(start_index)  : int(end_index) + 1]
                    else:
                        Id_sen_dict[sen_id] = post[int(start_index)  - 1: int(end_index) ]
        author_postId_post[conId_disId] = con_list
    '''     

def Create_dataset(Dis_Id_sen_dict, dataset_dir, polarity_outfile, factuality_outfile):
    for dis_id, Id_sen_dict in Dis_Id_sen_dict.items():
        pass
    lab_file_dir = os.path.join(dataset_dir, "labels")

    dataset_list = []
    for lab_file_name in os.listdir(lab_file_dir):
#        print (lab_file_name)
        if dis_id not in lab_file_name.split(".")[0]:
            continue
        elif "polarity" in lab_file_name.split(".")[0]:
            pFile = open(polarity_outfile, 'w')
            dataset_list = ["sentence_id\tsentence\tpolarity"]
        else:
            fFile = open(factuality_outfile, 'w')
            dataset_list = ["sentence_id\tsentence\tfactuality"]

        lab_file_path = os.path.join(lab_file_dir, lab_file_name)
        labFile = open(lab_file_path, 'r')
        lab_infos = labFile.readlines()[1:]
        for lab_info in lab_infos:
            sen_id = lab_info.split('\t')[0].strip().strip('""')
            label = lab_info.split('\t')[1].strip().strip('""')
            if sen_id in Id_sen_dict.keys():
                sentence = Id_sen_dict[sen_id]
#                if "\\".encode('utf-8') in sentence.encode('utf-8'):
#                    print (sentence)
#                Id_sen_dict[sen_id] = [sentence, label]
                dataset_list.append(sen_id + "\t" + str(sentence) + "\t" + label)
#        print (dataset_list)
        if "polarity" in dataset_list[0]:
            for item in dataset_list:
                print (item, file=pFile)
            pFile.close()
        else:
            for item in dataset_list:
                print (item, file=fFile)
            fFile.close()

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print ("Usage: python med_parser.py raw_page_file dataset_dir polarity_outfile factuality_outfile")
        sys.exit(-1)

    infile = sys.argv[1]
    dataset_dir = sys.argv[2]
    polarity_outfile = sys.argv[3]
    factuality_outfile = sys.argv[4]

    Url_author_post_dict = parse(infile)
    print (Url_author_post_dict)
    conId_author_posts = Link_conId_posts(Url_author_post_dict, dataset_dir)
    print (conId_author_posts)
    author_postId_post = Link_posId_posts(conId_author_posts, dataset_dir)
    print (author_postId_post)
    Dis_Id_sen_dict = Link_senId_sen(author_postId_post, dataset_dir)
    print (Dis_Id_sen_dict)
    Create_dataset(Dis_Id_sen_dict, dataset_dir, polarity_outfile, factuality_outfile)
