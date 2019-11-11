#coding=utf-8

import sys
import urllib
import requests
from urllib.request import Request
from urllib.request import urlopen
from urllib.request import build_opener
from bs4 import BeautifulSoup
import leveldb
import re
import time
import socket
import time
from io import StringIO
from io import BytesIO
import gzip
import random

#set default timeout
socket.setdefaulttimeout( 30 )

def requestPage(url, headers = {}):
    if "Accept-encoding" not in headers:
        headers.update({'Accept-encoding': 'gzip'})
    if "User-Agent" not in headers:
        headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36'})
    if "x-requested-with" not in headers:
        headers.update({'x-requested-with': 'XMLHttpRequest'})
    r = requests.get(url, headers = headers)
    return r.text


def getPage(url, data = None, retry=3, interval=0.5, headers = {}, proxy = None):
    #time.sleep(1)
    t = 0
    while t < retry:
        if t > 1:
            proxy = None
        fd = None
        try:
            if data != None:
                request = Request(url, data)
            else:
                request = Request(url)
#                print("getPage" + url)
            if "Accept-encoding" not in headers:
                request.add_header('Accept-encoding', 'gzip')
            if "User-Agent" not in headers:
                request.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36')
            if "x-requested-with" not in headers:
                request.add_header('x-requested-with', 'XMLHttpRequest')
            for k,v in headers.items():
                request.add_header(k, v)
            if proxy != None:
                request.set_proxy(proxy, "http")
#            opener = build_opener()

#            fd = opener.open(request)
            fd = urlopen(request)
            if fd != None:
                #if 'text/html' not in fd.headers.get("Content-Type") and \
                #    "text/xml" not in fd.headers.get("Content-Type") and \
                #    "text/css" not in fd.headers.get("Content-Type")  :
                #    return None

                contentEncoding = fd.headers.get("Content-Encoding")
                data = None
                if contentEncoding != None and 'gzip' in contentEncoding:
                    compresseddata = fd.read()
                    compressedstream = BytesIO(compresseddata)
                    gzipper = gzip.GzipFile(fileobj=compressedstream)
                    data = gzipper.read()
                else:
                    data = fd.read()

                return data
        except Exception as e:
            print ("download %s by proxy %s error: %s" %(url, proxy,str(e)), file=sys.stderr)
            if hasattr(e,'code') and e.code == 404:
                return ''
            t += 1
            time.sleep(interval * t)

        if fd != None:  fd.close()

    return ''

def toLine(page):
    '''
    pageLine = ''
    for line in page:
        if str(line).strip() == "":
            continue
        line = '%s\t\t' % str(line).strip()
        pageLine = '%s%s' % (pageLine, line)
    return pageLine
    '''
#    print (type(page))
    page = re.sub(r'[\r\n]{10,}', '\n', page.decode('utf-8'))
    return re.sub(r'[\r\n]', '\t\t\t', page)

def scroll_pages(url, raw_page):
    raw_pages = []
    soup = BeautifulSoup(raw_page, "lxml")
    new_links = soup.select('a[class="vertical_scroll_link"]')

    if new_links:
        has_new_link = True
    else:
        has_new_link = False
        return ""

    next_link = new_links[0].attrs['href']

    while has_new_link:
        next_link = url + re.findall(r'(\.js\?page=\d)', next_link)[0]
#        print ("next_link:" + next_link)

        raw_page = getPage(next_link)
        raw_page = toLine(raw_page)
#        raw_page = requestPage(next_link)
        raw_pages.append(raw_page)

        soup = BeautifulSoup(raw_page, "lxml")
        for link in soup.find_all('a'):
            if re.search(r"\?page=\d", link.get('href')):
                next_link = link.get('href').strip('""')
            else:
                next_link = None

        if next_link:
            has_new_link = True
        else:
            has_new_link = False
    return raw_pages

def Process(infile_name, raw_page_file):
    lineCnt = 0
    raw_page_File = open(raw_page_file, 'w')

    for line in open(infile_name):
        url = line.strip().split('\t')[0]
        lineCnt += 1
        time.sleep(1)
        first_raw_page = getPage(url)
        #first_raw_page = requestPage(url)

        if first_raw_page == None:
            first_raw_page = ''
        first_raw_page = toLine(first_raw_page)

        if len(first_raw_page) < 3000:
            print ("error", url)
        print (lineCnt, url)
        
        raw_pages = scroll_pages(url, first_raw_page)

        if raw_pages:
            raw_page = first_raw_page + '\t' + '\t'.join(raw_pages)
        else:
            raw_page = first_raw_page
        print ('%s\t%s' % (url, raw_page), file=raw_page_File)

        raw_page_File.flush()

        time.sleep(1)

    raw_page_File.close()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print ("Usage: python crawl_med.py infile raw_page_file")
        sys.exit(-1)
    infile_name = sys.argv[1]
    raw_page_file = sys.argv[2]
    Process(infile_name, raw_page_file)
