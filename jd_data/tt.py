
fw=open('./tt.txt','w')

for ele in open('./jd_data.txt','r').readlines():
    if ele not in ['\n']:
        fw.write(ele)