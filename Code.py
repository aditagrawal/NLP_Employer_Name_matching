#!/usr/bin/python3

import pandas as pd 
import numpy as np
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
#import jellyfish
import time
import pickle
import tqdm
import math
import scipy.stats as ss

# remove common words list 
to_remove=['0','-','1st','and','associates','business','care','co','company',
           'global','group','health','tv','isd','ins','uni','dist','mar','comm','com'
           'int','limited','llp','lt','ltd','inc','llc','servi','chemic','scho','securi',
           'of','partners','plc','service','services','corp','schoo','scho','svcs'
           'st','systems','system','the','trust','plc','store','office','com','tv','parts',
           'security','na','unkn','unk','unkown','unkwn','unknown','none','industries','lighting','television',
           'goods','companies','enterprises','bank','corporation','worldwide','district','distri','md','public',
           'center','shop','ri','s','g','ag','joint','1040','2013joint','2012','2013','2014','2015','2016',
           '2017','2018','2019','0','00','000','0000','00000','000000','0000000','00000000','000000000','0000000000',
           'emp','svcs']

def load_object(filename):
    with open(filename, 'rb') as input:
        pickle_object = pickle.load(input)
    return pickle_object
  
  
# removing special characters
spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+",",","-",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–","//",".","/"]

special_case={}
special_case['amex']='american express'
special_case['dominos']='dominos pizza'
special_case['ey']='ernest young'
special_case['j p morgan']='jp morgan'
special_case['j p morgan chase']='jp morgan'
special_case['jpmorgan']='jp morgan'
special_case['jpmorgan chase']='jp morgan'
special_case['pwc']='pricewaterhouse'
special_case['royal bank of scotland']='rbs'

special_case['self']='self employed'
special_case['myself']='self employed'
special_case['usaf']='us air force'
special_case['army']='us army'
special_case['air force']='us air force'
special_case['military']='military'
special_case['navy']='us navy'
special_case['united states air force']='us air force'
special_case['united states navy']='us navy'
special_case['united states army']='us army'
special_case['united states military']='us military'

special_case['albertson']='albertsons'
special_case['alcatellucent']='alcatel lucent'

special_case['self emp']='self employed'
special_case['selfemployed']='self employed'
special_case['ret']='retired'

special_case['flash unknown']=''
special_case['not provided']=''

special_case['wal mart']='walmart'
special_case['wellsfargo']='wells fargo'

special_case['wholefoods']='whole foods'

special_case['a t  t']='att'
special_case['a t t']='att'
special_case['a t and t']='att'
special_case['at t']='att'
special_case['at t mobility']='att mobility'
special_case['chase manhattan']='chase'
special_case['chase manhatt']='chase'

special_case['self employeed']='self employed'
special_case['abbot']='abbott'
special_case['abercrombie'] ='abercrombie fitch'
special_case['accountemps']='accoun temps'
special_case['advanta']='advantage'
special_case['amazon.com']='amazon'
special_case['amazoncom']='amazon'
special_case['america merrill']='america merrill lynch'
special_case['amerisourcebergen']='amerisource bergen'
special_case['andersen']='anderson'
special_case['andersen windows']='anderson windows'

special_case['architectural']='architect'
special_case['artist']='arts'
special_case['assoc']='associate'
special_case['astrazeneca']='astra zeneca'
special_case['avisbudget']='avis budget'
special_case['barclay']='barclays'
special_case['barnes nobles']='barnes noble'
special_case['bd ed']='bd education'
special_case['beachbody']='beach body'
special_case['bear sterns']='bear stearns'
special_case['bearingpoint']='bearing point'
special_case['benefit']='benefits'
special_case['benefis']='benefits'
special_case['berkley']='berkeley'
special_case['bernards']='bernard'
special_case['bershire hathaway']='berkshire hathaway'
special_case['bestbuy']='best buy'
special_case['housewife']='home maker'
special_case['broward sheriff']='broward sheriffs'
special_case['byu idaho']='byuidaho'
special_case['charles schwabb']='charles schwab'
special_case['core logic']='corelogic'
special_case['corner stone']='corner stone'
special_case['country wide']='countrywide'
special_case['cracker barrell']='cracker barrel'
special_case['retirement']='retired'
             
             
numeric_companies={}
numeric_companies['seveneleven']='7 11'
numeric_companies['seven eleven']='7 11'
numeric_companies['seven11']='7 11'
numeric_companies['7eleven']='7 11'
numeric_companies['7 eleven']='7 11'
numeric_companies['seven 11']='7 11'
numeric_companies['711']='7 11'

numeric_companies['threem']='3m'
numeric_companies['three m']='3m'
numeric_companies['3 m']='3m'

parser={}
parser['lab']='laboratories'
parser['labs']='laboratories'

parser['hos']='hospital'
parser['hosp']='hospital'
parser['hospit']='hospital'

parser['universit']='university'
parser['univ']='university'
parser['univer']='university'

parser['adminis']='administration'
parser['administ']='administration'
parser['admin']='administration'
parser['adm']='administration'

parser['market']='marketing'
parser['market']='marketing'
parser['marketin']='marketing'
parser['expr']='express'
parser['expres']='express'
parser['processi']='processing'
parser['fl']='florida'
parser['hotel']='hotels'
parser['ny']='new york'
parser['nyc']='new york'
parser['denv']='denver'
parser['la']='los angeles'
parser['lax']='los angeles'
parser['seven']='7'
parser['hr']='hour'
parser['24hr']='24 hour'


parser['sf']='san francisco'
parser['medic']='medicine'
parser['medicin']='medicine'
parser['builders']='builder'
parser['manageme']='management'
parser['catapillar']='caterpillar'
parser['catepillar']='caterpillar'
parser['agi1040']='agi'
parser['chemical']='chemicals'
parser['zimmermann']='zimmerman'


parser_keys= list(parser.keys())
special_case_keys= list(special_case.keys())

def cleartext(string):
    string,sep,throw=string.partition('/')
    return string

def clean(string):
    string=' '.join(c.lower() for c in string.replace('-',' ').split())    
    string=[w for w in string.split() if w not in to_remove]
    string=[parser[w] if w in parser_keys else w for w in string]
    string=' '.join(c.lower() for c in string)
    return string

def special_handling(string):
    if string in special_case:
        string=special_case[string]
    return string

def numeric_companies_handling(string):
    if string in numeric_companies:
        string=numeric_companies[string]
    return string

def spell_corrector(string):
    string=[w for w in string.split()]
    string=[speller[w] if w in speller_keys else w for w in string]
    string=' '.join(c.lower() for c in string)
    return string

def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

empl_names=pd.read_csv("/axp/rim/mldsml/dev/adit/Employer_data/raw_data/complete_data.csv",nrows=1000)

empl_names.tail()

empl_names['emp_name_clean']=empl_names.EMPLOYMENT_DATA.replace(np.nan,'')
empl_names['emp_name_clean']=empl_names['emp_name_clean'].apply(cleartext)
empl_names['emp_name_clean'] = empl_names['emp_name_clean'].replace('[^a-zA-Z0-9 ]', '', regex=True)
empl_names.emp_name_clean = empl_names.emp_name_clean.replace(r'^\s*$', "", regex=True)
empl_names.emp_name_clean = empl_names.emp_name_clean.str.replace('  ', ' ').replace('   ', ' ').replace('    ', ' ')
empl_names['emp_name_clean']=empl_names.emp_name_clean.apply(numeric_companies_handling)
#empl_names['emp_name_clean'] = empl_names['emp_name_clean'].str.replace('\d+', '')

empl_names['emp_name_clean']=empl_names['emp_name_clean'].apply(clean)

empl_names['emp_name_clean']=empl_names.emp_name_clean.apply(special_handling)


###STEPS BELOW BUILD THE VOCAB

vocab_list=build_vocab(empl_names.emp_name_clean)
vocab_df=pd.DataFrame()
vocab_df['words']=vocab_list.keys()
vocab_df['counts']=vocab_list.values()
vocab_df.sort_values(by='counts',inplace=True,ascending=False)
vocab_df.to_csv("/axp/rim/gpuml/dev/employer_data/vocab_list.csv",index=False)

frequent_words=list(vocab_df[vocab_df.counts>=100].words.values)
infrequent_words=list(vocab_df[vocab_df.counts<100].words.values)

def get_name_simi(words):
    dic={}
    for word in words:
        w_char=str(word)[:3]
        candi_lst=[w for w in frequent_words if str(w)[:3]==w_char]
        #print(word)
        ans=[jellyfish.jaro_winkler(str(word),str(x)) for x in candi_lst]
        if len(ans)>0:
            if max(ans)>=0.95:
                dic[word]=candi_lst[ans.index(max(ans))]
    return dic
dic=get_name_simi(infrequent_words)
import pickle
with open("/axp/rim/mldsml/dev/adit/Employer_data/v4_spell_correction_newjw/infreq_dict.pkl", 'wb') as output:  # Overwrites any existing file.
    pickle.dump(dic, output, pickle.HIGHEST_PROTOCOL)


speller=load_object("/axp/rim/mldsml/dev/adit/Employer_data/v4_spell_correction_newjw/infreq_dict.pkl")
speller["amazone"]

vocab_df=pd.read_csv("/axp/rim/gpuml/dev/employer_data/vocab_list.csv")

vocab_df[vocab_df.words=="amazing"]

speller_keys= list(speller.keys())
print("speller words:",len(speller_keys))

#%%time
empl_names['emp_name_clean']=empl_names.emp_name_clean.apply(spell_corrector)

empl_names.to_csv("/axp/rim/gpuml/dev/employer_data/clean_names.csv",index=False)

empl_names.to_csv("/axp/rim/mldsml/dev/adit/Employer_data/clean_names.csv",index=False)

### analysis part
newdf=pd.DataFrame()
newdf=empl_names.emp_name_clean.value_counts().reset_index()
topdf=newdf[newdf.emp_name_clean>=10]
bottomdf=newdf[newdf.emp_name_clean<10]

topdf.to_csv("/axp/rim/mldsml/dev/adit/Employer_data/top_comps.csv")
bottomdf.to_csv("/axp/rim/mldsml/dev/adit/Employer_data/bottom_comps.csv")

newdf.to_csv("/axp/rim/mldsml/dev/adit/Employer_data/all_comps.csv")

###analysis part
newdf.rename(columns = {'index':'name','emp_name_clean':'freq'}, inplace = True)
biggest_names=newdf[newdf.freq>=1000]
med_names=newdf[newdf.freq>=100]
top_names=newdf[newdf.freq>=10]
bottom_names=newdf[newdf.freq<10]

print(len(biggest_names),len(med_names),len(top_names),len(bottom_names))
print(biggest_names.freq.sum(),med_names.freq.sum(),top_names.freq.sum(),bottom_names.freq.sum())

### below part is to calculate JW similarity to map to parent names

def stringlen(string):
    lens=0
    try:
        lens=len(string)
    except TypeError:
        lens=0
    return lens
  
def process_data(df):
    df.rename(columns = {'index':'name','emp_name_clean':'freq'}, inplace = True)
    df["parent_name"]=df["name"]
    df.sort_values(by=["name","freq"], ascending=[True,False], inplace=True, kind='quicksort')
    df.reset_index(inplace=True)
    return df
  
#%%time
newdf=process_data(newdf)

def jaro_distance(s1, s2) : 
    len1=len(str(s1))
    len2=len(str(s2))
    if (s1 == s2) : 
        return 1.0  
    if (len1 == 0 or len2 == 0) : 
        return 0.0    
    # Maximum distance upto which matching  is allowed  
    max_dist = (max(len1, len2) // 2 ) - 1
    #max_dist2 = (max(len1, len2) // 2 ) - 1
    # Count of matches  
    match = 0       
    # Hash for matches  
    hash_s1 = [0] * len1 
    hash_s2 = [0] * len2
    
    t = 0    
    point = 0
    match=0

    for i in range(len1) :  
        # Check if there is any matches  
        for j in range(max(0, i - max_dist),min(len2, i + max_dist + 1)) :  
            # If there is a match  
            if (s1[i] == s2[j] and hash_s2[j] == 0) :  
                hash_s1[i] = 1  
                hash_s2[j] = 1
                match += 1
                break           

                        
    if (match == 0) : 
        return 0.0   
    for i in range(len1):    
        if (hash_s1[i]) :   
            # Find the next matched character  in second string

            while (hash_s2[point] == 0) : 
                    point += 1
            
            if (s1[i] != s2[point]):
                t += 1
                point+=1
            else:
                point+=1            
        t /= 2
    jd=((match / len1 + match / len2 + (match - t) / match ) / 3.0)
    
            # Find the length of common prefix  
    prefix = 0
  
    for i in range(min(len(s1), len(s2))) : 
        if (s1[i] == s2[i]):
            prefix += 1
        else:
            break
    prefix = min(4, prefix)
    jw =jd + 0.1 * prefix * (1 - jd)
    return jw

  def jwproc(parent_name,names,freq,rng,sim):
    for i in range(rng,len(names)-(rng+1)):
        lister=np.append(names[i-rng:i],names[i+1:i+(rng+1)])
        parentlister=np.append(parent_name[i-rng:i],parent_name[i+1:i+(rng+1)])
        freqlist=list(np.append(freq[i-rng:i],freq[i+1:i+(rng+1)]))
        newfreq=list(np.append(freq[i-rng:i],freq[i+1:i+(rng+1)]))
        similarity=[jaro_distance(str(names[i]),str(x)) for x in lister]
        maxsim=max(similarity)
        rem=0
        if maxsim>=sim:
            newfreq.sort(reverse = True)
            for counter in range(rng):
                if similarity[freqlist.index(newfreq[counter])]>=sim and freq[i]<newfreq[counter]:
                    if rem==0:
                        parent_name[i]=parentlister[freqlist.index(newfreq[counter])]
                    rem=1
    return parent_name
  
#%%time
parent_name=jwproc(newdf.parent_name.values,newdf.name.values,newdf.freq.values,20,0.95)
newdf["parent_name"]=parent_name

newdf.to_csv("/axp/rim/mldsml/dev/adit/Employer_data/v4_spell_correction_newjw/notebook_rng20_sim95.csv")

### Below part is to merge the clean names back

clean_names=pd.read_csv("/axp/rim/mldsml/dev/adit/Employer_data/clean_names.csv")
#rng20_sim95=pd.read_csv("/axp/rim/mldsml/dev/adit/Employer_data/v4_spell_correction_newjw/notebook_rng20_sim95.csv")
rng20_sim90=pd.read_csv("/axp/rim/mldsml/dev/adit/Employer_data/v4_spell_correction_newjw/notebook_rng20_sim90.csv")

def merging(clean_names,parent_names):
    parent_names=parent_names[["name","parent_name","freq"]]
    df=pd.merge(clean_names,parent_names,left_on="emp_name_clean",right_on="name",how="inner")
    df=df[["cust_xref_id","EMPLOYMENT_DATA","emp_name_clean","parent_name"]]
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    print(np.shape(df))
    return df
  
#rng20_sim95_merged=merging(clean_names,rng20_sim95)
rng20_sim90_merged=merging(clean_names,rng20_sim90)

##analysis part
def analysis(df,name):
    newdf=pd.DataFrame()
    newdf=df[name].value_counts().reset_index()
    newdf.rename(columns = {'index':'name','parent_name':'freq'}, inplace = True)
    print(newdf.head())
    biggest_names=newdf[newdf.freq>=1000]
    med_names=newdf[newdf.freq>=100]
    top_names=newdf[newdf.freq>=10]
    bottom_names=newdf[newdf.freq<10]
    print(len(biggest_names),len(med_names),len(top_names),len(bottom_names))
    print(biggest_names.freq.sum(),med_names.freq.sum(),top_names.freq.sum(),bottom_names.freq.sum())
    
##analysis part
#analysis(rng20_sim95_merged,"parent_name")
analysis(rng20_sim90_merged,"parent_name")

#rng20_sim95_merged.to_csv("/axp/rim/mldsml/dev/adit/Employer_data/v4_spell_correction_newjw/final_file_rn20_sim95.csv")
rng20_sim90_merged.to_csv("/axp/rim/mldsml/dev/adit/Employer_data/v4_spell_correction_newjw/final_file_rn20_sim90.csv")


