from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.sql.functions import col
from pyspark import SparkContext
from pyspark.sql import HiveContext
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from datetime import date, timedelta, datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import *
import os
import sys
from pyspark.sql.functions import col

import warnings
warnings.filterwarnings('ignore')
import time
import pickle

from pyspark.sql.functions import col, regexp_replace, udf, explode, split, lit, count, sum, asc, desc, monotonically_increasing_id, substring, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, FloatType

import pandas as pd
import numpy as np

from pyspark.sql import SparkSession

sqlContext = SparkSession \
        .builder \
        .appName("Employer Name Cleaning Code_App1") \
        .config("spark.executor.memory", "8g") \
        .config("spark.master", "yarn") \
        .config("spark.dynamicAllocation.enabled","true") \
        .config("spark.shuffle.service.enabled","true") \
        .config("spark.dynamicAllocation.minExecutors","20")\
        .config("spark.dynamicAllocation.maxExecutors","30") \
        .config("spark.debug.maxToStringFields","5000") \
        .config("spark.authenticate","true")\
        .config("spark.network.timeout","1800")\
        .config("spark.sql.parquet.enableVectorizedReader","false")\
        .config("spark.authenticate.enableSaslEncryption","true")\
        .config("spark.driver.memory", "10g") \
        .config("spark.executor.cores", "2") \
        .config("spark.speculation", "true") \
        .config("spark.sql.shuffle.partitions","4000") \
        .config("spark.default.parallelism","4000") \
        .config("spark.sql.orc.filterPushdown","true") \
        .config("spark.yarn.executor.memoryOverhead","2048") \
        .config("spark.driver.maxResultSize","4g") \
        .config("spark.shuffle.io.maxRetries","5") \
        .config("spark.sql.parquet.binaryAsString", "true") \
        .config("spark.driver.extraJavaOptions", "-XX:MaxDirectMemorySize=512m -XX:+UseConcMarkSweepGC -XX:+CMSParallelRemarkEnabled -XX:+UseCMSInitiatingOccupancyOnly -XX:CMSInitiatingOccupancyFraction=30 -XX:+ScavengeBeforeFullGC -XX:+CMSScavengeBeforeRemark")\
        .config("spark.executor.extraJavaOptions", "-XX:MaxDirectMemorySize=512m -XX:+UseConcMarkSweepGC -XX:+CMSParallelRemarkEnabled -XX:+UseCMSInitiatingOccupancyOnly -XX:CMSInitiatingOccupancyFraction=30 -XX:+ScavengeBeforeFullGC -XX:+CMSScavengeBeforeRemark")\
        .config("spark.sql.catalogImplementation", "hive") \
        .config("spark.sql.parquet.writeLegacyFormat","true") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch","20000") \
        .config("spark.sql.execution.arrow.enabled","true") \
        .enableHiveSupport() \
        .getOrCreate()
#assert isinstance(sqlContext.sparkContext, object)
sc = sqlContext.sparkContext

from datetime import date
from dateutil.relativedelta import relativedelta

start_date = date.today() - relativedelta(months=12)


### Pulliing past 12 months data for 5 markets
employer_data= sqlContext.sql("""select na_pcn_no, na_emplr_nm, na_curr_cd from cstonedb3.risk_new_acct where na_curr_cd in ('SGD','CAD','HKD','AUD','MXN') and na_final_dt>='"""+str(start_date)+"""'""" )

employer_data=employer_data.na.drop("any")

### Below are functions to compute JW similarity
def _check_type(s):
    if not isinstance(s, str):
        raise TypeError("expected str or unicode, got %s" % type(s).__name__)

def _jaro_winkler(s1, s2, long_tolerance, winklerize):
    _check_type(s1)
    _check_type(s2)
    s1_len = len(s1)
    s2_len = len(s2)
    if not s1_len or not s2_len:
        return 0.0
    min_len = max(s1_len, s2_len)
    search_range = (min_len // 2) - 1
    if search_range < 0:
        search_range = 0
    s1_flags = [False] * s1_len
    s2_flags = [False] * s2_len
    # looking only within search range, count & flag matched pairs
    common_chars = 0
    for i, s1_ch in enumerate(s1):
        low = max(0, i - search_range)
        hi = min(i + search_range, s2_len - 1)
        for j in range(low, hi + 1):
            if not s2_flags[j] and s2[j] == s1_ch:
                s1_flags[i] = s2_flags[j] = True
                common_chars += 1
                break
    # short circuit if no characters match
    if not common_chars:
        return 0.0
    # count transpositions
    k = trans_count = 0
    for i, s1_f in enumerate(s1_flags):
        if s1_f:
            for j in range(k, s2_len):
                if s2_flags[j]:
                    k = j + 1
                    break
            if s1[i] != s2[j]:
                trans_count += 1
    trans_count //= 2
    # adjust for similarities in nonmatched characters
    common_chars = float(common_chars)
    weight = (
                 (
                         common_chars / s1_len
                         + common_chars / s2_len
                         + (common_chars - trans_count) / common_chars
                 )
             ) / 3
    # winkler modification: continue to boost if strings are similar
    if winklerize and weight > 0.7 and s1_len > 3 and s2_len > 3:
        # adjust for up to first 4 chars in common
        j = min(min_len, 4)
        i = 0
        while i < j and s1[i] == s2[i] and s1[i]:
            i += 1
        if i:
            weight += i * 0.1 * (1.0 - weight)
        # optionally adjust for long strings
        # after agreeing beginning chars, at least two or more must agree and
        # agreed characters must be > half of remaining characters
        if (
                long_tolerance
                and min_len > 4
                and common_chars > i + 1
                and 2 * common_chars >= min_len + i
        ):
            weight += (1.0 - weight) * (
                    float(common_chars - i - 1) / float(s1_len + s2_len - i * 2 + 2)
            )
    return weight


def jaro_winkler_similarity(s1, s2, long_tolerance=False):
    return _jaro_winkler(str(s1), str(s2), long_tolerance, True)  # noqa

### Handling speical cases by creating a dictionary
special_case={}
special_case['amex'] = 'american express'
special_case['dominos'] = 'dominos pizza'
special_case['ey'] = 'ernest young'
special_case['j p morgan'] = 'jp morgan'
special_case['j p morgan chase'] = 'jp morgan'
special_case['jpmorgan'] = 'jp morgan'
special_case['jpmorgan chase'] = 'jp morgan'
special_case['pwc'] = 'pricewaterhouse'
special_case['royal bank of scotland'] = 'rbs'
special_case['self'] = 'self employed'
special_case['myself'] = 'self employed'
special_case['usaf'] = 'us air force'
special_case['army'] = 'us army'
special_case['air force'] = 'us air force'
special_case['military'] = 'military'
special_case['navy'] = 'us navy'
special_case['united states air force'] = 'us air force'
special_case['united states navy'] = 'us navy'
special_case['united states army'] = 'us army'
special_case['united states military'] = 'us military'
special_case['alcatellucent'] = 'alcatel lucent'
special_case['self emp'] = 'self employed'
special_case['selfemployed'] = 'self employed'
special_case['ret'] = 'retired'
special_case['flash unknown'] = ''
special_case['not provided'] = ''
special_case['wal mart'] = 'walmart'
special_case['wellsfargo'] = 'wells fargo'
special_case['wholefoods'] = 'whole foods'
special_case['a t  t'] = 'att'
special_case['a t t'] = 'att'
special_case['a t and t'] = 'att'
special_case['at t'] = 'att'
special_case['at t mobility'] = 'att mobility'
special_case['chase manhattan'] = 'chase'
special_case['chase manhatt'] = 'chase'
special_case['self employeed'] = 'self employed'
special_case['abbot'] = 'abbott'
special_case['abercrombie'] = 'abercrombie fitch'
special_case['accountemps'] = 'accoun temps'
special_case['advanta'] = 'advantage'
special_case['amazon.com'] = 'amazon'
special_case['amazoncom'] = 'amazon'
special_case['america merrill'] = 'america merrill lynch'
special_case['amerisourcebergen'] = 'amerisource bergen'
special_case['andersen'] = 'anderson'
special_case['andersen windows'] = 'anderson windows'
special_case['architectural'] = 'architect'
special_case['artist'] = 'arts'
special_case['assoc'] = 'associate'
special_case['astrazeneca'] = 'astra zeneca'
special_case['avisbudget'] = 'avis budget'
special_case['barclay'] = 'barclays'
special_case['barnes nobles'] = 'barnes noble'
special_case['bd ed'] = 'bd education'
special_case['beachbody'] = 'beach body'
special_case['bear sterns'] = 'bear stearns'
special_case['bearingpoint'] = 'bearing point'
special_case['benefit'] = 'benefits'
special_case['benefis'] = 'benefits'
special_case['berkley'] = 'berkeley'
special_case['bernards'] = 'bernard'
special_case['bershire hathaway'] = 'berkshire hathaway'
special_case['bestbuy'] = 'best buy'
special_case['housewife'] = 'home maker'
special_case['broward sheriff'] = 'broward sheriffs'
special_case['byu idaho'] = 'byuidaho'
special_case['charles schwabb'] = 'charles schwab'
special_case['core logic'] = 'corelogic'
special_case['corner stone'] = 'corner stone'
special_case['country wide'] = 'countrywide'
special_case['cracker barrell'] = 'cracker barrel'
special_case['retirement'] = 'retired'


### Handling numeric compaies cases by creating a dictionary
numeric_companies = {}
numeric_companies['seveneleven'] = '7 11'
numeric_companies['seven eleven'] = '7 11'
numeric_companies['seven11'] = '7 11'
numeric_companies['7eleven'] = '7 11'
numeric_companies['7 eleven'] = '7 11'
numeric_companies['seven 11'] = '7 11'
numeric_companies['711'] = '7 11'
numeric_companies['threem'] = '3m'
numeric_companies['three m'] = '3m'
numeric_companies['3 m'] = '3m'

### Handling abbreviatioins by creating a dictionary
parser = {}
parser['lab'] = 'laboratories'
parser['labs'] = 'laboratories'
parser['hos'] = 'hospital'
parser['hosp'] = 'hospital'
parser['hospit'] = 'hospital'
parser['universit'] = 'university'
parser['univ'] = 'university'
parser['univer'] = 'university'
parser['adminis'] = 'administration'
parser['administ'] = 'administration'
parser['admin'] = 'administration'
parser['adm'] = 'administration'
parser['market'] = 'marketing'
parser['market'] = 'marketing'
parser['marketin'] = 'marketing'
parser['expr'] = 'express'
parser['expres'] = 'express'
parser['processi'] = 'processing'
parser['fl'] = 'florida'
parser['hotel'] = 'hotels'
parser['ny'] = 'new york'
parser['nyc'] = 'new york'
parser['denv'] = 'denver'
parser['la'] = 'los angeles'
parser['lax'] = 'los angeles'
parser['seven'] = '7'
parser['hr'] = 'hour'
parser['24hr'] = '24 hour'
parser['sf'] = 'san francisco'
parser['medic'] = 'medicine'
parser['medicin'] = 'medicine'
parser['builders'] = 'builder'
parser['manageme'] = 'management'
parser['catapillar'] = 'caterpillar'
parser['catepillar'] = 'caterpillar'
parser['agi1040'] = 'agi'
parser['chemical'] = 'chemicals'
parser['zimmermann'] = 'zimmerman'


### Remove suffices to clean the company names
to_remove = ['0', '-', '1st', 'and', 'associates', 'business', 'care', 'co', 'company',
             'global', 'group', 'health', 'tv', 'isd', 'ins', 'uni', 'dist', 'mar', 'comm', 'com'
                                                                                            'int', 'limited', 'llp',
             'lt', 'ltd', 'inc', 'llc', 'servi', 'chemic', 'scho', 'securi',
             'of', 'partners', 'plc', 'service', 'services', 'corp', 'schoo', 'scho', 'svcs'
                                                                                      'st', 'systems', 'system', 'the',
             'trust', 'plc', 'store', 'office', 'com', 'tv', 'parts',
             'security', 'na', 'unkn', 'unk', 'unkown', 'unkwn', 'unknown', 'none', 'industries', 'lighting',
             'television',
             'goods', 'companies', 'enterprises', 'bank', 'corporation', 'worldwide', 'district', 'distri', 'md',
             'public',
             'center', 'shop', 'ri', 's', 'g', 'ag', 'joint', '1040', '2013joint', '2012', '2013', '2014', '2015',
             '2016',
             '2017', '2018', '2019', '0', '00', '000', '0000', '00000', '000000', '0000000', '00000000', '000000000',
             '0000000000',
             'emp', 'svcs']

### removing special characters
spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
              "*", "+", ",", "-", ":", ";", "<",
              "=", ">", "?", "@", "[", "\\", "]", "^", "_",
              "`", "{", "|", "}", "~", "?~@~S", "//", ".", "/"]

parser_keys = list(parser.keys())
special_case_keys = list(special_case.keys())


def load_object(filename):
    with open(filename, 'rb') as input:
        pickle_object = pickle.load(input)
    return pickle_object

### Below function removes the addresses from some company names
def cleartext(string):
    string, sep, throw = string.partition('/')
    return string

### Below functions cleans the names
def clean(string):
    string = ' '.join(c.lower() for c in string.replace('-', ' ').split())
    string = [w for w in string.split() if w not in to_remove]
    string = [parser[w] if w in parser_keys else w for w in string]
    string = ' '.join(c.lower() for c in string)
    return string


def special_handling(string):
    if string in special_case:
        string = special_case[string]
    return string


def numeric_companies_handling(string):
    if string in numeric_companies:
        string = numeric_companies[string]
    return string

### Below function creates the vocab of words present in company names
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

### Below function corrects the spelling of words
def spell_corrector(string):
    string = [w for w in string.split()]
    string = [speller[w] if w in speller_keys else w for w in string]
    string = ' '.join(c.lower() for c in string)
    return string


'''
UDFs
'''


### Creating UDFs from python function so that they can be used in pyspark
cleartext_udf = udf(cleartext, StringType())
numeric_companies_handling_udf = udf(numeric_companies_handling, StringType())
clean_udf = udf(clean, StringType())
special_handling_udf = udf(special_handling, StringType())
jaro_winkler_similarity_udf = udf(jaro_winkler_similarity, FloatType())
spell_corrector_udf = udf(spell_corrector, StringType())


### Cleaning raw employer company names
t = time.time()
employer_data = employer_data. \
    fillna({'na_emplr_nm': ''}). \
    withColumn("emp_name_clean", cleartext_udf("na_emplr_nm")). \
    withColumn("emp_name_clean", regexp_replace(col("emp_name_clean"), "[^a-zA-Z0-9 ]", "")). \
    withColumn("emp_name_clean", regexp_replace(col("emp_name_clean"), r'^\s*$', "")). \
    withColumn("emp_name_clean", regexp_replace(col("emp_name_clean"), '  ', ' ')).\
    withColumn("emp_name_clean", regexp_replace(col("emp_name_clean"), '   ', ' ')).\
    withColumn("emp_name_clean", regexp_replace(col("emp_name_clean"), '    ', ' ')).\
    withColumn("emp_name_clean", numeric_companies_handling_udf("emp_name_clean")).\
    withColumn("emp_name_clean", clean_udf("emp_name_clean")).\
    withColumn("emp_name_clean", special_handling_udf("emp_name_clean"))


###STEPS BELOW BUILD THE VOCAB and SPELL CORRECTOR
vocab_df = employer_data.\
    withColumn('word', explode(split(col('emp_name_clean'), ' '))).\
    groupBy('word').\
    count().\
    withColumnRenamed('count', 'counts').\
    sort('counts', ascending=False)

### Dividing vocab words into frequent and infrequent words for spelling corrector
frequent_words_df = vocab_df.filter(col("counts") >= 100).toDF("new_word", "count1")
infrequent_words_df = vocab_df.filter(col("counts") < 100).toDF("old_word", "count2")


### Computing JW siimlarity tbetween sets of frequent words against infrequent words if first 3 characters match
cross_joined_word_df = frequent_words_df.crossJoin(infrequent_words_df).\
    filter(substring(col("new_word"), 1, 3) == substring(col("old_word"), 1, 3)).\
    withColumn('jaro_winkler_score', jaro_winkler_similarity_udf('new_word', 'old_word'))

### Selecting correct frequent word as corrected spelling if JW similarity is more than 95% and if multiple candidates are there then selecting the most similar word
speller_df = cross_joined_word_df.\
    filter(col("jaro_winkler_score")>=0.95).\
    select("new_word", "old_word", "jaro_winkler_score").\
    withColumn('rank', row_number().over(Window().partitionBy("old_word").orderBy(col("jaro_winkler_score").desc()))).\
    filter(col("rank")=="1").\
    select("old_word", "new_word", "jaro_winkler_score").distinct()


raw_value_list = map(lambda row: row.asDict(), speller_df.collect())
speller = {row_value['old_word']: row_value['new_word'] for row_value in raw_value_list}

speller_keys = list(speller.keys())

employer_data = employer_data.\
    withColumn("emp_name_clean", spell_corrector_udf("emp_name_clean"))

### Creating newdf data for use in step 3
newdf=employer_data.groupBy('emp_name_clean','na_curr_cd').count().orderBy('count')

newdf = newdf.selectExpr("emp_name_clean as name", "na_curr_cd as na_curr_cd","count as freq")

newdf=newdf.na.drop("any")
employer_data=employer_data.na.drop("any")


### Making critical columns as non nullable as requirement for EI
def set_df_columns_nullable(spark,df, column_list, nullable=True):
    for struct_field in df.schema:
        if struct_field.name in column_list:
            struct_field.nullable = nullable
    df_mod = spark.createDataFrame(df.rdd, df.schema)
    return df_mod

employer_data=set_df_columns_nullable(sqlContext,employer_data,['na_emplr_nm'])

newdf=set_df_columns_nullable(sqlContext,newdf,['freq'])

sqlContext.sql("$createCSPDatabase")
sqlContext.sql("$useCSPDatabase")

employer_data.registerTempTable("employer_data_temp")
sqlContext.sql("drop table if exists employer_data")
sqlContext.sql("create table if not exists employer_data select na_pcn_no,na_curr_cd,na_emplr_nm,emp_name_clean from employer_data_temp")

#employer_data.write.mode("overwrite").saveAsTable("employer_data")

#newdf.registerTempTable("newdf")
#sqlContext.sql("drop table if exists clean_names_list")
#sqlContext.sql("create table if not exists clean_names_list select * from newdf")

newdf.coalesce(1).write.csv("$baseDir/clean_names_list",header = None,sep='\t',mode="overwrite")

#employer_data.coalesce(1).write.csv("$baseDir/adit_empl_clean/employer_data",header = None,sep='\t',mode="overwrite")


### Writing out the employer data table
insert overwrite directory '$baseDir/employer_data' row format delimited FIELDS TERMINATED BY '\t' select * from employer_data;			

import pandas as pd
import numpy as np
import sys
import glob

import warnings
warnings.filterwarnings('ignore')
import time
import pickle
import os


### Below are functions to compute JW similarity
def _check_type(s):
    if not isinstance(s, str):
        raise TypeError("expected str or unicode, got %s" % type(s).__name__)

def _jaro_winkler(s1, s2, long_tolerance, winklerize):
    _check_type(s1)
    _check_type(s2)
    s1_len = len(s1)
    s2_len = len(s2)
    if not s1_len or not s2_len:
        return 0.0
    min_len = max(s1_len, s2_len)
    search_range = (min_len // 2) - 1
    if search_range < 0:
        search_range = 0
    s1_flags = [False] * s1_len
    s2_flags = [False] * s2_len
    # looking only within search range, count & flag matched pairs
    common_chars = 0
    for i, s1_ch in enumerate(s1):
        low = max(0, i - search_range)
        hi = min(i + search_range, s2_len - 1)
        for j in range(low, hi + 1):
            if not s2_flags[j] and s2[j] == s1_ch:
                s1_flags[i] = s2_flags[j] = True
                common_chars += 1
                break
    # short circuit if no characters match
    if not common_chars:
        return 0.0
    # count transpositions
    k = trans_count = 0
    for i, s1_f in enumerate(s1_flags):
        if s1_f:
            for j in range(k, s2_len):
                if s2_flags[j]:
                    k = j + 1
                    break
            if s1[i] != s2[j]:
                trans_count += 1
    trans_count //= 2
    # adjust for similarities in nonmatched characters
    common_chars = float(common_chars)
    weight = (
                 (
                         common_chars / s1_len
                         + common_chars / s2_len
                         + (common_chars - trans_count) / common_chars
                 )
             ) / 3
    # winkler modification: continue to boost if strings are similar
    if winklerize and weight > 0.7 and s1_len > 3 and s2_len > 3:
        # adjust for up to first 4 chars in common
        j = min(min_len, 4)
        i = 0
        while i < j and s1[i] == s2[i] and s1[i]:
            i += 1
        if i:
            weight += i * 0.1 * (1.0 - weight)
        # optionally adjust for long strings
        # after agreeing beginning chars, at least two or more must agree and
        # agreed characters must be > half of remaining characters
        if (
                long_tolerance
                and min_len > 4
                and common_chars > i + 1
                and 2 * common_chars >= min_len + i
        ):
            weight += (1.0 - weight) * (
                    float(common_chars - i - 1) / float(s1_len + s2_len - i * 2 + 2)
            )
    return weight


def jaro_winkler_similarity(s1, s2, long_tolerance=False):
    return _jaro_winkler(str(s1), str(s2), long_tolerance, True)  # noqa


### Below function will go through alphabetelly sorted data on employer name and for each entery match with precednig and followiing 20 records to get the parent name if JW simialrity iis greater than threhold = 90%
def jwproc(parent_name,names,freq,rng,sim):
    for i in range(rng,len(names)-(rng+1)):
        lister=np.append(names[i-rng:i],names[i+1:i+(rng+1)])
        parentlister=np.append(parent_name[i-rng:i],parent_name[i+1:i+(rng+1)])
        freqlist=list(np.append(freq[i-rng:i],freq[i+1:i+(rng+1)]))
        newfreq=list(np.append(freq[i-rng:i],freq[i+1:i+(rng+1)]))
        similarity=[jaro_winkler_similarity(str(names[i]),str(x)) for x in lister]
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


### Reading data created in previous 2 steps
for file_name in glob.glob("$baseDir/clean_names_list/"+'*.csv'):
    newdf=pd.read_csv(file_name,delimiter="\t",header=None,names=["name","na_curr_cd","freq"])

for file_name in glob.glob("$baseDir/employer_data/"+'*'):
    employer_data=pd.read_csv(file_name,delimiter="\t",header=None,names=["na_pcn_no","na_curr_cd","na_emplr_nm","emp_name_clean"])
    
country_list= ['SGD','CAD','HKD','AUD','MXN']
newdf1= pd.DataFrame()

### JW siimilarity for parant name will be calcualted at market level to avoid mapping companies of different markets together
for country in country_list:
    df1 = newdf[newdf.na_curr_cd==country]
    parent_name=jwproc(df1.name.values,df1.name.values,df1.freq.values,20,0.90)
    df1["parent_name"]=parent_name
    df1=df1[["name","parent_name",'na_curr_cd']]
    newdf1 =pd.concat([newdf1,df1],axis=0)

### Merging data parent names to PCN level data on name key to produce final data at PCN level 
def merging(clean_names,parent_names):

    df=pd.merge(clean_names,parent_names,left_on="emp_name_clean",right_on="name",how="inner")
    df=df[["na_pcn_no","na_emplr_nm","emp_name_clean","parent_name","na_curr_cd_x"]]
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    print(np.shape(df))
    return df

final_df=merging(employer_data,newdf1)

### Creating dummy score to meet EI requirement
final_df['score_nu'] = 1

os.mkdir('$baseDir/Final_output/')
final_df.to_csv('$baseDir/Final_output/Final_clean_empl_names',index=None,header=None,sep='\t')
