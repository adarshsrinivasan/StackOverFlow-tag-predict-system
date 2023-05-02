#!/usr/bin/env python
# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from bs4 import BeautifulSoup

spark = SparkSession.builder.appName('BDA StackOverflow Tag prediction').getOrCreate()
questions_table = 'bigquery-public-data.stackoverflow.posts_questions'
df = spark.read.format('bigquery').load(questions_table)

df = df.select('body', 'answer_count', 'creation_date', 'tags', 'score', 'title')
df = df.where('answer_count > 0 AND creation_date >= DATE_SUB(CURRENT_TIMESTAMP(), 600) AND score > 3')


def convert_body(text):
    a = BeautifulSoup(text).get_text()
    a = [i for i in a if i not in '!"$%&\'#()*+,./:;<=>?@[\\]^_`{|}~']
    a = ''.join(a)
    return a


covert_body_udf = udf(convert_body, StringType())

df = df.withColumn("body", covert_body_udf(df['body']))
df = df.withColumn("title", covert_body_udf(df['title']))

path = "gs://bda-dataproc/query-result"
print('Writing table out to {}'.format(path))
df.write.format("csv").option("header", "true").mode("overwrite").save(path)
