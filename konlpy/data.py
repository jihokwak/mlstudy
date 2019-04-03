from konlpy.corpus import kolaw, kobill
#kolaw : 한국 법률 말뭉치 'constitution.txt'로 저장됨
#kobill : 대한민국 국회 의안 말뭉치. '1809890.txt ~ 1809899.txt'

kolaw.open('constitution.txt').read()[:100]
kobill.open('1809890.txt').read()

import re

text = "wow, it is awesome"
re.search("(\w+)", text)