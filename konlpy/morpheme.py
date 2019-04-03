from konlpy.tag import Mecab, Okt, Kkma, Twitter, Komoran

#Mecab 현존 가장 빠름 ( > Twitter )
#Oak Twitter에서 이름이 변경됨
#KKma 정확한 품사 정보를 추출
#Komoran 정확성, 시간 모두 중요할때
text = "한글 자연어 처리는 재밌다 이제부터 열심히 해야지 ㅎㅎㅎ"
mecab = Mecab()
mecab.morphs(text)

kkma = Kkma()
kkma.morphs(text)

komoran = Komoran()
komoran.morphs(text)

okt = Okt()
okt.morphs(text, stem=True)
okt.pos(text, stem=True) #스테밍해서 품사태깅
okt.pos(text, join=True) #품사태깅을 붙여서 리스트화




