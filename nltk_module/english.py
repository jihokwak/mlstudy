import nltk
nltk.download()

from nltk.tokenize import word_tokenize

sentence = '''Scripts are organized as projects. Projects can be of two types, standalone and
bounded to a gtype (Google Drive native file type, such as Sheets, Docs, and Forms)
file. Standalone scripts are created in a separate script file, you can see these files
listed among other files in Drive. Bounded scripts are embedded within individual
gtype files and created using the respective applications. As you can see, the
standalone script files, among other files in Drive, you can open directly from Drive,
but bounded script can be opened within respective applications only. However,
bounded script will have more privileges over parent file than standalone scripts. For
example, you can get access to the active document within bounded scripts, but not
within standalone scripts.'''

word_tokenize(sentence)

from nltk.tokenize import sent_tokenize

sent_tokenize(sentence)