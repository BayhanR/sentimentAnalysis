import pandas as pd
import re

# Veri setinizi yükleyin
df = pd.read_csv('comments.csv')

# Fonksiyon: Büyük harfleri küçük harfe dönüştürme ve özel karakterleri kaldırma
def temizle_yorum(yorum):
    # Büyük harfleri küçük harfe dönüştürme
    yorum = yorum.lower()
    # Emojileri ve özel karakterleri kaldırma
    yorum = re.sub(r'[^\w\s]', '', yorum)  # Yalnızca harf ve rakamları bırak
    yorum = re.sub(r'[^\x00-\x7F]+', '', yorum)  # Unicode karakterleri kaldır
    # Fazla boşlukları temizle
    yorum = ' '.join(yorum.split())
    return yorum

# Yorum sütununu temizleme
df['cleaned_comment'] = df['comment'].apply(temizle_yorum)

# Yalnızca temizlenmiş yorumları içeren yeni bir DataFrame oluşturma
cleaned_df = df[['cleaned_comment']]

# Temizlenmiş yorumları yeni bir CSV dosyasına kaydetme
cleaned_df.to_csv('temizlenmis_yorumlar.csv', index=False)
