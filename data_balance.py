import pandas as pd

# Veri setinin yolu
splits = {'train': 'train.csv', 'test': 'test.csv'}
file_path = "hf://datasets/winvoker/turkish-sentiment-analysis-dataset/" + splits["train"]

# Çıktı CSV dosyası ve hata log dosyası
output_csv = 'balanced_sentiment_dataset.csv'
error_log = 'error_log.txt'

# burada notr bileşen en azdı ona referans alarak yaptım
sample_count_per_class = 1448

# İlk olarak CSV'ye header ekleyelim
with open(output_csv, 'w', encoding='utf-8') as f:
    f.write('text,label\n')

# Hata log dosyasını temizleyelim (varsa)
with open(error_log, 'w') as f:
    f.write("")


chunksize = 1000  # Her seferinde 1000 satır okuyacak ramim yetmedi

# Sayma için listeler
notr_samples, positive_samples, negative_samples = [], [], []

try:
    for chunk_number, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize, on_bad_lines='skip')):
        print(f"İşleniyor: {chunk_number * chunksize} ile {(chunk_number + 1) * chunksize} arasındaki satırlar")

        # Sadece gerekli sütunları alalım
        if 'text' not in chunk.columns or 'label' not in chunk.columns:
            print("Beklenen sütunlar bulunamadı. Atlanıyor...")
            continue

        chunk = chunk[['text', 'label']].copy()

        # Tüm label değerlerini küçük harfe çevir
        chunk.loc[:, 'label'] = chunk['label'].str.lower()

        # İlk 5 satırı yazdır (hata ayıklama amacıyla)
        print(chunk.head())

        # Her sınıftan eşit sayıda örnek alalım
        notr_samples.extend(chunk[chunk['label'] == 'notr'].to_dict('records'))
        positive_samples.extend(chunk[chunk['label'] == 'positive'].to_dict('records'))
        negative_samples.extend(chunk[chunk['label'] == 'negative'].to_dict('records'))

        print(
            f"Şu anki birikim: notr={len(notr_samples)}, positive={len(positive_samples)}, negative={len(negative_samples)}")

        # Her sınıftan yeterli sayıda örnek toplandıysa kaydedelim
        if len(notr_samples) >= sample_count_per_class and len(positive_samples) >= sample_count_per_class and len(
                negative_samples) >= sample_count_per_class:
            # Yalnızca gerekli sayıda örnek alalım
            balanced_df = pd.DataFrame(
                notr_samples[:sample_count_per_class] +
                positive_samples[:sample_count_per_class] +
                negative_samples[:sample_count_per_class]
            )

            print(f"Yazılıyor: Toplam {sample_count_per_class * 3} satır ---> CSV")

            # CSV'ye ekle
            balanced_df.to_csv(output_csv, mode='a', header=False, index=False, encoding='utf-8')
            print(f"CSV'ye {sample_count_per_class * 3} satır yazıldı.")

            # Hafızayı temizle
            notr_samples = notr_samples[sample_count_per_class:]
            positive_samples = positive_samples[sample_count_per_class:]
            negative_samples = negative_samples[sample_count_per_class:]

except Exception as e:
    with open(error_log, 'a') as f:
        f.write(
            f"Hata oluştu: Satır grubu {chunk_number * chunksize} ile {(chunk_number + 1) * chunksize} arasında. Hata mesajı: {e}\n")
    print(f"Hata: {e}")

# Kalan örnekleri dosyaya yazma
remaining_samples = notr_samples + positive_samples + negative_samples
if remaining_samples:
    balanced_df = pd.DataFrame(remaining_samples)
    balanced_df.to_csv(output_csv, mode='a', header=False, index=False, encoding='utf-8')
    print(f"CSV'ye kalan {len(remaining_samples)} satır yazıldı.")

print("İşlem tamamlandı. Dengelenmiş veri seti 'balanced_sentiment_dataset.csv' olarak kaydedildi.")