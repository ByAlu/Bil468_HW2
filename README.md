# Başlamadan
- https://www.kaggle.com/datasets/alincijov/self-driving-cars veri setinin proje klasörüne "kaggle_dataset" adlı klasör içinde yerleştirin
- CreateDataset.ipynb dosyasında değişiklik yapmakdıysanız son hücreyi çalıştırın.
- ExtractHogFeatures.ipynb dosyasını çalıştırın.

# Dosyalar Hakkında
- CreateDataset.ipynb: Kaggle veri seti içinden ödeve uygun olan frameleri seçerek frames_train/val/test adlı csv dosyalarına ayırır. Hangi çerçevede nesne olduğunu bulur. Ayrıca Frameleri grayscale olarak yeniden boyutlandırarak(256*256) images klasörüne kaydeder.

- ExtractHogFeatures: Oluşuturulan imageler üzerinden istenildiği gibi 64*128 window ile ard arda 2 bölge için hog descriptor çıkarıp bir çerçeve için birleştirerek 7560 boyutunda feature vectorler oluşturur. frames_train/val/test_with_features adlı csv dosyalarına yazar. Her sütunda belirtilen frame için: y 1D bool arrayı çerçevelerin nesneyi içerip içermediği hakkında bilgiyi, features 2D arrayı ise aynı bölgelerin hog featurlarını içerir.

- Çerçeveler her resim için sol üst 1. çerçeve olmak üzere önce yatay sonra dikey eksende artarak devam edecek şekilde indexlenmiştir. 
