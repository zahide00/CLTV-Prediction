#Görev 1: Veriyi Hazırlama
import pandas as pd
import datetime as dt
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.1f" % x)

# Adım 1: flo_data_20K.csv verisini okuyunuz.
df_ = pd.read_csv(r"C:\Users\arsla\PycharmProjects\pythonProject\zahide\flo_data_20k.csv")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()
df.dropna(inplace=True)

# Adım 2: Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe,variable):
    q1 = dataframe[variable].quantile(0.01)
    q2  = dataframe[variable].quantile(0.99)
    interquantile_range = q2-q1
    up_limit = q2 + 1.5*interquantile_range
    low_limit = q1 - 1.5*interquantile_range
    return low_limit,up_limit

def replace_withthresholds(dataframe, variable ):
    low_limit, up_limit = outlier_thresholds(dataframe,variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = round().low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit)




# Adım 3: "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
#"customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayanız.
replace_withthresholds(df,"order_num_total_ever_online")
list = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
"customer_value_total_ever_online"]

for col in list:
    replace_withthresholds(df,col)

print(dir(list))

# Adım 4: Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.
df["top_alisveris"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
df["top_harcama"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# Adım 5: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
col_dates = df.columns[df.columns.str.contains("date")]
df[col_dates]= df[col_dates].apply(pd.to_datetime)
# df[col_dates].astype("datetime64")

df[col_dates].dtypes
type(col_dates)

#
# Görev 2: CLTV Veri Yapısının Oluşturulması
# Adım 1: Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
#
df["last_order_date"].max()
# "2021-05-30"
date_analyse = dt.datetime(2021,6,2)
df["date_analyse"] = dt.datetime(2021,6,2)




df.columns
#Adım 2: customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i
# oluşturunuz. Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.
cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"]-df["first_order_date"]).astype('timedelta64[D]')/7)
##### Neden tekrar astype diyip datetime çevrildi, zaten çevirmiştik ??
cltv_df["T"] = ((date_analyse - df["first_order_date"]).astype('timedelta64[D]')/7)
cltv_df["frequency"] = df["top_alisveris"]
cltv_df["monetary"] = df["top_harcama"]
cltv_df["monetary_cltv_avg"] = cltv_df["monetary"]/cltv_df["frequency"]
cltv_df = cltv_df[cltv_df["frequency"] >1]
# print(cltv_df[cltv_df["frequency"] == 0]["frequency"])

####### cltv_df["T"] = ((date_analyse - df["first_order_date"])/7).days  ------days ekleyince neden hata veriyor. ?????


# Görev 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
# Adım 1: BG/NBD modelini fit ediniz.
from lifetimes import BetaGeoFitter
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T'])


# • 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv
# dataframe'ine ekleyiniz.
cltv_df["3_months_sales"] = bgf.predict(4*3,
                                        cltv_df['frequency'],
                                        cltv_df['recency_cltv_weekly'],
                                        cltv_df['T'])

###Neden 4*3??????

# • 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv
# dataframe'ine ekleyiniz.

cltv_df["6_months_sales"] = bgf.predict(4*6, cltv_df['frequency'],
                                        cltv_df['recency_cltv_weekly'],
                                        cltv_df['T'])
# Adım 2: Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv
# dataframe'ine ekleyiniz.
from lifetimes import GammaGammaFitter
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                cltv_df['monetary_cltv_avg'])
cltv_df.head()
# Adım 3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
# • Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.


cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv_df["cltv"] = cltv

cltv_df.sort_values("cltv",ascending=False)[:20]
#
# cltv_df["cltv"].sum()
#
# Görev 4: CLTV Değerine Göre Segmentlerin Oluşturulması
# Adım 1: 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
cltv_df["segmentler"] = pd.qcut(cltv_df["cltv"], 4 , labels=[1,2,3,4])

# Adım 2: 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.
#cltv_df.groupby("segmentler").agg({"frequency":("mean","std"),
                                     "monetary_cltv_avg": "std"})
