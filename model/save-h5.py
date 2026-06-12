# import zipfile
# import numpy as np
# import keras
# from keras import layers

# # 1. สร้างโมเดลใหม่ตามโครงสร้างเดิม
# model = keras.Sequential([
#     layers.Input(shape=(48,)),
#     layers.Dense(256, activation="relu"),
#     layers.Dropout(0.3),
#     layers.Dense(128, activation="relu"),
#     layers.Dropout(0.3),
#     layers.Dense(64,  activation="relu"),
#     layers.Dense(19,  activation="softmax"),
# ])

# # 2. โหลด weights จากไฟล์ .keras โดยตรง
# model.load_weights("model/MyGestureProject_bata_V1_0.keras")

# # 3. save เป็น .h5
# model.save("model/MyGestureProject_bata_V1_0.h5")
# print("done")