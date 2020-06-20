import streamlit as st

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import pickle
import joblib
from tensorflow.keras.models import load_model


@st.cache
def read_test_data():
    test_data = np.load("bins/test_data.npz", allow_pickle=True)

    x = test_data["x"]
    y = test_data["y"]
    return x, y


def load_models():
    knn = joblib.load("bins/cluster_knn.joblib")
    segment_binarizer = joblib.load("bins/segment_binarizer.joblib")
    segment_net = load_model("bins/segment_net.15_epoch.0.80_val_acc.hdf5")

    return knn, segment_binarizer, segment_net


def display_graph(x, segment_binarizer):
    fig, ax = plt.subplots(1, figsize=(10, 5))

    ax.set_title("Граф покупок пользователя")
    ax.set_xlabel("Время в днях")

    x_data = [0]
    for time_delta in x[:-1, -1]:
        x_data.append(x_data[-1] + time_delta)
    y_data = [10] * len(x_data)

    ax.plot([x_data[0], x_data[-1]], [y_data[0], y_data[-1]], c="black", zorder=0)

    nodes = segment_binarizer.inverse_transform(x[:, :-1]).reshape(-1)

    jet = cm = plt.get_cmap("jet") 
    a, b = 0, 39
    cNorm  = colors.Normalize(vmin=a, vmax=b)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    color_mapping = dict()
    for color in range(a, b+1):
        color_mapping[color] = scalarMap.to_rgba(color)
    
    y_colors = [color_mapping[node] for node in nodes]

    ax.scatter(x_data, y_data, c=y_colors, s=80, zorder=1)

    return fig


def data_in_n_days(x, n, segment_net):
    x_batch = x.reshape(1, x.shape[0], x.shape[1])
    x_batch[-1, -1] = n

    prediction = segment_net.predict(x_batch).reshape(-1)

    fig, ax = plt.subplots(1, figsize=(8, 4))
    ax.set_title("Вероятности перехода в новые сегменты")
    ax.set_xlabel("id сегмента")
    ax.set_ylabel("вероятность")
    ax.set_ylim(-0.05, 1.05)

    ax.bar(range(len(prediction)), prediction)

    return fig


st.title("Segmentator")

knn, segment_binarizer, segment_net = load_models()

xs, ys = read_test_data()
st.write("Загружено", len(ys), "семплов")

index = st.radio("Какого пользователя анализировать", (0, 2390, 49074))

x_cur = xs[index]
y_cur = ys[index]

st.write(display_graph(x_cur, segment_binarizer))

n = st.slider("Сколько дней анализировать", 1, 40)
days = [1] + list(range(5, n+1, 5))

st.write("### Прогноз переходов на ближайшие {} дней".format(n))

for cur_days in days:
    st.write("Через", cur_days, "дней:")
    st.write(data_in_n_days(x_cur, cur_days, segment_net))
