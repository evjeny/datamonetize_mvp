import streamlit as st

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import pickle
import joblib
from keras.models import load_model


@st.cache
def read_test_data():
    test_data = np.load("bins/test_data.npz", allow_pickle=True)

    x = test_data["x"]
    y = test_data["y"]
    return x, y


@st.cache
def load_models():
    knn = joblib.load("bins/cluster_knn.joblib")
    segment_binarizer = joblib.load("bins/segment_binarizer.joblib")
    segment_net = load_model("bins/segment_net.15_epoch.0.80_val_acc.hdf5")

    return knn, segment_binarizer, segment_net


def display_graph(x, segment_binarizer):
    fig, ax = plt.subplots(1, figsize=(10, 5))

    x = [5]
    for time_delta in x[:-1, -1]:
        x.append(x[-1] + time_delta)
    y = [10] * len(x)

    nodes = segment_binarizer.inverse_transform(x[:, :-1]).reshape(-1)

    jet = cm = plt.get_cmap("jet") 
    a, b = 0, 39
    cNorm  = colors.Normalize(vmin=a, vmax=b)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    color_mapping = dict()
    for color in range(a, b+1):
        color_mapping[color] = scalarMap.to_rgba(color)
    
    y_colors = [color_mapping[y_i] for y_i in y]

    ax.scatter(x, y, c=y_colors)

    return ax


st.title("Segmentator")

xs, ys = read_test_data()
st.write(len(ys), "samples in total")

index = st.slider("Inex to analyze", 0, len(xs)-1)

x_cur = xs[index]
y_cur = ys[index]

knn, segment_binarizer, segment_net = load_models()

st.write(display_graph(x_cur, segment_binarizer))

