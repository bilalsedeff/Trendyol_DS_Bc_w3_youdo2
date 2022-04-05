import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import streamlit as st

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from time import sleep

##
df = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data', delimiter=r'\t',
                 names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
r_test = df.pivot(index='user_id', columns='item_id', values='rating').values
r_copy = r_test.copy()

irow, jcol = np.where(~np.isnan(r_copy))

##

idx = np.random.choice(np.arange(100000), size=1000, replace=False)
test_irow = irow[idx]
test_jcol = jcol[idx]

for i in test_irow:
    for j in test_jcol:
        r_copy[i, j] = np.nan


##
# Pearson similarity
def sim(r_: np.ndarray, i: int, j: int, mask: np.ndarray, mu: np.ndarray) -> float:
    num = 0.
    denum = 0.

    for k in range(len(mask[0][1:])):
        num += np.dot((r_[i, mask[0, k]] - mu[i]), (r_[j, mask[0, k]] - mu[j]).T)
        denum += np.dot(np.sqrt((r_[i, mask[0, k]] - mu[i]) ** 2), np.sqrt((r_[j, mask[0, k]] - mu[j]) ** 2))

    if denum == 0:
        return 0.
    else:
        return num / denum


##
def sim_matrix(r_: np.ndarray) -> np.ndarray:
    mu = np.nanmean(r_, axis=1)
    sim_matrix = np.zeros((r_.shape[0], r_.shape[0]))

    for i in range(r_.shape[0]):
        for j in range(r_.shape[0]):
            mask = np.where(~np.isnan(r_[i, :]) & ~np.isnan(r_[j, :]))
            mask = np.asarray(mask)
            sim_matrix[i][j] = sim(r_, i, j, mask, mu)
    return sim_matrix


##
sim_matrix = sim_matrix(r_copy)


##
def top_n(r_: np.ndarray, i: int, j: int, n: int) -> np.ndarray:
    other_users_idx = np.asarray(np.where(~np.isnan(r_[:, j])))[0]
    other_users_idx = other_users_idx[other_users_idx != i]
    sort_other_users = other_users_idx[np.argsort(sim_matrix[i, other_users_idx])[::-1][:n]]
    return sort_other_users


##
def w(top_n: int) -> np.ndarray:
    rnd = np.random.rand(top_n)
    rnd = rnd * 2 - 1
    return rnd


##
def r_hat(r_: np.ndarray, i: int, j: int, mu: np.ndarray, w: np.ndarray, top_n_user, k) -> float:
    top_n_user_ = top_n_user
    top_n_user_list = top_n(r_, i, j, k)

    return mu[i] + np.sum(
        np.dot(w[top_n_user_], (r_[top_n_user_list[top_n_user_], j] - mu[top_n_user_list[top_n_user_]])))


##
def loss(r_: np.ndarray, i: int, j: int, mu: np.ndarray, w: np.ndarray, top_n_user, k) -> float:
    return np.sum(np.sum((r_[i, j] - r_hat(r_, i, j, mu, w, top_n_user, k)) ** 2)) / r_.shape[0]


##
def gradient(r_: np.ndarray, i: int, j: int, mu: np.ndarray, w_: np.ndarray, top_n_users: int) -> float:
    if np.isnan(r_[i, j]):
        r_[i, j] = 0

    return -2 * np.sum((r_[i, j] - (mu[i] + np.sum(np.dot(w_[top_n_users], (r_[i, j] - mu[i])))))) * (
            r_[i, j] - mu[i])


##
def gradient_descent(r_: np.ndarray, i: int, j: int, mu: np.ndarray, w_: np.ndarray,
                     alpha: float, top_n_users: int) -> np.ndarray:
    gradient_ = gradient(r_, i, j, mu, w_, top_n_users)
    w_[top_n_users] = w_[top_n_users] - np.sum(np.dot(alpha, gradient_))
    return w_[top_n_users]


##
def error(r_ij, r_hat):
    return (r_ij - r_hat) ** 2


##
def total_loss(r_test, r_train):
    total_loss = 0.
    for i in range(r_test.shape[0]):
        for j in range(r_test.shape[1]):
            if not np.isnan(r_test[i, j]) & np.isnan(r_train[i, j]):
                total_loss += r_test[i, j] - r_train[i, j]
    return total_loss


##
def w_matrix(r_: np.ndarray, k):
    matrix = (r_.shape[0], k)
    matrix = np.zeros(matrix)
    return matrix


##
matrix_ = w_matrix(r_copy, 4)


##
def fill_w_matrix(matrix, k):
    for i in range(matrix.shape[0]):
        w_ = w(k)
        for t in range(k):
            matrix[i,t] = w_[t]

    return matrix


##
matrix_ = fill_w_matrix(matrix_, 4)


##
def fit(r_test: np.ndarray, r_: np.ndarray, k, epoch, alpha, matrix_: np.ndarray):
    mu = np.nanmean(r_, axis=1)
    m, n = r_.shape
    m = list(range(m))
    performance = []
    predicted_matrix = r_.copy()
    for ep in range(epoch):
        print(f"Epoch: {ep}, Total Loss: {total_loss(r_test, predicted_matrix)}")
        performance.append([ep, total_loss(r_test, predicted_matrix)])
        for i in tqdm(m):
            sleep(0.05)
            w_ = matrix_[i,:]
            for j in range(n):
                if not np.isnan(r_test[i, j]):
                    top_n_users = top_n(r_, i, j, k)
                    r_hat_ = 0.
                    for top in range(len(top_n_users)):
                        w_[top] = gradient_descent(r_, i, j, mu, w_, alpha, top)
                        r_hat_ += r_hat(r_, i, j, mu, w_, top, k)

                    # print("real value: ", r_test[i, j], "  ", "predicted value: ", r_hat_ / k, "error: ", error(r_test[i,j], r_hat_ / k))

                    predicted_value = r_hat_ / k
                    if predicted_value >= 5.:
                        predicted_value = 5.
                    elif predicted_value < 0.:
                        predicted_value = 0.
                    predicted_matrix[i, j] = predicted_value
            matrix_[i, :] = w_

    return predicted_matrix, performance, matrix_


##
def main():
    k = 4
    epoch_ = 20
    alpha_ = 0.00001
    results = fit(r_test, r_copy, k, epoch_, alpha_, matrix_)
    predicted_matrix = results[0]
    performance = results[1]
    matrix = results[2]

    return predicted_matrix, performance, matrix


##
if __name__ == "__main__":
    fitted_r_train = main()[0]
    performance = main()[1]
    matrix_ = main()[2]
    print("Final Loss: ", total_loss(r_test, fitted_r_train))

##
print("Final Loss: ", total_loss(r_test, fitted_r_train))

##
