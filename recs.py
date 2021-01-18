from torch import load, device, sigmoid
from mf import MF
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch
import pickle
import json


plt.style.use('ggplot')


def from_checkpoint(path, gpu=True):
    config = json.load(open('data/config.json'))
    model = MF(config)
    if gpu:
        state_dict = load(path)
        model.load_state_dict(state_dict)
        return model
    else:
        state_dict = load(path, map_location=device('cpu'))
        model.load_state_dict(state_dict)
        return model


def rec_user(model, user, n=10):
    item_from_int = pickle.load(open('data/item_mappings', 'rb'))['item_from_int']
    negs = negatives(user)
    scores = []
    for neg in negs:
        user, neg = torch.as_tensor(user).view(-1), torch.as_tensor(neg).view(-1)
        scores.append((item_from_int[neg.item()], sigmoid(model(user, neg)).item()))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:n]


def item_roc_and_auc(model, item):
    dataset = pickle.load(open('data/dataset', 'rb'))
    users = set(dataset.userId)
    interactions = set(dataset[dataset.itemId == item].userId)
    scores = []
    for user in users:
        user, item = torch.as_tensor(user).view(-1), torch.as_tensor(item).view(-1)
        score = sigmoid(model(user, item)).item()
        if user.item() in interactions:
            scores.append((1, score))
        else:
            scores.append((0, score))
    targets, preds = zip(*scores)
    fpr, tpr, _ = roc_curve(targets, preds)
    roc_auc = auc(fpr, tpr)
    # plot
    plt.figure(figsize=(8, 6))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'lightsalmon', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def user_roc_and_auc(model, user):
    dataset = pickle.load(open('data/dataset', 'rb'))
    items = set(dataset.itemId)
    interactions = set(dataset[dataset.userId == user].itemId)
    scores = []
    for item in items:
        user, item = torch.as_tensor(user).view(-1), torch.as_tensor(item).view(-1)
        score = sigmoid(model(user, item)).item()
        if item.item() in interactions:
            scores.append((1, score))
        else:
            scores.append((0, score))
    targets, preds = zip(*scores)
    fpr, tpr, _ = roc_curve(targets, preds)
    roc_auc = auc(fpr, tpr)
    # plot
    plt.figure(figsize=(8, 6))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'lightsalmon', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def negatives(user):
    dataset = pickle.load(open('data/dataset', 'rb'))
    item_pool = set(dataset.itemId)
    negs = item_pool - set(dataset[dataset.userId == user].itemId)
    return negs


def open_links(recs):
    chrome_options = Options()
    chrome_options.add_experimental_option("detach", True)
    driver = webdriver.Chrome('driver/chromedriver', options=chrome_options)
    for i, id in enumerate(recs, start=1):
        link = f'http://imobiliare.ro/anunt/{id[0]}'
        driver.get(link)
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[i])
