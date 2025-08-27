import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import *
from tensorflow.keras.models import load_model
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras.utils import to_categorical
import torch
from transformers import BertTokenizer, BertModel
import ast
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Dropout, LayerNormalization, TimeDistributed, Attention, Concatenate
from tensorflow.keras.models import Model
from transformers import AutoModel,TFAutoModel,AutoTokenizer







################################## data

questions = {
    "Shrestha store": [
        [('What', 1), ('are', 2), ('the', 2), ('prices', 5), ('for', 0), ('vegetables', 5), ('at', 0), ('Shrestha', 3), ('store', 4)],
        [('Do', 1), ('you', 2), ('have', 2), ('fresh', 6), ('fruits', 5), ('at', 0), ('Shrestha', 3), ('store', 4)],
        [('Where', 1), ('is', 2), ('Shrestha', 3), ('store', 4), ('located', 7)],
        [('Can', 1), ('I', 0), ('buy', 7), ('milk', 5), ('at', 0), ('Shrestha', 3), ('store', 4)],
        [('Do', 1), ('you', 2), ('sell', 7), ('organic', 6), ('produce', 5), ('at', 0), ('Shrestha', 3), ('store', 4)],
        [('What', 1), ('are', 2), ('the', 2), ('operating', 7), ('hours', 0), ('for', 0), ('Shrestha', 3), ('store', 4)],
        [('Do', 1), ('you', 2), ('offer', 7), ('delivery', 5), ('services', 0), ('at', 0), ('Shrestha', 3), ('store', 4)],
        [('Are', 1), ('there', 0), ('any', 0), ('discounts', 5), ('on', 0), ('groceries', 5), ('at', 0), ('Shrestha', 3), ('store', 4)],
        [('What', 1), ('types', 0), ('of', 0), ('meat', 5), ('are', 2), ('available', 7), ('at', 0), ('Shrestha', 3), ('store', 4)],
        [('Do', 1), ('you', 2), ('have', 2), ('frozen', 6), ('foods', 5), ('at', 0), ('Shrestha', 3), ('store', 4)],
        [('What', 1), ('are', 2), ('the', 2), ('vegetable', 5), ('varieties', 5), ('at', 0), ('Shrestha', 3), ('store', 4)],
        [('Is', 1), ('there', 0), ('a', 0), ('loyalty', 6), ('program', 0), ('at', 0), ('Shrestha', 3), ('store', 4)],
        [('Can', 1), ('I', 0), ('buy', 7), ('spices', 5), ('at', 0), ('Shrestha', 3), ('store', 4)],
        [('Do', 1), ('you', 2), ('sell', 7), ('canned', 6), ('goods', 5), ('at', 0), ('Shrestha', 3), ('store', 4)],
        [('What', 1), ('are', 2), ('the', 2), ('opening', 7), ('hours', 0), ('of', 0), ('Shrestha', 3), ('store', 4)],
        [('Do', 1), ('you', 2), ('offer', 7), ('home', 6), ('delivery', 5), ('from', 0), ('Shrestha', 3), ('store', 4)],
        [('What', 1), ('brands', 5), ('of', 0), ('pasta', 5), ('do', 1), ('you', 2), ('have', 2), ('at', 0), ('Shrestha', 3), ('store', 4)],
        [('Is', 1), ('there', 0), ('a', 0), ('discount', 5), ('on', 0), ('cereals', 5), ('this', 0), ('week', 6), ('at', 0), ('Shrestha', 3), ('store', 4)],
        [('What', 1), ('are', 2), ('the', 2), ('payment', 0), ('methods', 5), ('accepted', 7), ('at', 0), ('Shrestha', 3), ('store', 4)],
        [('Can', 1), ('I', 0), ('get', 7), ('bulk', 6), ('discounts', 5), ('on', 0), ('items', 5), ('at', 0), ('Shrestha', 3), ('store', 4)],
        [('Do', 1), ('you', 2), ('stock', 7), ('imported', 6), ('goods', 5), ('at', 0), ('Shrestha', 3), ('store', 4)],
    ],
    "Manakamana dairy": [
        [('Where', 1), ('is', 2), ('Manakamana', 3), ('dairy', 4), ('located', 5)],
        [('What', 1), ('types', 2), ('of', 0), ('milk', 5), ('do', 1), ('you', 2), ('sell', 7), ('at', 0), ('Manakamana', 3), ('dairy', 4)],
        [('Do', 1), ('you', 2), ('have', 2), ('fresh', 6), ('milk', 5), ('at', 0), ('Manakamana', 3), ('dairy', 4)],
        [('What', 1), ('are', 2), ('the', 2), ('prices', 5), ('for', 0), ('milk', 5), ('at', 0), ('Manakamana', 3), ('dairy', 4)],
        [('Do', 1), ('you', 2), ('sell', 7), ('yogurt', 5), ('at', 0), ('Manakamana', 3), ('dairy', 4)],
        [('Can', 1), ('I', 0), ('order', 7), ('milk', 5), ('online', 0), ('from', 0), ('Manakamana', 3), ('dairy', 4)],
        [('What', 1), ('is', 2), ('the', 2), ('delivery', 5), ('fee', 5), ('for', 0), ('Manakamana', 3), ('dairy', 4)],
        [('Do', 1), ('you', 2), ('offer', 7), ('home', 6), ('delivery', 5), ('at', 0), ('Manakamana', 3), ('dairy', 4)],
        [('Are', 1), ('there', 0), ('any', 0), ('discounts', 5), ('on', 0), ('dairy', 5), ('products', 0), ('at', 0), ('Manakamana', 3), ('dairy', 4)],
        [('What', 1), ('is', 2), ('the', 2), ('best', 0), ('selling', 6), ('item', 5), ('at', 0), ('Manakamana', 3), ('dairy', 4)],
        [('Do', 1), ('you', 2), ('sell', 7), ('cheese', 5), ('at', 0), ('Manakamana', 3), ('dairy', 4)],
        [('Where', 1), ('can', 1), ('I', 0), ('buy', 7), ('fresh', 6), ('butter', 5), ('from', 0), ('Manakamana', 3), ('dairy', 4)],
        [('Is', 1), ('there', 0), ('a', 0), ('loyalty', 6), ('program', 0), ('at', 0), ('Manakamana', 3), ('dairy', 4)],
        [('What', 1), ('are', 2), ('the', 2), ('different', 0), ('brands', 5), ('of', 0), ('milk', 5), ('available', 7), ('at', 0), ('Manakamana', 3), ('dairy', 4)],
        [('Do', 1), ('you', 2), ('sell', 7), ('puddings', 5), ('at', 0), ('Manakamana', 3), ('dairy', 4)],
        [('Can', 1), ('I', 0), ('order', 7), ('milk', 5), ('at', 0), ('Manakamana', 3), ('dairy', 4), ('through', 0), ('phone', 6)],
        [('What', 1), ('is', 2), ('the', 2), ('quality', 5), ('of', 0), ('milk', 5), ('at', 0), ('Manakamana', 3), ('dairy', 4)],
        [('Do', 1), ('you', 2), ('sell', 7), ('dairy', 5), ('based', 0), ('ice', 5), ('cream', 5), ('at', 0), ('Manakamana', 3), ('dairy', 4)],
        [('What', 1), ('is', 2), ('the', 2), ('delivery', 5), ('time', 5), ('for', 0), ('Manakamana', 3), ('dairy', 4)],
    ],
    "Nisum cold store": [
        [('Where', 1), ('is', 2), ('Nisum', 3), ('cold', 4), ('store', 5), ('located', 7)],
        [('Do', 1), ('you', 2), ('have', 2), ('fresh', 6), ('meat', 5), ('at', 0), ('Nisum', 3), ('cold', 4), ('store', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('prices', 5), ('for', 0), ('fish', 5), ('at', 0), ('Nisum', 3), ('cold', 4), ('store', 5)],
        [('Can', 1), ('I', 0), ('buy', 7), ('frozen', 6), ('foods', 5), ('at', 0), ('Nisum', 3), ('cold', 4), ('store', 5)],
        [('Do', 1), ('you', 2), ('sell', 7), ('ice', 5), ('cream', 5), ('at', 0), ('Nisum', 3), ('cold', 4), ('store', 5)],
        [('What', 1), ('is', 2), ('the', 2), ('quality', 5), ('of', 0), ('meat', 5), ('at', 0), ('Nisum', 3), ('cold', 4), ('store', 5)],
        [('Do', 1), ('you', 2), ('offer', 7), ('delivery', 5), ('services', 0), ('at', 0), ('Nisum', 3), ('cold', 4), ('store', 5)],
        [('Are', 1), ('there', 0), ('any', 0), ('discounts', 5), ('on', 0), ('frozen', 6), ('products', 0), ('at', 0), ('Nisum', 3), ('cold', 4), ('store', 5)],
        [('What', 1), ('types', 0), ('of', 0), ('frozen', 6), ('foods', 5), ('do', 1), ('you', 2), ('sell', 7), ('at', 0), ('Nisum', 3), ('cold', 4), ('store', 5)],
        [('Where', 1), ('can', 1), ('I', 0), ('get', 7), ('frozen', 6), ('vegetables', 5), ('at', 0), ('Nisum', 3), ('cold', 4), ('store', 5)],
        [('Do', 1), ('you', 2), ('sell', 7), ('frozen', 6), ('fruit', 5), ('at', 0), ('Nisum', 3), ('cold', 4), ('store', 5)],
        [('Can', 1), ('I', 0), ('buy', 7), ('frozen', 6), ('pizza', 5), ('at', 0), ('Nisum', 3), ('cold', 4), ('store', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('special', 0), ('offers', 5), ('at', 0), ('Nisum', 3), ('cold', 4), ('store', 5)],
        [('Do', 1), ('you', 2), ('stock', 7), ('dairy', 5), ('products', 0), ('at', 0), ('Nisum', 3), ('cold', 4), ('store', 5)],
        [('What', 1), ('is', 2), ('the', 2), ('shelf', 5), ('life', 5), ('of', 0), ('frozen', 6), ('meat', 5), ('at', 0), ('Nisum', 3), ('cold', 4), ('store', 5)],
        [('Do', 1), ('you', 2), ('offer', 7), ('bulk', 6), ('discounts', 5), ('on', 0), ('frozen', 6), ('goods', 5), ('at', 0), ('Nisum', 3), ('cold', 4), ('store', 5)],
        [('What', 1), ('is', 2), ('the', 2), ('price', 5), ('for', 0), ('frozen', 6), ('fish', 5), ('at', 0), ('Nisum', 3), ('cold', 4), ('store', 5)],
        [('Do', 1), ('you', 2), ('sell', 7), ('ready', 6), ('to', 0), ('cook', 7), ('meals', 5), ('at', 0), ('Nisum', 3), ('cold', 4), ('store', 5)],
    ]
}
questions.update({
    "The station cafe": [
        [('Where', 1), ('is', 2), ('The', 3), ('station', 4), ('cafe', 5), ('located', 7)],
        [('What', 1), ('are', 2), ('the', 2), ('opening', 7), ('hours', 0), ('for', 0), ('The', 3), ('station', 4), ('cafe', 5)],
        [('Do', 1), ('you', 2), ('serve', 7), ('breakfast', 5), ('at', 0), ('The', 3), ('station', 4), ('cafe', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('most', 0), ('popular', 0), ('dishes', 5), ('at', 0), ('The', 3), ('station', 4), ('cafe', 5)],
        [('Is', 1), ('there', 0), ('a', 0), ('vegetarian', 6), ('menu', 5), ('at', 0), ('The', 3), ('station', 4), ('cafe', 5)],
        [('Do', 1), ('you', 2), ('have', 2), ('coffee', 5), ('and', 0), ('tea', 5), ('at', 0), ('The', 3), ('station', 4), ('cafe', 5)],
        [('Can', 1), ('I', 0), ('order', 7), ('food', 5), ('online', 0), ('from', 0), ('The', 3), ('station', 4), ('cafe', 5)],
        [('Do', 1), ('you', 2), ('offer', 7), ('delivery', 5), ('services', 0), ('from', 0), ('The', 3), ('station', 4), ('cafe', 5)],
        [('What', 1), ('types', 0), ('of', 0), ('coffee', 5), ('do', 1), ('you', 2), ('serve', 7), ('at', 0), ('The', 3), ('station', 4), ('cafe', 5)],
        [('Is', 1), ('there', 0), ('a', 0), ('loyalty', 6), ('program', 0), ('at', 0), ('The', 3), ('station', 4), ('cafe', 5)],
        [('What', 1), ('is', 2), ('the', 2), ('price', 5), ('for', 0), ('a', 0), ('coffee', 5), ('at', 0), ('The', 3), ('station', 4), ('cafe', 5)],
        [('Do', 1), ('you', 2), ('offer', 7), ('discounts', 5), ('on', 0), ('drinks', 5), ('at', 0), ('The', 3), ('station', 4), ('cafe', 5)],
        [('Where', 1), ('can', 1), ('I', 0), ('get', 7), ('vegetarian', 6), ('snacks', 5), ('at', 0), ('The', 3), ('station', 4), ('cafe', 5)],
        [('Is', 1), ('there', 0), ('an', 0), ('outdoor', 6), ('seating', 5), ('area', 5), ('at', 0), ('The', 3), ('station', 4), ('cafe', 5)],
        [('Do', 1), ('you', 2), ('have', 2), ('free', 6), ('Wi-Fi', 5), ('at', 0), ('The', 3), ('station', 4), ('cafe', 5)],
        [('Can', 1), ('I', 0), ('reserve', 7), ('a', 0), ('table', 5), ('at', 0), ('The', 3), ('station', 4), ('cafe', 5)],
        [('Do', 1), ('you', 2), ('serve', 7), ('desserts', 5), ('at', 0), ('The', 3), ('station', 4), ('cafe', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('payment', 0), ('options', 5), ('available', 7), ('at', 0), ('The', 3), ('station', 4), ('cafe', 5)],
        [('Is', 1), ('there', 0), ('a', 0), ('live', 6), ('music', 5), ('event', 7), ('at', 0), ('The', 3), ('station', 4), ('cafe', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('special', 0), ('offers', 5), ('this', 0), ('month', 6), ('at', 0), ('The', 3), ('station', 4), ('cafe', 5)],
    ],
    "Metro cafe": [
        [('Where', 1), ('is', 2), ('Metro', 3), ('cafe', 4), ('located', 7)],
        [('What', 1), ('are', 2), ('the', 2), ('menu', 5), ('options', 5), ('at', 0), ('Metro', 3), ('cafe', 4)],
        [('Do', 1), ('you', 2), ('serve', 7), ('vegetarian', 6), ('dishes', 5), ('at', 0), ('Metro', 3), ('cafe', 4)],
        [('Can', 1), ('I', 0), ('order', 7), ('food', 5), ('from', 0), ('Metro', 3), ('cafe', 4)],
        [('Do', 1), ('you', 2), ('have', 2), ('coffee', 5), ('at', 0), ('Metro', 3), ('cafe', 4)],
        [('What', 1), ('is', 2), ('the', 2), ('price', 5), ('for', 0), ('lunch', 5), ('at', 0), ('Metro', 3), ('cafe', 4)],
        [('Do', 1), ('you', 2), ('offer', 7), ('takeout', 5), ('services', 0), ('at', 0), ('Metro', 3), ('cafe', 4)],
        [('What', 1), ('are', 2), ('the', 2), ('opening', 7), ('hours', 0), ('for', 0), ('Metro', 3), ('cafe', 4)],
        [('Where', 1), ('can', 1), ('I', 0), ('get', 7), ('a', 0), ('good', 0), ('coffee', 5), ('at', 0), ('Metro', 3), ('cafe', 4)],
        [('Do', 1), ('you', 2), ('serve', 7), ('snacks', 5), ('at', 0), ('Metro', 3), ('cafe', 4)],
        [('Is', 1), ('there', 0), ('a', 0), ('discount', 5), ('on', 0), ('coffee', 5), ('this', 0), ('week', 6), ('at', 0), ('Metro', 3), ('cafe', 4)],
        [('What', 1), ('are', 2), ('the', 2), ('available', 7), ('beverages', 5), ('at', 0), ('Metro', 3), ('cafe', 4)],
        [('Can', 1), ('I', 0), ('book', 7), ('a', 0), ('table', 5), ('at', 0), ('Metro', 3), ('cafe', 4)],
        [('Do', 1), ('you', 2), ('offer', 7), ('wifi', 6), ('at', 0), ('Metro', 3), ('cafe', 4)],
        [('Are', 1), ('there', 0), ('any', 0), ('vegetarian', 6), ('options', 5), ('on', 0), ('the', 2), ('menu', 5), ('at', 0), ('Metro', 3), ('cafe', 4)],
        [('What', 1), ('is', 2), ('the', 2), ('special', 0), ('of', 0), ('the', 2), ('day', 6), ('at', 0), ('Metro', 3), ('cafe', 4)],
    ],
    # More cafes, hotels, and businesses can be added following a similar pattern
})
questions.update({
    "Best food cafe": [
        [('Where', 1), ('is', 2), ('Best', 3), ('food', 4), ('cafe', 5), ('located', 7)],
        [('What', 1), ('is', 2), ('the', 2), ('menu', 5), ('at', 0), ('Best', 3), ('food', 4), ('cafe', 5)],
        [('Do', 1), ('you', 2), ('serve', 7), ('vegetarian', 6), ('dishes', 5), ('at', 0), ('Best', 3), ('food', 4), ('cafe', 5)],
        [('Can', 1), ('I', 0), ('order', 7), ('takeout', 5), ('from', 0), ('Best', 3), ('food', 4), ('cafe', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('prices', 5), ('for', 0), ('coffee', 5), ('at', 0), ('Best', 3), ('food', 4), ('cafe', 5)],
        [('Do', 1), ('you', 2), ('have', 2), ('vegan', 6), ('options', 5), ('at', 0), ('Best', 3), ('food', 4), ('cafe', 5)],
        [('Where', 1), ('can', 1), ('I', 0), ('get', 7), ('a', 0), ('good', 0), ('burger', 5), ('at', 0), ('Best', 3), ('food', 4), ('cafe', 5)],
        [('What', 1), ('is', 2), ('the', 2), ('special', 0), ('dish', 5), ('of', 0), ('the', 2), ('day', 6), ('at', 0), ('Best', 3), ('food', 4), ('cafe', 5)],
        [('Is', 1), ('there', 0), ('a', 0), ('discount', 5), ('on', 0), ('drinks', 5), ('at', 0), ('Best', 3), ('food', 4), ('cafe', 5)],
        [('Do', 1), ('you', 2), ('offer', 7), ('delivery', 5), ('service', 0), ('from', 0), ('Best', 3), ('food', 4), ('cafe', 5)],
        [('Do', 1), ('you', 2), ('have', 2), ('a', 0), ('lunch', 5), ('menu', 5), ('at', 0), ('Best', 3), ('food', 4), ('cafe', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('opening', 7), ('hours', 0), ('of', 0), ('Best', 3), ('food', 4), ('cafe', 5)],
        [('Can', 1), ('I', 0), ('reserve', 7), ('a', 0), ('table', 5), ('at', 0), ('Best', 3), ('food', 4), ('cafe', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('payment', 0), ('methods', 5), ('accepted', 7), ('at', 0), ('Best', 3), ('food', 4), ('cafe', 5)],
        [('Do', 1), ('you', 2), ('serve', 7), ('desserts', 5), ('at', 0), ('Best', 3), ('food', 4), ('cafe', 5)],
        [('Is', 1), ('there', 0), ('a', 0), ('wifi', 6), ('at', 0), ('Best', 3), ('food', 4), ('cafe', 5)],
        [('Do', 1), ('you', 2), ('have', 2), ('a', 0), ('children', 6), ('play', 5), ('area', 5), ('at', 0), ('Best', 3), ('food', 4), ('cafe', 5)],
        [('Where', 1), ('can', 1), ('I', 0), ('find', 7), ('your', 0), ('menu', 5), ('online', 0), ('for', 0), ('Best', 3), ('food', 4), ('cafe', 5)],
        [('What', 1), ('is', 2), ('the', 2), ('most', 0), ('popular', 0), ('dish', 5), ('at', 0), ('Best', 3), ('food', 4), ('cafe', 5)],
    ],
    "The brew house": [
        [('Where', 1), ('is', 2), ('The', 3), ('brew', 4), ('house', 5), ('located', 7)],
        [('What', 1), ('is', 2), ('on', 0), ('the', 2), ('menu', 5), ('at', 0), ('The', 3), ('brew', 4), ('house', 5)],
        [('Do', 1), ('you', 2), ('have', 2), ('live', 6), ('music', 5), ('at', 0), ('The', 3), ('brew', 4), ('house', 5)],
        [('Is', 1), ('there', 0), ('a', 0), ('beer', 5), ('special', 0), ('this', 0), ('week', 6), ('at', 0), ('The', 3), ('brew', 4), ('house', 5)],
        [('Can', 1), ('I', 0), ('order', 7), ('alcohol', 5), ('from', 0), ('The', 3), ('brew', 4), ('house', 5)],
        [('What', 1), ('is', 2), ('the', 2), ('price', 5), ('for', 0), ('a', 0), ('beer', 5), ('at', 0), ('The', 3), ('brew', 4), ('house', 5)],
        [('Do', 1), ('you', 2), ('have', 2), ('a', 0), ('happy', 6), ('hour', 5), ('at', 0), ('The', 3), ('brew', 4), ('house', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('special', 0), ('cocktails', 5), ('served', 7), ('at', 0), ('The', 3), ('brew', 4), ('house', 5)],
        [('Can', 1), ('I', 0), ('book', 7), ('a', 0), ('table', 5), ('for', 0), ('drinks', 5), ('at', 0), ('The', 3), ('brew', 4), ('house', 5)],
        [('Do', 1), ('you', 2), ('offer', 7), ('snacks', 5), ('at', 0), ('The', 3), ('brew', 4), ('house', 5)],
        [('Wh', 1), ('you', 2), ('have', 2), ('online', 0), ('ordering', 5), ('available', 7), ('at', 0), ('Hd', 3), ('fast', 4), ('food', 5)],
        [('What', 1), ('time', 7), ('do', 1), ('you', 2), ('close', 7), ('at', 0), ('The', 3), ('brew', 4), ('house', 5)],
        [('Is', 1), ('there', 0), ('a', 0), ('bouncer', 6), ('at', 0), ('The', 3), ('brew', 4), ('house', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('payment', 0), ('methods', 5), ('accepted', 7), ('at', 0), ('The', 3), ('brew', 4), ('house', 5)],
        [('Do', 1), ('you', 2), ('have', 2), ('outdoor', 6), ('seating', 5), ('at', 0), ('The', 3), ('brew', 4), ('house', 5)],
        [('Can', 1), ('I', 0), ('take', 7), ('away', 0), ('alcohol', 5), ('from', 0), ('The', 3), ('brew', 4), ('house', 5)],
        [('Do', 1), ('you', 2), ('offer', 7), ('delivery', 5), ('services', 0), ('from', 0), ('The', 3), ('brew', 4), ('house', 5)],
        [('Is', 1), ('there', 0), ('a', 0), ('dress', 0), ('code', 5), ('at', 0), ('The', 3), ('brew', 4), ('house', 5)],
        [('Do', 1), ('you', 2), ('serve', 7), ('non-alcoholic', 6), ('beverages', 5), ('at', 0), ('The', 3), ('brew', 4), ('house', 5)],
        [('Where', 1), ('can', 1), ('I', 0), ('find', 7), ('your', 0), ('menu', 5), ('online', 0), ('for', 0), ('The', 3), ('brew', 4), ('house', 5)],
    ],
    "Monsoon hotel": [
        [('Where', 1), ('is', 2), ('Monsoon', 3), ('hotel', 4), ('located', 7)],
        [('What', 1), ('is', 2), ('the', 2), ('price', 5), ('for', 0), ('a', 0), ('night', 5), ('at', 0), ('Monsoon', 3), ('hotel', 4)],
        [('Do', 1), ('you', 2), ('offer', 7), ('breakfast', 5), ('included', 7), ('at', 0), ('Monsoon', 3), ('hotel', 4)],
        [('Can', 1), ('I', 0), ('book', 7), ('a', 0), ('room', 5), ('online', 0), ('at', 0), ('Monsoon', 3), ('hotel', 4)],
        [('What', 1), ('are', 2), ('the', 2), ('amenities', 5), ('available', 7), ('at', 0), ('Monsoon', 3), ('hotel', 4)],
        [('Is', 1), ('there', 0), ('a', 0), ('pool', 5), ('at', 0), ('Monsoon', 3), ('hotel', 4)],
        [('What', 1), ('are', 2), ('the', 2), ('check-in', 7), ('and', 0), ('check-out', 7), ('times', 0), ('at', 0), ('Monsoon', 3), ('hotel', 4)],
        [('Can', 1), ('I', 0), ('get', 7), ('a', 0), ('discount', 5), ('if', 0), ('I', 0), ('book', 7), ('early', 6), ('at', 0), ('Monsoon', 3), ('hotel', 4)],
        [('What', 1), ('is', 2), ('the', 2), ('address', 5), ('of', 0), ('Monsoon', 3), ('hotel', 4)],
        [('Do', 1), ('you', 2), ('offer', 7), ('free', 6), ('Wi-Fi', 5), ('at', 0), ('Monsoon', 3), ('hotel', 4)],
        [('Is', 1), ('there', 0), ('room', 5), ('service', 0), ('available', 7), ('at', 0), ('Monsoon', 3), ('hotel', 4)],
        [('Do', 1), ('you', 2), ('allow', 7), ('pets', 5), ('at', 0), ('Monsoon', 3), ('hotel', 4)],
        [('What', 1), ('are', 2), ('the', 2), ('price', 5), ('range', 0), ('for', 0), ('rooms', 5), ('at', 0), ('Monsoon', 3), ('hotel', 4)],
        [('Do', 1), ('you', 2), ('have', 2), ('conference', 6), ('rooms', 5), ('at', 0), ('Monsoon', 3), ('hotel', 4)],
        [('Can', 1), ('I', 0), ('check', 7), ('in', 0), ('earlier', 6), ('at', 0), ('Monsoon', 3), ('hotel', 4)],
        [('What', 1), ('type', 0), ('of', 0), ('rooms', 5), ('are', 2), ('available', 7), ('at', 0), ('Monsoon', 3), ('hotel', 4)],
    ],
    # Add more businesses following this pattern
})
questions.update({
    "Laxmi fast food restaurant": [
        [('Where', 1), ('is', 2), ('Laxmi', 3), ('fast', 4), ('food', 5), ('restaurant', 6), ('located', 7)],
        [('What', 1), ('are', 2), ('the', 2), ('menu', 5), ('items', 6), ('at', 0), ('Laxmi', 3), ('fast', 4), ('food', 5), ('restaurant', 6)],
        [('Do', 1), ('you', 2), ('offer', 7), ('delivery', 5), ('service', 0), ('from', 0), ('Laxmi', 3), ('fast', 4), ('food', 5), ('restaurant', 6)],
        [('What', 1), ('is', 2), ('the', 2), ('special', 0), ('dish', 5), ('at', 0), ('Laxmi', 3), ('fast', 4), ('food', 5), ('restaurant', 6)],
        [('Where', 1), ('can', 1), ('I', 0), ('find', 7), ('your', 0), ('menu', 5), ('online', 0), ('for', 0), ('Laxmi', 3), ('fast', 4), ('food', 5), ('restaurant', 6)],
        [('Do', 1), ('you', 2), ('have', 2), ('a', 0), ('kids', 6), ('menu', 5), ('at', 0), ('Laxmi', 3), ('fast', 4), ('food', 5), ('restaurant', 6)],
        [('Can', 1), ('I', 0), ('order', 7), ('takeout', 6), ('from', 0), ('Laxmi', 3), ('fast', 4), ('food', 5), ('restaurant', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('prices', 5), ('of', 0), ('burgers', 5), ('at', 0), ('Laxmi', 3), ('fast', 4), ('food', 5), ('restaurant', 6)],
        [('Do', 1), ('you', 2), ('serve', 7), ('vegetarian', 6), ('dishes', 5), ('at', 0), ('Laxmi', 3), ('fast', 4), ('food', 5), ('restaurant', 6)],
        [('What', 1), ('is', 2), ('the', 2), ('timing', 7), ('for', 0), ('lunch', 5), ('at', 0), ('Laxmi', 3), ('fast', 4), ('food', 5), ('restaurant', 6)],
        [('Is', 1), ('there', 0), ('wifi', 6), ('available', 7), ('at', 0), ('Laxmi', 3), ('fast', 4), ('food', 5), ('restaurant', 6)],
        [('Do', 1), ('you', 2), ('offer', 7), ('online', 0), ('ordering', 5), ('service', 0), ('at', 0), ('Laxmi', 3), ('fast', 4), ('food', 5), ('restaurant', 6)],
        [('What', 1), ('is', 2), ('the', 2), ('price', 5), ('for', 0), ('a', 0), ('plate', 5), ('of', 0), ('dal', 5), ('bhat', 5), ('at', 0), ('Laxmi', 3), ('fast', 4), ('food', 5), ('restaurant', 6)],
        [('Can', 1), ('I', 0), ('reserve', 7), ('a', 0), ('table', 5), ('at', 0), ('Laxmi', 3), ('fast', 4), ('food', 5), ('restaurant', 6)],
        [('Do', 1), ('you', 2), ('have', 2), ('free', 6), ('refills', 5), ('for', 0), ('drinks', 5), ('at', 0), ('Laxmi', 3), ('fast', 4), ('food', 5), ('restaurant', 6)],
        [('Is', 1), ('there', 0), ('a', 0), ('lunch', 6), ('combo', 5), ('at', 0), ('Laxmi', 3), ('fast', 4), ('food', 5), ('restaurant', 6)],
        [('Do', 1), ('you', 2), ('serve', 7), ('desserts', 5), ('at', 0), ('Laxmi', 3), ('fast', 4), ('food', 5), ('restaurant', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('best', 0), ('food', 5), ('options', 6), ('at', 0), ('Laxmi', 3), ('fast', 4), ('food', 5), ('restaurant', 6)],
        [('Can', 1), ('I', 0), ('get', 7), ('a', 0), ('discount', 5), ('if', 0), ('I', 0), ('bring', 7), ('a', 0), ('friend', 6), ('to', 0), ('Laxmi', 3), ('fast', 4), ('food', 5), ('restaurant', 6)],
    ],
    "Hd fast food": [
        [('Where', 1), ('is', 2), ('Hd', 3), ('fast', 4), ('food', 5), ('located', 7)],
        [('What', 1), ('are', 2), ('the', 2), ('specials', 5), ('at', 0), ('Hd', 3), ('fast', 4), ('food', 5)],
        [('Doat', 1), ('is', 2), ('the', 2), ('price', 5), ('for', 0), ('a', 0), ('burger', 5), ('at', 0), ('Hd', 3), ('fast', 4), ('food', 5)],
        [('Can', 1), ('I', 0), ('order', 7), ('delivery', 5), ('from', 0), ('Hd', 3), ('fast', 4), ('food', 5)],
        [('Do', 1), ('you', 2), ('offer', 7), ('vegetarian', 6), ('options', 5), ('at', 0), ('Hd', 3), ('fast', 4), ('food', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('working', 7), ('hours', 0), ('of', 0), ('Hd', 3), ('fast', 4), ('food', 5)],
        [('Where', 1), ('can', 1), ('I', 0), ('get', 7), ('your', 0), ('menu', 5), ('online', 0), ('for', 0), ('Hd', 3), ('fast', 4), ('food', 5)],
        [('What', 1), ('is', 2), ('the', 2), ('best', 0), ('dish', 5), ('at', 0), ('Hd', 3), ('fast', 4), ('food', 5)],
        [('Do', 1), ('you', 2), ('serve', 7), ('alcohol', 6), ('at', 0), ('Hd', 3), ('fast', 4), ('food', 5)],
        [('Is', 1), ('there', 0), ('a', 0), ('discount', 5), ('for', 0), ('students', 6), ('at', 0), ('Hd', 3), ('fast', 4), ('food', 5)],
        [('Can', 1), ('I', 0), ('get', 7), ('takeout', 6), ('from', 0), ('Hd', 3), ('fast', 4), ('food', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('popular', 0), ('items', 5), ('at', 0), ('Hd', 3), ('fast', 4), ('food', 5)],
        [('Do', 1), ('you', 2), ('have', 2), ('a', 0), ('kids', 6), ('menu', 5), ('at', 0), ('Hd', 3), ('fast', 4), ('food', 5)],
        [('Where', 1), ('is', 2), ('Hd', 3), ('fast', 4), ('food', 5), ('restaurant', 6), ('located', 7)],
        [('Do', 1), ('you', 2), ('serve', 7), ('vegan', 6), ('dishes', 5), ('at', 0), ('Hd', 3), ('fast', 4), ('food', 5)],
        [('Can', 1), ('I', 0), ('reserve', 7), ('a', 0), ('table', 5), ('at', 0), ('Hd', 3), ('fast', 4), ('food', 5)],
        [('Is', 1), ('there', 0), ('free', 6), ('wifi', 5), ('available', 7), ('at', 0), ('Hd', 3), ('fast', 4), ('food', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('popular', 0), ('dishes', 5), ('at', 0), ('Hd', 3), ('fast', 4), ('food', 5)],
        [('Do', 1), ('you', 2), ('have', 2), ('free', 6), ('refills', 5), ('for', 0), ('soda', 5), ('at', 0), ('Hd', 3), ('fast', 4), ('food', 5)],
    ],
})
questions.update({
    "Janata newari khaja ghar": [
        [('Where', 1), ('is', 2), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6), ('located', 7)],
        [('What', 1), ('are', 2), ('the', 2), ('signature', 0), ('dishes', 5), ('at', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6)],
        [('Do', 1), ('you', 2), ('offer', 7), ('delivery', 5), ('service', 0), ('from', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6)],
        [('What', 1), ('is', 2), ('the', 2), ('price', 5), ('for', 0), ('newari', 4), ('thali', 5), ('at', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6)],
        [('Can', 1), ('I', 0), ('order', 7), ('takeout', 6), ('from', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('timings', 7), ('of', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6)],
        [('Do', 1), ('you', 2), ('have', 2), ('vegetarian', 6), ('dishes', 5), ('at', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6)],
        [('Is', 1), ('there', 0), ('wifi', 6), ('available', 7), ('at', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6)],
        [('Where', 1), ('can', 1), ('I', 0), ('find', 7), ('your', 0), ('menu', 5), ('online', 0), ('for', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6)],
        [('Do', 1), ('you', 2), ('serve', 7), ('alcohol', 6), ('at', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6)],
        [('What', 1), ('is', 2), ('the', 2), ('special', 0), ('dish', 5), ('at', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6)],
        [('Can', 1), ('I', 0), ('get', 7), ('takeaway', 6), ('from', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6)],
        [('Do', 1), ('you', 2), ('offer', 7), ('online', 0), ('ordering', 5), ('for', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('best', 0), ('dishes', 5), ('at', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6)],
        [('Is', 1), ('there', 0), ('a', 0), ('discount', 5), ('for', 0), ('students', 6), ('at', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6)],
        [('Do', 1), ('you', 2), ('have', 2), ('a', 0), ('kids', 6), ('menu', 5), ('at', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6)],
        [('Can', 1), ('I', 0), ('reserve', 7), ('a', 0), ('table', 5), ('at', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6)],
        [('What', 1), ('is', 2), ('the', 2), ('timing', 7), ('for', 0), ('lunch', 5), ('at', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6)],
        [('Where', 1), ('is', 2), ('the', 2), ('nearest', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6), ('located', 7)],
        [('Do', 1), ('you', 2), ('offer', 7), ('breakfast', 6), ('at', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6)],
        [('Can', 1), ('I', 0), ('get', 7), ('food', 5), ('delivered', 7), ('from', 0), ('Janata', 3), ('newari', 4), ('khaja', 5), ('ghar', 6)],
    ],
    "Tridev khaja ghar": [
        [('Where', 1), ('is', 2), ('Tridev', 3), ('khaja', 4), ('ghar', 5), ('located', 7)],
        [('What', 1), ('are', 2), ('the', 2), ('specials', 5), ('at', 0), ('Tridev', 3), ('khaja', 4), ('ghar', 5)],
        [('Do', 1), ('you', 2), ('offer', 7), ('delivery', 5), ('service', 0), ('from', 0), ('Tridev', 3), ('khaja', 4), ('ghar', 5)],
        [('What', 1), ('is', 2), ('the', 2), ('price', 5), ('for', 0), ('khaja', 4), ('at', 0), ('Tridev', 3), ('khaja', 4), ('ghar', 5)],
        [('Can', 1), ('I', 0), ('order', 7), ('takeout', 6), ('from', 0), ('Tridev', 3), ('khaja', 4), ('ghar', 5)],
        [('Do', 1), ('you', 2), ('have', 2), ('vegetarian', 6), ('dishes', 5), ('at', 0), ('Tridev', 3), ('khaja', 4), ('ghar', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('working', 7), ('hours', 0), ('of', 0), ('Tridev', 3), ('khaja', 4), ('ghar', 5)],
        [('Where', 1), ('can', 1), ('I', 0), ('get', 7), ('your', 0), ('menu', 5), ('online', 0), ('for', 0), ('Tridev', 3), ('khaja', 4), ('ghar', 5)],
        [('Do', 1), ('you', 2), ('serve', 7), ('alcohol', 6), ('at', 0), ('Tridev', 3), ('khaja', 4), ('ghar', 5)],
        [('What', 1), ('is', 2), ('the', 2), ('best', 0), ('dish', 5), ('at', 0), ('Tridev', 3), ('khaja', 4), ('ghar', 5)],
        [('Can', 1), ('I', 0), ('get', 7), ('takeaway', 6), ('from', 0), ('Tridev', 3), ('khaja', 4), ('ghar', 5)],
        [('Do', 1), ('you', 2), ('offer', 7), ('online', 0), ('ordering', 5), ('for', 0), ('Tridev', 3), ('khaja', 4), ('ghar', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('most', 0), ('popular', 0), ('dishes', 5), ('at', 0), ('Tridev', 3), ('khaja', 4), ('ghar', 5)],
        [('Is', 1), ('there', 0), ('free', 6), ('wifi', 5), ('available', 7), ('at', 0), ('Tridev', 3), ('khaja', 4), ('ghar', 5)],
        [('Can', 1), ('I', 0), ('reserve', 7), ('a', 0), ('table', 5), ('at', 0), ('Tridev', 3), ('khaja', 4), ('ghar', 5)],
        [('Do', 1), ('you', 2), ('have', 2), ('a', 0), ('kids', 6), ('menu', 5), ('at', 0), ('Tridev', 3), ('khaja', 4), ('ghar', 5)],
        [('What', 1), ('is', 2), ('the', 2), ('timing', 7), ('for', 0), ('lunch', 5), ('at', 0), ('Tridev', 3), ('khaja', 4), ('ghar', 5)],
        [('Where', 1), ('is', 2), ('the', 2), ('nearest', 0), ('Tridev', 3), ('khaja', 4), ('ghar', 5), ('located', 7)],
        [('Do', 1), ('you', 2), ('serve', 7), ('desserts', 6), ('at', 0), ('Tridev', 3), ('khaja', 4), ('ghar', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('signature', 0), ('dishes', 5), ('at', 0), ('Tridev', 3), ('khaja', 4), ('ghar', 5)],
    ],
})
questions.update({
    "Ishanvi fast food and cafe": [
        [('Where', 1), ('is', 2), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6), ('located', 7)],
        [('What', 1), ('are', 2), ('the', 2), ('special', 0), ('dishes', 5), ('at', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
        [('Do', 1), ('you', 2), ('offer', 7), ('delivery', 5), ('service', 0), ('from', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
        [('What', 1), ('is', 2), ('the', 2), ('price', 5), ('for', 0), ('burger', 6), ('at', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
        [('Can', 1), ('I', 0), ('order', 7), ('takeout', 6), ('from', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('working', 7), ('hours', 0), ('for', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
        [('Do', 1), ('you', 2), ('have', 2), ('vegetarian', 6), ('options', 5), ('at', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
        [('Where', 1), ('can', 1), ('I', 0), ('find', 7), ('your', 0), ('menu', 5), ('online', 0), ('for', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
        [('Do', 1), ('you', 2), ('serve', 7), ('alcohol', 6), ('at', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
        [('What', 1), ('is', 2), ('the', 2), ('best', 0), ('dish', 5), ('at', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
        [('Can', 1), ('I', 0), ('get', 7), ('takeaway', 6), ('from', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
        [('Do', 1), ('you', 2), ('offer', 7), ('online', 0), ('ordering', 5), ('for', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('most', 0), ('popular', 0), ('dishes', 5), ('at', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
        [('Is', 1), ('there', 0), ('wifi', 6), ('available', 7), ('at', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
        [('Can', 1), ('I', 0), ('reserve', 7), ('a', 0), ('table', 5), ('at', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
        [('Do', 1), ('you', 2), ('have', 2), ('a', 0), ('kids', 6), ('menu', 5), ('at', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
        [('What', 1), ('is', 2), ('the', 2), ('timing', 7), ('for', 0), ('lunch', 5), ('at', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
        [('Where', 1), ('is', 2), ('the', 2), ('nearest', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6), ('located', 7)],
        [('Do', 1), ('you', 2), ('serve', 7), ('desserts', 6), ('at', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('signature', 0), ('dishes', 5), ('at', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
        [('Do', 1), ('you', 2), ('have', 2), ('meal', 6), ('packages', 5), ('at', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
        [('Is', 1), ('there', 0), ('free', 6), ('wifi', 5), ('available', 7), ('at', 0), ('Ishanvi', 3), ('fast', 4), ('food', 5), ('and', 0), ('cafe', 6)],
    ],
    "Jhilko food land": [
        [('Where', 1), ('is', 2), ('Jhilko', 3), ('food', 4), ('land', 5), ('located', 7)],
        [('What', 1), ('are', 2), ('the', 2), ('signature', 0), ('dishes', 5), ('at', 0), ('Jhilko', 3), ('food', 4), ('land', 5)],
        [('Do', 1), ('you', 2), ('offer', 7), ('delivery', 5), ('service', 0), ('from', 0), ('Jhilko', 3), ('food', 4), ('land', 5)],
        [('What', 1), ('is', 2), ('the', 2), ('price', 5), ('for', 0), ('momo', 6), ('at', 0), ('Jhilko', 3), ('food', 4), ('land', 5)],
        [('Can', 1), ('I', 0), ('order', 7), ('takeout', 6), ('from', 0), ('Jhilko', 3), ('food', 4), ('land', 5)],
        [('Do', 1), ('you', 2), ('have', 2), ('vegetarian', 6), ('options', 5), ('at', 0), ('Jhilko', 3), ('food', 4), ('land', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('working', 7), ('hours', 0), ('of', 0), ('Jhilko', 3), ('food', 4), ('land', 5)],
        [('Where', 1), ('can', 1), ('I', 0), ('find', 7), ('your', 0), ('menu', 5), ('online', 0), ('for', 0), ('Jhilko', 3), ('food', 4), ('land', 5)],
        [('Do', 1), ('you', 2), ('serve', 7), ('alcohol', 6), ('at', 0), ('Jhilko', 3), ('food', 4), ('land', 5)],
        [('What', 1), ('is', 2), ('the', 2), ('best', 0), ('dish', 5), ('at', 0), ('Jhilko', 3), ('food', 4), ('land', 5)],
        [('Can', 1), ('I', 0), ('get', 7), ('takeaway', 6), ('from', 0), ('Jhilko', 3), ('food', 4), ('land', 5)],
        [('Do', 1), ('you', 2), ('offer', 7), ('online', 0), ('ordering', 5), ('for', 0), ('Jhilko', 3), ('food', 4), ('land', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('most', 0), ('popular', 0), ('dishes', 5), ('at', 0), ('Jhilko', 3), ('food', 4), ('land', 5)],
        [('Is', 1), ('there', 0), ('wifi', 6), ('available', 7), ('at', 0), ('Jhilko', 3), ('food', 4), ('land', 5)],
        [('Can', 1), ('I', 0), ('reserve', 7), ('a', 0), ('table', 5), ('at', 0), ('Jhilko', 3), ('food', 4), ('land', 5)],
        [('Do', 1), ('you', 2), ('have', 2), ('a', 0), ('kids', 6), ('menu', 5), ('at', 0), ('Jhilko', 3), ('food', 4), ('land', 5)],
        [('What', 1), ('is', 2), ('the', 2), ('timing', 7), ('for', 0), ('lunch', 5), ('at', 0), ('Jhilko', 3), ('food', 4), ('land', 5)],
        [('Where', 1), ('is', 2), ('the', 2), ('nearest', 0), ('Jhilko', 3), ('food', 4), ('land', 5), ('located', 7)],
        [('Do', 1), ('you', 2), ('serve', 7), ('desserts', 6), ('at', 0), ('Jhilko', 3), ('food', 4), ('land', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('signature', 0), ('dishes', 5), ('at', 0), ('Jhilko', 3), ('food', 4), ('land', 5)],
    ],
})
questions.update({
    "Pawan sweet and chaat house": [
        [('Where', 1), ('is', 2), ('Pawan', 3), ('sweet', 5), ('and', 0), ('chaat', 5), ('house', 6), ('located', 7)],
        [('What', 1), ('are', 2), ('the', 2), ('specialties', 5), ('of', 0), ('Pawan', 3), ('sweet', 5), ('and', 0), ('chaat', 5), ('house', 6)],
        [('Do', 1), ('you', 2), ('serve', 7), ('snacks', 6), ('at', 0), ('Pawan', 3), ('sweet', 5), ('and', 0), ('chaat', 5), ('house', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('prices', 5), ('for', 0), ('chaat', 6), ('items', 5), ('at', 0), ('Pawan', 3), ('sweet', 5), ('and', 0), ('chaat', 5), ('house', 6)],
        [('Can', 1), ('I', 0), ('order', 7), ('sweets', 5), ('from', 0), ('Pawan', 3), ('sweet', 5), ('and', 0), ('chaat', 5), ('house', 6)],
        [('What', 1), ('is', 2), ('the', 2), ('special', 0), ('dish', 5), ('served', 7), ('at', 0), ('Pawan', 3), ('sweet', 5), ('and', 0), ('chaat', 5), ('house', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('working', 7), ('hours', 0), ('of', 0), ('Pawan', 3), ('sweet', 5), ('and', 0), ('chaat', 5), ('house', 6)],
        [('Do', 1), ('you', 2), ('offer', 7), ('delivery', 5), ('services', 0), ('from', 0), ('Pawan', 3), ('sweet', 5), ('and', 0), ('chaat', 5), ('house', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('most', 0), ('popular', 0), ('items', 5), ('at', 0), ('Pawan', 3), ('sweet', 5), ('and', 0), ('chaat', 5), ('house', 6)],
        [('Do', 1), ('you', 2), ('serve', 7), ('tea', 5), ('or', 0), ('coffee', 6), ('at', 0), ('Pawan', 3), ('sweet', 5), ('and', 0), ('chaat', 5), ('house', 6)],
        [('Is', 1), ('there', 0), ('a', 0), ('parking', 6), ('facility', 0), ('available', 7), ('at', 0), ('Pawan', 3), ('sweet', 5), ('and', 0), ('chaat', 5), ('house', 6)],
        [('Can', 1), ('I', 0), ('make', 7), ('a', 0), ('reservation', 5), ('at', 0), ('Pawan', 3), ('sweet', 5), ('and', 0), ('chaat', 5), ('house', 6)],
        [('Do', 1), ('you', 2), ('offer', 7), ('catering', 5), ('services', 0), ('at', 0), ('Pawan', 3), ('sweet', 5), ('and', 0), ('chaat', 5), ('house', 6)],
        [('What', 1), ('types', 5), ('of', 0), ('sweets', 6), ('are', 2), ('available', 7), ('at', 0), ('Pawan', 3), ('sweet', 5), ('and', 0), ('chaat', 5), ('house', 6)],
        [('Do', 1), ('you', 2), ('have', 2), ('gluten-free', 6), ('options', 5), ('at', 0), ('Pawan', 3), ('sweet', 5), ('and', 0), ('chaat', 5), ('house', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('best', 0), ('sellers', 5), ('at', 0), ('Pawan', 3), ('sweet', 5), ('and', 0), ('chaat', 5), ('house', 6)],
        [('Can', 1), ('I', 0), ('customize', 7), ('my', 0), ('order', 5), ('at', 0), ('Pawan', 3), ('sweet', 5), ('and', 0), ('chaat', 5), ('house', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('seasonal', 0), ('specials', 5), ('offered', 7), ('by', 0), ('Pawan', 3), ('sweet', 5), ('and', 0), ('chaat', 5), ('house', 6)],
        [('Do', 1), ('you', 2), ('have', 2), ('a', 0), ('kids', 5), ('menu', 5), ('at', 0), ('Pawan', 3), ('sweet', 5), ('and', 0), ('chaat', 5), ('house', 6)],
        [('Where', 1), ('can', 1), ('I', 0), ('find', 7), ('your', 0), ('location', 5), ('on', 0), ('Google', 5), ('Maps', 5)],
    ],
    "Royal bakundol pool and snooker house": [
        [('Where', 1), ('is', 2), ('Royal', 3), ('bakundol', 5), ('pool', 5), ('and', 0), ('snooker', 5), ('house', 6), ('located', 7)],
        [('What', 1), ('are', 2), ('the', 2), ('membership', 5), ('fees', 0), ('at', 0), ('Royal', 3), ('bakundol', 5), ('pool', 5), ('and', 0), ('snooker', 5), ('house', 6)],
        [('Can', 1), ('I', 0), ('book', 7), ('a', 0), ('table', 5), ('at', 0), ('Royal', 3), ('bakundol', 5), ('pool', 5), ('and', 0), ('snooker', 5), ('house', 6)],
        [('Do', 1), ('you', 2), ('offer', 7), ('coaching', 5), ('sessions', 0), ('at', 0), ('Royal', 3), ('bakundol', 5), ('pool', 5), ('and', 0), ('snooker', 5), ('house', 6)],
        [('What', 1), ('is', 2), ('the', 2), ('timing', 5), ('for', 0), ('games', 5), ('at', 0), ('Royal', 3), ('bakundol', 5), ('pool', 5), ('and', 0), ('snooker', 5), ('house', 6)],
        [('Is', 1), ('there', 0), ('a', 0), ('snooker', 5), ('tournament', 5), ('held', 7), ('at', 0), ('Royal', 3), ('bakundol', 5), ('pool', 5), ('and', 0), ('snooker', 5), ('house', 6)],
        # Add more questions following the same structure.
    ],
    # Continue for other businesses...
})
questions.update({
    "Royal bakundol pool and snooker house": [
        [('What', 1), ('are', 2), ('the', 2), ('working', 7), ('hours', 0), ('of', 0), ('Royal', 3), ('bakundol', 5), ('pool', 5), ('and', 0), ('snooker', 5), ('house', 6)],
        [('Do', 1), ('you', 2), ('have', 2), ('a', 0), ('snooker', 5), ('tournament', 5), ('schedule', 6), ('at', 0), ('Royal', 3), ('bakundol', 5), ('pool', 5), ('and', 0), ('snooker', 5), ('house', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('rules', 5), ('for', 0), ('using', 7), ('the', 2), ('facilities', 5), ('at', 0), ('Royal', 3), ('bakundol', 5), ('pool', 5), ('and', 0), ('snooker', 5), ('house', 6)],
        [('Do', 1), ('you', 2), ('offer', 7), ('refreshments', 5), ('or', 0), ('snacks', 6), ('at', 0), ('Royal', 3), ('bakundol', 5), ('pool', 5), ('and', 0), ('snooker', 5), ('house', 6)],
        [('Can', 1), ('I', 0), ('bring', 7), ('my', 0), ('own', 0), ('equipment', 5), ('to', 0), ('Royal', 3), ('bakundol', 5), ('pool', 5), ('and', 0), ('snooker', 5), ('house', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('entry', 0), ('charges', 5), ('for', 0), ('non-members', 5), ('at', 0), ('Royal', 3), ('bakundol', 5), ('pool', 5), ('and', 0), ('snooker', 5), ('house', 6)],
        [('Do', 1), ('you', 2), ('provide', 7), ('equipment', 5), ('on', 0), ('rent', 6), ('at', 0), ('Royal', 3), ('bakundol', 5), ('pool', 5), ('and', 0), ('snooker', 5), ('house', 6)],
        [('Are', 1), ('there', 0), ('any', 0), ('special', 0), ('events', 5), ('hosted', 7), ('at', 0), ('Royal', 3), ('bakundol', 5), ('pool', 5), ('and', 0), ('snooker', 5), ('house', 6)],
        [('Do', 1), ('you', 2), ('offer', 7), ('discounts', 5), ('for', 0), ('students', 6), ('or', 0), ('seniors', 5), ('at', 0), ('Royal', 3), ('bakundol', 5), ('pool', 5), ('and', 0), ('snooker', 5), ('house', 6)],
        [('Can', 1), ('I', 0), ('host', 7), ('a', 0), ('party', 5), ('or', 0), ('event', 5), ('at', 0), ('Royal', 3), ('bakundol', 5), ('pool', 5), ('and', 0), ('snooker', 5), ('house', 6)],
    ],

    "New star fitness gym": [
        [('Where', 1), ('is', 2), ('New', 3), ('star', 5), ('fitness', 5), ('gym', 6), ('located', 7)],
        [('What', 1), ('are', 2), ('the', 2), ('membership', 5), ('plans', 0), ('at', 0), ('New', 3), ('star', 5), ('fitness', 5), ('gym', 6)],
        [('Do', 1), ('you', 2), ('offer', 7), ('personal', 5), ('training', 5), ('sessions', 6), ('at', 0), ('New', 3), ('star', 5), ('fitness', 5), ('gym', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('working', 7), ('hours', 0), ('of', 0), ('New', 3), ('star', 5), ('fitness', 5), ('gym', 6)],
        [('Do', 1), ('you', 2), ('have', 2), ('special', 0), ('offers', 5), ('for', 0), ('first-time', 0), ('members', 5), ('at', 0), ('New', 3), ('star', 5), ('fitness', 5), ('gym', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('group', 5), ('classes', 5), ('offered', 7), ('at', 0), ('New', 3), ('star', 5), ('fitness', 5), ('gym', 6)],
        [('Can', 1), ('I', 0), ('try', 7), ('a', 0), ('free', 0), ('trial', 5), ('session', 5), ('at', 0), ('New', 3), ('star', 5), ('fitness', 5), ('gym', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('facilities', 5), ('available', 7), ('at', 0), ('New', 3), ('star', 5), ('fitness', 5), ('gym', 6)],
        [('Do', 1), ('you', 2), ('provide', 7), ('nutrition', 5), ('counseling', 0), ('services', 0), ('at', 0), ('New', 3), ('star', 5), ('fitness', 5), ('gym', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('rules', 5), ('and', 0), ('etiquette', 5), ('at', 0), ('New', 3), ('star', 5), ('fitness', 5), ('gym', 6)],
    ],
})
questions.update({
    "Tamsaling nursery": [
        [('Where', 1), ('is', 2), ('Tamsaling', 3), ('nursery', 5), ('located', 7)],
        [('What', 1), ('types', 5), ('of', 0), ('plants', 5), ('are', 2), ('available', 7), ('at', 0), ('Tamsaling', 3), ('nursery', 5)],
        [('Do', 1), ('you', 2), ('sell', 7), ('flower', 5), ('seeds', 5), ('at', 0), ('Tamsaling', 3), ('nursery', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('prices', 5), ('of', 0), ('saplings', 5), ('at', 0), ('Tamsaling', 3), ('nursery', 5)],
        [('Can', 1), ('I', 0), ('get', 7), ('organic', 0), ('fertilizers', 5), ('at', 0), ('Tamsaling', 3), ('nursery', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('working', 7), ('hours', 0), ('of', 0), ('Tamsaling', 3), ('nursery', 5)],
        [('Do', 1), ('you', 2), ('provide', 7), ('gardening', 5), ('consultations', 0), ('at', 0), ('Tamsaling', 3), ('nursery', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('best', 0), ('selling', 7), ('plants', 5), ('at', 0), ('Tamsaling', 3), ('nursery', 5)],
        [('Do', 1), ('you', 2), ('sell', 7), ('indoor', 5), ('plants', 5), ('at', 0), ('Tamsaling', 3), ('nursery', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('special', 0), ('offers', 5), ('available', 7), ('at', 0), ('Tamsaling', 3), ('nursery', 5)],
        [('Can', 1), ('I', 0), ('buy', 7), ('succulents', 5), ('at', 0), ('Tamsaling', 3), ('nursery', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('seasonal', 0), ('plants', 5), ('available', 7), ('at', 0), ('Tamsaling', 3), ('nursery', 5)],
        [('Do', 1), ('you', 2), ('offer', 7), ('bulk', 0), ('discounts', 5), ('on', 0), ('plant', 5), ('purchases', 6), ('at', 0), ('Tamsaling', 3), ('nursery', 5)],
        [('Can', 1), ('I', 0), ('order', 7), ('plants', 5), ('online', 5), ('from', 0), ('Tamsaling', 3), ('nursery', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('rare', 0), ('plants', 5), ('available', 7), ('at', 0), ('Tamsaling', 3), ('nursery', 5)],
        [('Do', 1), ('you', 2), ('sell', 7), ('bonsai', 5), ('trees', 5), ('at', 0), ('Tamsaling', 3), ('nursery', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('shipping', 5), ('options', 5), ('for', 0), ('plants', 5), ('from', 0), ('Tamsaling', 3), ('nursery', 5)],
        [('Do', 1), ('you', 2), ('provide', 7), ('planting', 5), ('instructions', 0), ('with', 0), ('purchases', 6), ('from', 0), ('Tamsaling', 3), ('nursery', 5)],
        [('Can', 1), ('I', 0), ('return', 7), ('plants', 5), ('if', 0), ('they', 0), ('don’t', 0), ('grow', 7), ('from', 0), ('Tamsaling', 3), ('nursery', 5)],
        [('What', 1), ('are', 2), ('the', 2), ('care', 0), ('tips', 5), ('for', 0), ('plants', 5), ('bought', 7), ('from', 0), ('Tamsaling', 3), ('nursery', 5)],
    ],

    "Clothes collection business": [
        [('Where', 1), ('is', 2), ('Clothes', 3), ('collection', 5), ('business', 6), ('located', 7)],
        [('What', 1), ('are', 2), ('the', 2), ('clothing', 5), ('categories', 5), ('available', 7), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('Do', 1), ('you', 2), ('sell', 7), ('formal', 5), ('wear', 5), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('prices', 5), ('of', 0), ('t-shirts', 5), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('Can', 1), ('I', 0), ('find', 7), ('designer', 5), ('clothes', 5), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('Do', 1), ('you', 2), ('have', 2), ('seasonal', 0), ('sales', 5), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('return', 7), ('policies', 5), ('of', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('Do', 1), ('you', 2), ('offer', 7), ('customization', 5), ('services', 0), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('popular', 0), ('brands', 5), ('available', 7), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('Can', 1), ('I', 0), ('purchase', 7), ('clothes', 5), ('online', 5), ('from', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('Do', 1), ('you', 2), ('have', 2), ('clothes', 5), ('for', 0), ('children', 5), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('delivery', 5), ('options', 5), ('from', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('Can', 1), ('I', 0), ('get', 7), ('gift', 0), ('cards', 5), ('from', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('special', 0), ('offers', 5), ('available', 7), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('Do', 1), ('you', 2), ('have', 2), ('tailoring', 5), ('services', 0), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('most', 0), ('popular', 0), ('products', 5), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('Do', 1), ('you', 2), ('offer', 7), ('discounts', 5), ('for', 0), ('bulk', 0), ('purchases', 6), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('Can', 1), ('I', 0), ('exchange', 7), ('clothes', 5), ('purchased', 7), ('from', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('payment', 5), ('methods', 0), ('accepted', 7), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('Do', 1), ('you', 2), ('sell', 7), ('seasonal', 0), ('collections', 5), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
    ],
})
questions.update({
    "Clothing": [
        [('Where', 1), ('is', 2), ('the', 2), ('Clothes', 3), ('collection', 5), ('business', 6), ('located', 7)],
        [('What', 1), ('types', 5), ('of', 0), ('clothes', 5), ('do', 1), ('you', 2), ('sell', 7), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('Can', 1), ('I', 0), ('find', 7), ('custom', 0), ('tailored', 5), ('clothes', 5), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('Do', 1), ('you', 2), ('offer', 7), ('discounts', 5), ('on', 0), ('clothing', 5), ('purchases', 6), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('What', 1), ('is', 2), ('the', 2), ('average', 0), ('price', 5), ('range', 5), ('of', 0), ('clothes', 5), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('Can', 1), ('I', 0), ('buy', 7), ('clothes', 5), ('online', 5), ('from', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('working', 7), ('hours', 0), ('of', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('Do', 1), ('you', 2), ('have', 2), ('a', 5), ('loyalty', 5), ('program', 0), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('most', 0), ('popular', 0), ('clothing', 5), ('items', 5), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
        [('Can', 1), ('I', 0), ('get', 7), ('clothing', 5), ('alterations', 5), ('at', 0), ('Clothes', 3), ('collection', 5), ('business', 6)],
    ],

    "Stationary": [
        [('Where', 1), ('is', 2), ('KU', 3), ('gate', 5), ('stationary', 6), ('store', 0), ('located', 7)],
        [('What', 1), ('types', 5), ('of', 0), ('stationery', 5), ('are', 2), ('available', 7), ('at', 0), ('KU', 3), ('gate', 5), ('stationary', 6)],
        [('Do', 1), ('you', 2), ('sell', 7), ('notebooks', 5), ('and', 0), ('pens', 5), ('at', 0), ('KU', 3), ('gate', 5), ('stationary', 6)],
        [('Can', 1), ('I', 0), ('buy', 7), ('custom', 0), ('printed', 5), ('stationery', 5), ('from', 0), ('KU', 3), ('gate', 5), ('stationary', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('prices', 5), ('of', 0), ('stationery', 5), ('items', 6), ('at', 0), ('KU', 3), ('gate', 5), ('stationary', 6)],
        [('Do', 1), ('you', 2), ('offer', 7), ('bulk', 0), ('discounts', 5), ('on', 0), ('stationery', 5), ('purchases', 6), ('at', 0), ('KU', 3), ('gate', 5), ('stationary', 6)],
        [('Can', 1), ('I', 0), ('find', 7), ('art', 5), ('supplies', 5), ('at', 0), ('KU', 3), ('gate', 5), ('stationary', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('operating', 7), ('hours', 0), ('of', 0), ('KU', 3), ('gate', 5), ('stationary', 6)],
        [('Do', 1), ('you', 2), ('have', 2), ('office', 5), ('supplies', 5), ('available', 7), ('at', 0), ('KU', 3), ('gate', 5), ('stationary', 6)],
        [('Can', 1), ('I', 0), ('order', 7), ('stationery', 5), ('online', 5), ('from', 0), ('KU', 3), ('gate', 5), ('stationary', 6)],
    ],

    "Hostel": [
        [('Where', 1), ('is', 2), ('Divya', 3), ('girls', 5), ('hostel', 6), ('located', 7)],
        [('What', 1), ('are', 2), ('the', 2), ('room', 5), ('rates', 5), ('at', 0), ('Divya', 3), ('girls', 5), ('hostel', 6)],
        [('Do', 1), ('you', 2), ('provide', 7), ('food', 5), ('services', 0), ('at', 0), ('Divya', 3), ('girls', 5), ('hostel', 6)],
        [('Can', 1), ('I', 0), ('get', 7), ('single', 5), ('rooms', 5), ('at', 0), ('Divya', 3), ('girls', 5), ('hostel', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('check', 7), ('in', 0), ('and', 0), ('check', 7), ('out', 5), ('timings', 0), ('at', 0), ('Divya', 3), ('girls', 5), ('hostel', 6)],
        [('Do', 1), ('you', 2), ('have', 2), ('Wi-Fi', 5), ('available', 7), ('at', 0), ('Divya', 3), ('girls', 5), ('hostel', 6)],
        [('Can', 1), ('I', 0), ('reserve', 7), ('a', 5), ('room', 5), ('in', 0), ('advance', 5), ('at', 0), ('Divya', 3), ('girls', 5), ('hostel', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('amenities', 5), ('offered', 7), ('at', 0), ('Divya', 3), ('girls', 5), ('hostel', 6)],
        [('Do', 1), ('you', 2), ('allow', 7), ('visitors', 5), ('at', 0), ('Divya', 3), ('girls', 5), ('hostel', 6)],
        [('What', 1), ('is', 2), ('the', 2), ('duration', 5), ('of', 0), ('stay', 5), ('allowed', 7), ('at', 0), ('Divya', 3), ('girls', 5), ('hostel', 6)],
    ],

    "Electronics": [
        [('Where', 1), ('is', 2), ('Sampada', 3), ('lights', 5), ('and', 0), ('electrical', 5), ('suppliers', 6), ('located', 7)],
        [('What', 1), ('types', 5), ('of', 0), ('electronics', 5), ('do', 1), ('you', 2), ('sell', 7), ('at', 0), ('Sampada', 3), ('lights', 5), ('and', 0), ('electrical', 5), ('suppliers', 6)],
        [('Do', 1), ('you', 2), ('sell', 7), ('lighting', 5), ('fixtures', 5), ('at', 0), ('Sampada', 3), ('lights', 5), ('and', 0), ('electrical', 5), ('suppliers', 6)],
        [('Can', 1), ('I', 0), ('buy', 7), ('electrical', 5), ('appliances', 5), ('online', 5), ('from', 0), ('Sampada', 3), ('lights', 5), ('and', 0), ('electrical', 5), ('suppliers', 6)],
        [('What', 1), ('are', 2), ('the', 2), ('prices', 5), ('of', 0), ('electronic', 5), ('items', 6), ('at', 0), ('Sampada', 3), ('lights', 5), ('and', 0), ('electrical', 5), ('suppliers', 6)],
        [('Do', 1), ('you', 2), ('offer', 7), ('warranty', 5), ('on', 0), ('electronics', 5), ('purchases', 6), ('at', 0), ('Sampada', 3), ('lights', 4), ('and', 4), ('electrical', 5), ('suppliers', 6)],
        [('Can', 1), ('I', 0), ('get', 7), ('repair', 5), ('services', 5), ('at', 0), ('Sampada', 3), ('lights', 5), ('and', 0), ('electrical', 4), ('suppliers', 4)],
        [('What', 1), ('are', 2), ('the', 2), ('available', 7), ('brands', 5), ('of', 0), ('electronics', 5), ('at', 0), ('Sampada', 3), ('lights', 4), ('and', 4), ('electrical', 5), ('suppliers', 6)],
        [('Do', 1), ('you', 2), ('have', 2), ('any', 5), ('discounts', 5), ('on', 0), ('electronic', 5), ('products', 6), ('at', 0), ('Sampada', 3), ('lights', 4), ('and', 4), ('electrical', 4), ('suppliers', 4)],
        [('What', 1), ('are', 2), ('the', 2), ('working', 7), ('hours', 0), ('of', 0), ('Sampada', 3), ('lights', 4), ('and', 4), ('electrical', 4), ('suppliers', 4)],
    ]
})
questions.update({
    "Food Delivery": [
        [('Where', 1), ('can', 1), ('I', 0), ('order', 7), ('food', 5), ('from', 0), ('Downtown', 3), ('food', 4), ('delivery', 4)],
        [('What', 1), ('types', 5), ('of', 0), ('food', 5), ('do', 1), ('you', 2), ('deliver', 7), ('at', 0), ('Downtown', 3), ('food', 4), ('delivery', 4)],
        [('Can', 1), ('I', 0), ('order', 7), ('food', 5), ('online', 5), ('from', 0), ('Downtown', 3), ('food', 4), ('delivery', 4)],
        [('Do', 1), ('you', 2), ('offer', 7), ('express', 5), ('delivery', 5), ('at', 0), ('Downtown', 3), ('food', 4), ('delivery', 4)],
        [('What', 1), ('are', 2), ('the', 2), ('delivery', 5), ('charges', 5), ('at', 0), ('Downtown', 3), ('food', 4), ('delivery', 4)],
        [('Do', 1), ('you', 2), ('deliver', 7), ('to', 0), ('other', 0), ('areas', 5), ('besides', 0), ('the', 2), ('main', 5), ('zone', 5), ('at', 0), ('Downtown', 3), ('food', 4), ('delivery', 4)],
        [('What', 1), ('are', 2), ('the', 2), ('working', 7), ('hours', 0), ('of', 0), ('Downtown', 3), ('food', 4), ('delivery', 4)],
        [('Do', 1), ('you', 2), ('have', 2), ('any', 5), ('discounts', 5), ('on', 0), ('food', 5), ('orders', 6), ('at', 0), ('Downtown', 3), ('food', 4), ('delivery', 4)],
        [('Can', 1), ('I', 0), ('schedule', 7), ('a', 5), ('food', 5), ('delivery', 6), ('for', 0), ('later', 5), ('at', 0), ('Downtown', 3), ('food', 4), ('delivery', 4)],
        [('Do', 1), ('you', 2), ('offer', 7), ('meal', 5), ('plans', 0), ('at', 0), ('Downtown', 3), ('food', 4), ('delivery', 4)],
    ],

    "Liquor": [
        [('Where', 1), ('can', 1), ('I', 0), ('buy', 7), ('liquor', 5), ('from', 0), ('Nischal', 3), ('liquor', 4)],
        [('What', 1), ('types', 5), ('of', 0), ('liquor', 5), ('do', 1), ('you', 2), ('sell', 7), ('at', 0), ('Nischal', 3), ('liquor', 4)],
        [('Do', 1), ('you', 2), ('offer', 7), ('discounts', 5), ('on', 0), ('liquor', 5), ('purchases', 6), ('at', 0), ('Nischal', 3), ('liquor', 4)],
        [('Can', 1), ('I', 0), ('buy', 7), ('liquor', 5), ('online', 5), ('from', 0), ('Nischal', 3), ('liquor', 4)],
        [('What', 1), ('are', 2), ('the', 2), ('prices', 5), ('of', 0), ('liquor', 5), ('at', 0), ('Nischal', 3), ('liquor', 4)],
        [('Do', 1), ('you', 2), ('sell', 7), ('spirits', 5), ('and', 0), ('wine', 5), ('at', 0), ('Nischal', 3), ('liquor', 4)],
        [('What', 1), ('are', 2), ('the', 2), ('working', 5), ('hours', 5), ('of', 0), ('Nischal', 3), ('liquor', 4)],
        [('Can', 1), ('I', 0), ('find', 7), ('craft', 5), ('beer', 5), ('at', 0), ('Nischal', 3), ('liquor', 4)],
        [('Do', 1), ('you', 2), ('offer', 7), ('delivery', 7), ('services', 0), ('for', 0), ('liquor', 5), ('at', 0), ('Nischal', 3), ('liquor', 4)],
        [('Can', 1), ('I', 0), ('get', 7), ('liquor', 5), ('gift', 5), ('packs', 5), ('at', 0), ('Nischal', 3), ('liquor', 4)],
    ],

    "Parks": [
        [('Where', 1), ('is', 2), ('Bakundole', 3), ('buddha', 4), ('park', 4), ('located', 5)],
        [('What', 1), ('activities', 5), ('can', 1), ('I', 0), ('do', 7), ('at', 0), ('Bakundole', 3), ('buddha', 4), ('park', 4)],
        [('Can', 1), ('I', 0), ('book', 7), ('a', 5), ('spot', 5), ('at', 0), ('Bakundole', 3), ('buddha', 4), ('park', 4)],
        [('What', 1), ('are', 2), ('the', 2), ('entrance', 5), ('fees', 6), ('for', 0), ('Bakundole', 3), ('buddha', 4), ('park', 4)],
        [('Do', 1), ('you', 2), ('offer', 7), ('guided', 5), ('tours', 5), ('at', 0), ('Bakundole', 3), ('buddha', 4), ('park', 4)],
        [('Are', 1), ('there', 2), ('any', 5), ('seasonal', 5), ('events', 5), ('at', 0), ('Bakundole', 3), ('buddha', 4), ('park', 4)],
        [('What', 1), ('are', 2), ('the', 2), ('working', 7), ('hours', 0), ('of', 0), ('Bakundole', 3), ('buddha', 4), ('park', 4)],
        [('Do', 1), ('you', 2), ('offer', 7), ('picnic', 5), ('areas', 5), ('at', 0), ('Bakundole', 3), ('buddha', 4), ('park', 4)],
        [('Can', 1), ('I', 0), ('host', 7), ('events', 5), ('at', 0), ('Bakundole', 3), ('buddha', 4), ('park', 4)],
        [('Is', 1), ('there', 2), ('a', 5), ('nature', 5), ('trail', 5), ('at', 0), ('Bakundole', 3), ('buddha', 4), ('park', 4)],
    ]
})

question2 = {}
actual_list2 = []
for key,value_list in questions.items():
    to_append = {}
    temp_list =[]
    for lists in value_list:
        val = False
        val_list1 = []

        for (word,index) in lists:
            word , value = word , index
            if val and value in [5,6,0]:
                val1 = 4
            else:
                val1 = value
            val_list1.append((word ,val1))
            if value == 3:
                val = True

        temp_list.append(val_list1)







    to_append[key] = temp_list
    question2.update(to_append)
actual_list = []
for key,value in question2.items():
    for lists in value:
        actual_list.append(lists)

print("data_loaded")






import tensorflow as tf
from keras.layers import Layer

class AlbertLayer(Layer):
    def __init__(self, albert_model, **kwargs):
        super().__init__(**kwargs)
        self.albert_model = albert_model

    def call(self, inputs):
        input_ids, attention_mask = inputs
        output = self.albert_model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state

    def get_config(self):
        # Get the configuration of the layer
        config = super().get_config()
        config.update({
            "albert_model": self.albert_model  # Include the model in the configuration
        })
        return config

###################################
class model_forthsem_Ner:
    def __init__(self):
        self.units = 64
        self.ner_tags = 9
        self.embedding_dims = 768
        self.optimizer = "adam"
        self.loss = "categorical_crossentropy" #for the outputs that are already in one hot encodded or to_categorical
        self.name="forth_sem_ner.keras"
        self.file_name = "recent3.txt"
        self.tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        self.albert = TFAutoModel.from_pretrained("albert-base-v2")



    def make_model(self):
        input_tokens = Input(shape=(None,), dtype="int32", name="input_token")
        input_attention = Input(shape=(None,), dtype="int32", name="input_attention")

    # Wrap the ALBERT model in a custom Keras layer
        albert_output = AlbertLayer(self.albert)([input_tokens, input_attention])

    # Additional layers
        layer1 = Bidirectional(LSTM(self.units, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(albert_output)
        norm1 = LayerNormalization()(layer1)

        layer2 = Bidirectional(LSTM(self.units, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(norm1)
        norm2 = LayerNormalization()(layer2)

        layer3 = Bidirectional(LSTM(self.units, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(norm2)

    # Attention mechanism
        attention = Attention()([layer3, layer3])  # Self-attention mechanism
        concatenated = Concatenate()([layer3, attention])  # Combine LSTM outputs with attention

    # Fully connected output layer
        output_model = TimeDistributed(Dense(self.ner_tags, activation="softmax"), name="output_model")(concatenated)

        Modelx = Model(inputs = [input_tokens , input_attention] , outputs = output_model)
        Modelx.compile(optimizer = self.optimizer , loss = self.loss , metrics=["accuracy"])

        return Modelx
    """
    def vectorize(self, sentences):
        sentence = " ".join(sentences)

        tokens = sentence.split()  # Simple split on spaces
        tokens = [token.lower() for token in tokens]

        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')

        token_ids = []
        for token in tokens:
            token_id = bert_tokenizer.vocab.get(token, bert_tokenizer.vocab['[UNK]'])  # Map to [UNK] if not found
            token_ids.append(token_id)

        input_ids = torch.tensor([token_ids])
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

        with torch.no_grad():
            outputs = bert_model(input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state[0]
        word_vector_dict = {}
        list_vectors = []
        for token, embedding in zip(tokens, token_embeddings):
            word_vector_dict[token] = embedding.numpy()
            list_vectors.append(embedding.numpy().tolist())

        return word_vector_dict, list_vectors
"""


    def make_data_for_model(self , combined_list1):
        combined_list = []
        file_name = self.file_name
        """
        with open(file_name, 'r') as file:
            for line in file:
                line = line.strip()  # Remove leading/trailing whitespace
                if line:  # Ignore empty lines
                    try:
                    # Convert the string representation of the list to an actual Python object
                        parsed_line = ast.literal_eval(line)
                        combined_list.append(parsed_line)

                    except (ValueError, SyntaxError) as e:
                        print(f"Error parsing line: {line}\n{e}")
        """

        input_list = []
        output_list = []
        sentences=[]
        y_labels = []
        combined_list  = combined_list1[1:2]

        for sentence in combined_list:
            temp_list = []
            temp_label = []
            for (word, index) in sentence:

                temp_list.append(word)
                temp_label.append(index)
            print("temp_list is ",temp_list)
            temp_sent = " ".join(temp_list)

            atten = self.tokenizer(
            temp_sent,
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt"
            )
            input_ids = atten["input_ids"]
            attention_mask = atten["attention_mask"]

            input_list.append(input_ids)
            output_list.append(attention_mask)
            temp_label_with_specials = [0] + temp_label + [0]



            y_labels.append(temp_label_with_specials)







        max_leng = 30
        #x_input = pad_sequences(sentences , maxlen = max_leng , dtype="float32" , value=0.0)

        y_labels = pad_sequences(y_labels,maxlen = max_leng , dtype="int32", value=0)
        y_input = to_categorical(y_labels , num_classes = self.ner_tags)
        input_list , output_list = np.array(input_list) , np.array(output_list)
        return input_list , output_list , y_input
    def make_or_load_model(self):
        try:
            model = load_model(self.name)
            print("model loaded")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            model = self.make_model()

            model.save(self.name)
            print("model saved")
            return model

    def train_model(self , combined_list):
        model = self.make_or_load_model()
        input_list , output_list, y_input = self.make_data_for_model(combined_list)
        print(input_list.shape)
        print(output_list.shape)
        print(y_input.shape)
        input_list = tf.squeeze(input_list, axis = 0)
        output_list = tf.squeeze(output_list , axis = 0)
        model.fit([input_list , output_list] , y_input , epochs = 150 , batch_size = 32)
        model.save(self.name)
        return model
    def predict_from_model(self, prediction_variable):
    # Load the model
        model = self.make_or_load_model()

    # Tokenize the input prediction variable
        temp_var = self.tokenizer(
        prediction_variable,
        padding="max_length",
        truncation=True,
        max_length=30,
        return_tensors="tf",

    )
        input_ids = temp_var["input_ids"]
        input_attention = temp_var["attention_mask"]
        word_ids = temp_var.word_ids()  # Track word IDs to align tokens back to words

    # Predict with the model
        prediction = model.predict([input_ids, input_attention])
        prediction = np.argmax(prediction, axis=2)  # Shape: (batch_size, sequence_length)

    # Flatten predictions and remove padding
        predictions = prediction[0]  # First batch
        word_ids = np.array(word_ids)  # Convert to array for indexing
        valid_indices = word_ids != None  # Ignore padding tokens
        predictions = predictions[valid_indices]
        word_ids = word_ids[valid_indices]

    # Map token-level predictions to word-level predictions
        word_to_prediction = {}
        for word_id, pred in zip(word_ids, predictions):
            if word_id not in word_to_prediction:
                word_to_prediction[word_id] = []
            word_to_prediction[word_id].append(pred)

    # Use the most frequent NER tag for each word
        final_predictions = []
        for word_id in sorted(word_to_prediction.keys()):
            word_preds = word_to_prediction[word_id]
            final_predictions.append(max(set(word_preds), key=word_preds.count))

    # NER tag dictionary
        dict_ner = {
        0: "Others",
        1: "B-question_type",
        2: "I-question_type",
        3: "B-target_entity",
        4: "I-target_entity",
        5: "B-context",
        6: "I-context",
        7: "B-action",
        8: "I-action"
    }

    # Convert prediction to readable tags
        result_list = [dict_ner[pred] for pred in final_predictions]

    # Map the predicted tags to their corresponding labels
        new_dict = {
        "question": "",
        "business_entity": "",
        "context": "",
        "action": "",
        "context_word": "",
        "others": "",  # To handle unclassified tokens
        "action_words": ""  # To handle action-related words
    }

        tag_to_key = {
        "B-question_type": "question",
        "I-question_type": "question",
        "B-context": "context",
        "I-context": "context",
        "B-target_entity": "business_entity",
        "I-target_entity": "business_entity",
        "B-action": "action",
        "I-action": "action"
    }

    # Populate new_dict with the prediction results
        words = prediction_variable.split()  # Original words
        for word, tag in zip(words, result_list):
            key = tag_to_key.get(tag, "others")
            new_dict[key] += f" {word}"
            if tag in ["B-context", "I-context"]:
                new_dict["context_word"] += f" {word}"
            if tag in ["B-action", "I-action"]:
                new_dict["action_words"] += f" {word}"

    # Assign the action words to the action key
        new_dict["action"] = new_dict["action_words"]

    # Clean up empty strings in the new_dict
        for key in new_dict:
            new_dict[key] = new_dict[key].strip() or False

    # Return the final new_dict
        return new_dict











sc = model_forthsem_Ner()
print(len(actual_list))
sc.train_model(actual_list[1:3])
#sc.predict_from_model("Where is Bakundole buddha park located")



