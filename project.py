# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 19:51:15 2025

@author: Junhao (Timothy) Lou, C00567636
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Paper(object):
     def __init__(self, paper_id, paper_label):
         self.label = paper_label
         self.features = np.array([paper_id])
        
     def get_label(self):
         return self.label
    
     def get_features(self):
         return self.features
     
def split_50_50(examples):
    # randomly pick about 50% of examples for training
    # and rest 50% for testing
    select_for_train_percentage = 0.5
    training_set = []
    test_set = []
    for i in range(len(examples)):
        if np.random.rand(1) < select_for_train_percentage:
            training_set.append(examples.iloc[i])
        else:
            test_set.append(examples.iloc[i])
    return training_set, test_set

# Section 1: prepare data

# method 1, use example [1]'s source to download cora.cites and cora.contents
# in example [1]'s format
zip_file = keras.utils.get_file(
    fname="cora.tgz",
    origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    extract=True,
    )

data_dir = os.path.join(os.path.dirname(zip_file), "cora_extracted\\cora")

citations = pd.read_csv(
    os.path.join(data_dir, "cora.cites"),
    sep="\t",
    header=None,
    names=["cited_paper_id","citing_paper_id"],
    )

print(citations)

papers = pd.read_csv(
    os.path.join(data_dir, "cora.content"),
    sep="\t",
    header=None,
    names=["paper_id"] + [f"word_{index}" for index in range(1433)] + ["class_label"],
    )

print(papers)
    
# # method 2:
# # directly download CORA database tables (cites, content and paper) from 
# # relational-data.org using SQL workbench's table export function with csv 
# # format with "," separator. Then merge content and paper table to generate
# # similar table in example [1]'s format. Because original content table's
# # word_cited_id column is not sorted, so get_dummies function cannot get a
# # sorted collumns, but still ok for training or testing as long as software
# # remember the order of columns in generate contents_word_vector table.

# cite_file_name = "CORA.cites.csv"

# citations = pd.read_csv(cite_file_name)

# print(citations)

# content_file_name = "CORA.content.csv"

# contents = pd.read_csv(content_file_name)
# print("contents: \n", contents)

# contents_dummies = pd.get_dummies(contents, columns=["word_cited_id"])
# contents_word_vector = contents_dummies.groupby("paper_id").sum()
# print("contents_word_vector: \n", contents_word_vector)


# paper_file_name = "CORA.paper.csv"

# papers_original = pd.read_csv(paper_file_name)
# print("Original CORA paper.csv size: ", papers_original.shape)

# papers = pd.merge(contents_word_vector,papers_original, how ="left", on="paper_id")
# print("Original papers table size: ", papers_original.shape)
# print("Converted papers table size: ", papers.shape)

# print(papers)

#scale inputs
# paper id in the original CORA.paper.csv is already sorted and unique
paper_id_index = {name:idx for idx, name in enumerate(papers["paper_id"].unique())}
print(paper_id_index)
papers["paper_id"] = papers["paper_id"].apply(lambda name:paper_id_index[name])
print(papers)

#index citations
citations["cited_paper_id"] = citations["cited_paper_id"].apply(lambda name:paper_id_index[name])
citations["citing_paper_id"] = citations["citing_paper_id"].apply(lambda name:paper_id_index[name])
print(citations)

#scale outputs
unique_class_labels = sorted(papers["class_label"].unique())
print(unique_class_labels)
class_label_index = {name:idx for idx, name in enumerate(unique_class_labels)}
print(class_label_index)
papers["class_label"] = papers["class_label"].apply(lambda name:class_label_index[name])
print("papers: \n", papers)

# print("Papers group by class label", papers.groupby("class_label"))
# split newly indexed papers table 50% for training, 50% for test
(training_set, test_set) = split_50_50(papers)

print("training_set: \n", training_set)
print("test_set: \n", test_set)

training_set_DataFrame = pd.DataFrame.from_records(training_set)
print(training_set_DataFrame)
test_set_DataFrame = pd.DataFrame.from_records(test_set)
print(test_set_DataFrame)


# Section 2 - implement baseline feedforward neural network
def create_ffn(hidden_units,dropout_rate, name=None):
    fnn_layers = []
    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))
    return keras.Sequential(fnn_layers,name=name)

def baseline_ffn_model(hidden_units, num_classes, dropout_rate):
    baseline_inputs = layers.Input(shape=(num_features,), name="input_features")
    x = create_ffn(hidden_units, dropout_rate, name=f"ffn_block1")(baseline_inputs)
    for i in range(4):
        x_add = create_ffn(hidden_units, dropout_rate, name=f"ffn_block{i+2}")(x)
        x = layers.Add(name=f"skip_connection{i+2}")([x,x_add])
    logits = layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs=baseline_inputs, outputs=logits, name="baseline")

def run_experiment(model, x_train, y_train):
    model.compile(optimizer = keras.optimizers.Adam(learn_rate), 
                  loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = [keras.metrics.SparseCategoricalAccuracy(name="acc")],
                  )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor = "val_acc", 
        patience = 50, 
        restore_best_weights = True,
        )
    history = model.fit(
        x = x_train,
        y = y_train,
        epochs = num_epochs,
        validation_split = 0.15,
        batch_size = batch_size,
        callbacks = [early_stopping],
        )
    return history

def display_learning_curve(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train","validate"], loc="upper right")
    ax1.set_xlabel("number of epochs")
    ax1.set_ylabel("loss")
    
    ax2.plot(history.history["acc"])
    ax2.plot(history.history["val_acc"])
    ax2.legend(["train","validate"], loc="lower right")
    ax2.set_xlabel("number of epochs")
    ax2.set_ylabel("accuracy")
    return
    

hidden_units = [32,32]
dropout_rate = 0.5
learn_rate = 0.01
num_epochs = 100
batch_size = 256

feature_names = list(set(papers.columns) - {"paper_id", "class_label"})
num_features = len(feature_names)
num_classes = len(class_label_index)

# baseline train data
x_train_baseline = training_set_DataFrame[feature_names].to_numpy()
print("x_train_baseline:", x_train_baseline)
y_train_baseline = training_set_DataFrame["class_label"]
print("y_train_baseline:", y_train_baseline)
# baseline test data
x_test_baseline = test_set_DataFrame[feature_names].to_numpy()
print("x_test_baseline:", x_test_baseline)
y_test_baseline = test_set_DataFrame["class_label"]
print("y_test_baseline:", y_test_baseline)

baseline_model = baseline_ffn_model(hidden_units, num_classes, dropout_rate)
baseline_model.summary()                    

history = run_experiment(baseline_model, x_train_baseline, y_train_baseline)
display_learning_curve(history)

_, test_accuracy = baseline_model.evaluate(x=x_test_baseline, y=y_test_baseline, verbose=0)
print("Baseline FFN test accuracy = ", test_accuracy)

#Section 3: Graph Neural Network
def create_gru(hidden_units, dropout_rate):
    inputs = keras.layers.Input(shape=(2, hidden_units[0]))
    x = inputs
    for units in hidden_units:
        x = layers.GRU(
            units = units,
            activation = "tanh",
            recurrent_activation = "sigmoid",
            return_sequences = True,
            dropout = dropout_rate,
            return_state = False,
            recurrent_dropout = dropout_rate,
            )(x)
    return keras.Model(inputs=inputs, outputs=x)

class GCL(layers.Layer):
    def __init__(self,hidden_units,dropout_rate=0.2,aggregation_type="mean",
                 combination_type="concat", normalize=False, *args, **kwargs,):
        super().__init__(*args,**kwargs)
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize
        self.ffn_prepare = create_ffn(hidden_units,dropout_rate)
        if (self.aggregation_type=="gru"):
            self.update_fn = create_gru(hidden_units,dropout_rate)
        else:
            self.update_fn = create_ffn(hidden_units,dropout_rate)
        
    def prepare(self, node_representations, weights=None):
        messages = self.ffn_prepare(node_representations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1)
        return messages
                
    def aggregate(self, node_indices, neighbor_messages, node_representations):

        num_nodes = node_representations.shape[0]
        if(self.aggregation_type == "sum"):
            aggregated_message = tf.math.unsorted_segment_sum(neighbor_messages, node_indices, num_segments=num_nodes)
        elif(self.aggregation_type == "mean"):
            aggregated_message = tf.math.unsorted_segment_mean(neighbor_messages, node_indices, num_segments=num_nodes)            
        elif(self.aggregation_type == "max"):
            aggregated_message = tf.math.unsorted_segment_max(neighbor_messages, node_indices, num_segments=num_nodes)
        else:
            raise ValueError(f"Invalid aggregate type: {self.aggregation_type}.")
            
        return aggregated_message
    
    def update(self, node_representations, aggregated_messages):
        
        if (self.combination_type == "gru"):
            h = tf.stack([node_representations, aggregated_messages], axis = 1)
        elif (self.combination_type == "concat"):
            h = tf.concat([node_representations, aggregated_messages], axis = 1)
        elif (self.combination_type == "add"):
            h = node_representations + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")
            
        node_embeddings = self.update_fn(h)
        
        if (self.combination_type == "gru"):
            node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]
            
        if (self.normalize):
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        
        return node_embeddings
    
    def call(self, inputs):
        
        node_representations, edges, edge_weights = inputs
        node_indices, neighbor_indices = edges[0], edges[1]
        
        #generate neighbors' messages from all neighbors' node representations
        neighbor_representations = tf.gather(node_representations, neighbor_indices)
        neighbor_messages = self.prepare(neighbor_representations, edge_weights)
        
        #aggregate neighbors' messages to a single aggregate message
        aggregated_messages = self.aggregate(node_indices, neighbor_messages, node_representations)
        
        return self.update(node_representations, aggregated_messages)
    
class GNNNodeClassifier(tf.keras.Model):
    def __init__(self, 
                 graph_info, 
                 num_classes, 
                 hidden_units, 
                 aggregation_type = "sum",
                 combination_type = "concat",
                 dropout_rate = 0.2,
                 normalize = True,
                 *arg,
                 **kwarg,
                 ):
        super().__init__(*arg, **kwarg)
        node_features, edges, edge_weights = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights
        
        if (self.edge_weights is None):
            self.edge_weights = tf.ones(shape=edges.shape[1])
        self.edge_weights = self.edge_weights/tf.math.reduce_sum(self.edge_weights)
        # 1 ffn layer to preprocess node features to node presentations
        self.preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")
        # 1st and 2nd graph convolutional layers to convert node presentations
        # to node embeddings
        self.conv1 = GCL(hidden_units,dropout_rate,aggregation_type,
                     combination_type, normalize, name="Graph_Conv1")
        self.conv2 = GCL(hidden_units,dropout_rate,aggregation_type,
                     combination_type, normalize, name="Graph_Conv2")
        # another ffn layer to postprocess node embeddings outputs from 2 graph
        # conv layers
        self.postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")
        self.compute_logits = layers.Dense(units=num_classes, name="logits")
        
    def call(self, input_node_indices):
        
        x = self.preprocess(self.node_features)
        x1 = self.conv1((x, self.edges, self.edge_weights))
        x = x1 + x #skip connection
        x2 = self.conv2((x, self.edges, self.edge_weights))
        x = x2 + x
        x = self.postprocess(x)
        node_embeddings = tf.gather(x, input_node_indices)
        
        return self.compute_logits(node_embeddings)


edges = citations[["citing_paper_id","cited_paper_id"]].to_numpy().T
num_edges = edges.shape[1]
edge_weights = tf.ones(shape=num_edges)
node_features = tf.cast(
    papers.sort_values("paper_id")[feature_names].to_numpy(), dtype=tf.dtypes.float32)
graph_info = (node_features, edges, edge_weights)

print("Edges shape:", edges.shape)
print("Nodes shape:", node_features.shape)

gnn_model = GNNNodeClassifier(graph_info=graph_info,
                              num_classes=num_classes,
                              hidden_units=hidden_units,
                              dropout_rate=dropout_rate,
                              name="gnn_model",)

#input_node_indices = [1, 10, 100]
#print("GNN output shape: ", gnn_model([1, 10, 100]))

gnn_model.summary()

# train_data_GNN2 = pd.concat(training_set).sample(frac=1)
# x_train_GNN2 = train_data_GNN2.paper_id.to_numpy()
# y_train_GNN2 = train_data_GNN2["class_label"]

x_train_GNN = training_set_DataFrame["paper_id"]
y_train_GNN = training_set_DataFrame["class_label"]
GNN_history = run_experiment(gnn_model, x_train_GNN, y_train_GNN)
display_learning_curve(GNN_history)

# test_data_GNN2 = pd.concat(test_set).sample(frac=1)
# x_test_GNN2 = test_data_GNN2.paper_id.to_numpy()
# y_test_GNN2 = test_data_GNN2["class_label"]

x_test_GNN = test_set_DataFrame["paper_id"]
y_test_GNN = test_set_DataFrame["class_label"]
_, test_accuracy_GNN = gnn_model.evaluate(x=x_test_GNN, y=y_test_GNN, verbose=0)
print("GNN test accuracy = ", test_accuracy_GNN)
        
        
        
            
        
    
    
        
        
        
        