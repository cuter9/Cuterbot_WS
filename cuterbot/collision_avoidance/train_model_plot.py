#!/usr/bin/env python
# coding: utf-8

# # Collision Avoidance - Train Model (with live graph)
# 
# Welcome to this host side Jupyter Notebook!  This should look familiar if you ran through the notebooks that run on the robot.  In this notebook we'll train our image classifier to detect two classes
# ``free`` and ``blocked``, which we'll use for avoiding collisions.  For this, we'll use a popular deep learning library *PyTorch*

# In[ ]:


import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


# ### Upload and extract dataset
# 
# Before you start, you should upload the ``dataset.zip`` file that you created in the ``data_collection.ipynb`` notebook on the robot.
# 
# You should then extract this dataset by calling the command below

# In[ ]:


# get_ipython().system('unzip -q dataset.zip')


# You should see a folder named ``dataset`` appear in the file browser.

# ### Create dataset instance

# Now we use the ``ImageFolder`` dataset class available with the ``torchvision.datasets`` package.  We attach transforms from the ``torchvision.transforms`` package to prepare the data for training.  

# In[ ]:


dataset = datasets.ImageFolder(
    'dataset',
    transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)


# ### Split dataset into train and test sets

# Next, we split the dataset into *training* and *test* sets.  The test set will be used to verify the accuracy of the model we train.

# In[ ]:


train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 50, 50])


# ### Create data loaders to load data in batches

# We'll create two ``DataLoader`` instances, which provide utilities for shuffling data, producing *batches* of images, and loading the samples in parallel with multiple workers.

# In[ ]:


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)


# ### Define the neural network
# 
# Now, we define the neural network we'll be training.  The *torchvision* package provides a collection of pre-trained models that we can use.
# 
# In a process called *transfer learning*, we can repurpose a pre-trained model (trained on millions of images) for a new task that has possibly much less data available.
# 
# Important features that were learned in the original training of the pre-trained model are re-usable for the new task.  We'll use the ``alexnet`` model.

# In[ ]:


model = models.alexnet(pretrained=True)


# The ``alexnet`` model was originally trained for a dataset that had 1000 class labels, but our dataset only has two class labels!  We'll replace
# the final layer with a new, untrained layer that has only two outputs.  

# In[ ]:


model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)


# Finally, we transfer our model for execution on the GPU

# In[ ]:


device = torch.device('cuda')
model = model.to(device)


# ### Visualization utilities
# 
# Execute the cell below to enable live plotting. 
# 
# > You need to install bokeh (https://docs.bokeh.org/en/latest/docs/installation.html)
# 
# ```bash
# sudo pip3 install bokeh
# sudo jupyter labextension install @jupyter-widgets/jupyterlab-manager
# sudo jupyter labextension install @bokeh/jupyter_bokeh
# ```

# In[ ]:


from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models.tickers import SingleIntervalTicker
output_notebook()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

p1 = figure(title="Loss", x_axis_label="Epoch", plot_height=300, plot_width=360)
p2 = figure(title="Accuracy", x_axis_label="Epoch", plot_height=300, plot_width=360)

source1 = ColumnDataSource(data={'epochs': [], 'trainlosses': [], 'testlosses': [] })
source2 = ColumnDataSource(data={'epochs': [], 'train_accuracies': [], 'test_accuracies': []})

#r = p1.multi_line(ys=['trainlosses', 'testlosses'], xs='epochs', color=colors, alpha=0.8, legend_label=['Training','Test'], source=source)
r1 = p1.line(x='epochs', y='trainlosses', line_width=2, color=colors[0], alpha=0.8, legend_label="Train", source=source1)
r2 = p1.line(x='epochs', y='testlosses', line_width=2, color=colors[1], alpha=0.8, legend_label="Test", source=source1)

r3 = p2.line(x='epochs', y='train_accuracies', line_width=2, color=colors[0], alpha=0.8, legend_label="Train", source=source2)
r4 = p2.line(x='epochs', y='test_accuracies', line_width=2, color=colors[1], alpha=0.8, legend_label="Test", source=source2)

p1.legend.location = "top_right"
p1.legend.click_policy="hide"

p2.legend.location = "bottom_right"
p2.legend.click_policy="hide"


# ### Train the neural network
# 
# Using the code below we will train the neural network for 30 epochs, saving the best performing model after each epoch.
# 
# > An epoch is a full run through our data.

# In[ ]:


NUM_EPOCHS = 30
BEST_MODEL_PATH = 'best_model.pth'
best_accuracy = 0.0

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

handle = show(row(p1, p2), notebook_handle=True)

for epoch in range(NUM_EPOCHS):
    
    train_loss = 0.0
    train_error_count = 0.0
    for images, labels in iter(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        train_loss += loss
        train_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    
    test_loss = 0.0
    test_error_count = 0.0
    for images, labels in iter(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        test_loss += loss
        test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
    test_loss /= len(test_loader)
    
    train_accuracy = 1.0 - float(train_error_count) / float(len(train_dataset))
    test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
    print('%d: %f, %f, %f, %f' % (epoch+1, train_loss, test_loss, train_accuracy, test_accuracy))
    
    
    new_data1 = {'epochs': [epoch+1],
                 'trainlosses': [float(train_loss)],
                 'testlosses': [float(test_loss)] }
    source1.stream(new_data1)
    new_data2 = {'epochs': [epoch+1],
                 'train_accuracies': [float(train_accuracy)],
                 'test_accuracies': [float(test_accuracy)] }
    source2.stream(new_data2)
    push_notebook(handle=handle)
    
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_accuracy = test_accuracy


# Once that is finished, you should see a file ``best_model.pth`` in the Jupyter Lab file browser.  Select ``Right click`` -> ``Download`` to download the model to your workstation

# In[ ]:




