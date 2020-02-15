#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchvision import utils
# from basic_fcn import *
from model_to_run import *
from dataloader import *
from utils import *
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
from matplotlib import pyplot as plt
import os
from pprint import pprint
import sys


# In[ ]:





# In[2]:


train_dataset = CityScapesDataset(csv_file='./Data/train.csv')
val_dataset = CityScapesDataset(csv_file='./Data/val.csv')
test_dataset = CityScapesDataset(csv_file='./Data/test.csv')


_num_workers = 4
iou_classes = [11, 20, 24, 26, 33]
epochs     = 1000


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          num_workers=_num_workers,
                          shuffle=True)
val_loader = DataLoader(dataset=val_dataset,
                          batch_size=batch_size,
                          num_workers=_num_workers,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=1,
                          num_workers=_num_workers,
                          shuffle=False)



# In[3]:


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        #torch.nn.init.xavier_uniform(m.bias.data)
        m.bias.data.zero_()
        
        
criterion = nn.CrossEntropyLoss()# Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
model = modelToRun
# for name, params in fcn_model.named_parameters():
#     print(name, params.shape, params.requires_grad)
    
model.apply(init_weights)
#fcn_model = torch.load('best_model')

optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# In[4]:


modelName = modelNameSpace
MYDIR = "./Results" + "/" + modelName
CHECK_FOLDER = os.path.isdir(MYDIR)
if not CHECK_FOLDER:
    os.makedirs(MYDIR)
    print("created folder : ", MYDIR)
else:
    print(MYDIR, "folder already exists.")

def plotTwoGraphs(base, train_metrics, val_metrics):
    plt.figure()
    plt.plot(base, train_metrics)
    plt.plot(base, val_metrics)
    plt.gca().legend(('train','validation'))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.savefig(MYDIR + '/loss.png')

def plotGraph(base, metrics, name):
    plt.figure()
    plt.plot(base, metrics)
    plt.xlabel('Epochs')
    plt.ylabel(name)
    plt.title(name + ' vs Epochs')
    plt.savefig(MYDIR + '/' + name + '.png')
    
def plotIousClassWise(base, d):
    plt.figure()
    legends = []
    for c in iou_classes:
        plt.plot(base, d[c])
        legends.append(idsNames[c])
    plt.gca().legend(tuple(legends))
    plt.xlabel('Epochs')
    plt.ylabel('IOU Class Wise')
    plt.title('Loss vs Class Wise IOU')
    plt.savefig(MYDIR + '/class_wise_ious.png')
        #plotGraph([i for i in range(epoch+1)], all_val_iou_class[c], 'iou_'+idsNames[c]+'')
    
    


# In[ ]:

loaded_cls_weights = 100*torch.from_numpy(np.load('class_weigths.npz')['class_weights'])
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()
    loaded_cls_weights = loaded_cls_weights.cuda()
    
all_train_losses = []
all_val_losses = []
all_val_accuracies = []
all_val_iou = []
all_val_iou_class = {}
startingValLoss = 10000000.0
curEpoch = 0
epochList = []

        
def train():
    global curEpoch
    global startingValLoss
    while(curEpoch < epochs):
        model.train()
        curEpochLoss = []
        ts = time.time()
        for iter, (X, tar, Y) in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = X.cuda()   # Move your inputs onto the gpu
                labels = Y.cuda() # Move your labels onto the gpu
                targets_onehot = tar.cuda()
            else:
                inputs, labels, targets_onehot = X, Y, tar# Unpack variables into inputs and labels
            outputs = model(inputs)
            if weighted_loss:
#                 loss = weighted_ce_loss(outputs, targets_onehot, loaded_cls_weights, weighted=weighted_loss) + dice_loss(outputs, targets_onehot, loaded_cls_weights, weighted=weighted_loss)
                loss = weighted_ce_loss(outputs, targets_onehot, loaded_cls_weights, weighted=weighted_loss)
            else:
                loss = criterion(outputs, labels)
            
            curEpochLoss.append(loss.item())
            loss.backward()
            optimizer.step()

            if iter % 100 == 0:
                print("epoch{}, iter{}, loss: {}".format(curEpoch, iter, loss.item(), time.time() - ts))
        curEpochLoss = sum(curEpochLoss)/len(curEpochLoss)
        print('training loss : ', curEpochLoss)
        print("Finish epoch {}, time elapsed {}".format(curEpoch, time.time() - ts))
        
        if(curEpoch % 4 == 0):
            epochList.append(curEpoch)
            all_train_losses.append(curEpochLoss)
            val_loss, val_accuracy, val_iou_net, val_iou_net_class_wise = val(curEpoch)
            all_val_losses.append(val_loss)
            all_val_accuracies.append(val_accuracy)
            all_val_iou.append(val_iou_net)
            for c in val_iou_net_class_wise:
                if(c not in all_val_iou_class):
                    all_val_iou_class[c] = []
                all_val_iou_class[c].append(val_iou_net_class_wise[c])

            plotTwoGraphs(epochList, all_train_losses, all_val_losses)
            plotGraph(epochList, all_val_accuracies, 'validation_accuracy')
            plotGraph(epochList, all_val_iou, 'average_iou')
            plotIousClassWise(epochList, all_val_iou_class)

            if(val_loss < startingValLoss):
                startingValLoss = val_loss
                results = {}
                results['accuracy'] = val_accuracy
                results['iou_net'] = val_iou_net
                results['iou_net_class_wise'] = val_iou_net_class_wise
                results['total_loss'] = val_loss
                results['epoch'] = curEpoch
                results['epoch_list'] = epochList
                with open(MYDIR + '/results_val.txt', 'wt') as out:
                    pprint(results, stream=out)
                torch.save({'model_state_dict' : model.state_dict()}, MYDIR + '/best_model')
                test(1)
        
        torch.save({
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'data' : {
                'all_train_losses' : all_train_losses,
                'all_val_accuracies' : all_val_accuracies,
                'all_val_losses' : all_val_losses,
                'all_val_iou' : all_val_iou,
                'all_val_iou_class' : all_val_iou_class,
                'epoch' : curEpoch,
                'starting_val_loss' : startingValLoss,
                'epoch_list' : epochList
            }
        }, MYDIR+'/'+'model_params')
        
        print('model saved for later use')
        
        curEpoch += 1
    


def val(epoch):
    model.eval()
    #Complete this function - Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model
    total_loss = []
    acc_num = 0
    acc_den = 0
    #iou_num = len(iou_classes)*[0] 
    #iou_den = len(iou_classes)*[0] 
    iou_num = {}
    iou_den = {}
    ts = time.time()
    for iter, (X_val, tar_val, Y_val) in enumerate(val_loader):
        if use_gpu:
            inputs_val = X_val.cuda()   # Move your inputs onto the gpu
            labels_val = Y_val.cuda() # Move your labels onto the gpu
            tar_val = tar_val.cuda()
        else:
            inputs_val, labels_val, tar_val = X_val, Y_val, tar_val# Unpack variables into inputs and labels

        outputs_val = model(inputs_val)
        
        _, preds = torch.max(outputs_val, 1)
        
        if weighted_loss:
#             loss_val = weighted_ce_loss(outputs_val, tar_val, loaded_cls_weights, weighted=weighted_loss) + dice_loss(outputs_val, tar_val, loaded_cls_weights, weighted=weighted_loss)
            loss_val = weighted_ce_loss(outputs_val, tar_val, loaded_cls_weights, weighted=weighted_loss)
        else:
            loss_val = criterion(outputs_val, labels_val)
        
        preds_cpu = preds.cpu()
        Y_cpu = Y_val.cpu()
        
        total_loss.append(loss_val.item())
        
        nn, dd = pixel_acc(preds_cpu, Y_cpu)
        acc_num += nn
        acc_den += dd
        
        for iou_cc_idx in range(len(useful_ids)):
            i_nn, i_dd = iou(preds_cpu, Y_cpu, useful_ids[iou_cc_idx])
            if useful_ids[iou_cc_idx] not in iou_num:
                iou_num[useful_ids[iou_cc_idx]] = 0
                iou_den[useful_ids[iou_cc_idx]] = 0
            iou_num[useful_ids[iou_cc_idx]] += i_nn
            iou_den[useful_ids[iou_cc_idx]] += i_dd
            
        del inputs_val
        del labels_val
        del loss_val
    total_loss = sum(total_loss)/len(total_loss)
    print("Validation loss after epoch {}, , loss: {}".format(epoch, total_loss))
    print("Accuracy on val after epoch {}, accuracy: {}".format(epoch, 100*acc_num/acc_den))
    for iou_cc_idx in range(len(iou_classes)):
        print("IOU for class {} on val after epoch{}, iou: {}".format(iou_classes[iou_cc_idx], epoch, 100*iou_num[iou_classes[iou_cc_idx]]/iou_den[iou_classes[iou_cc_idx]]))

    iou_net_class_wise = dict((c, iou_num[c]/iou_den[c]) for c in iou_num)
    print(iou_net_class_wise)
    accuracy = acc_num / acc_den
    iou_net_num = 0
    iou_net_den = 0
    for c in iou_num:
        iou_net_num += iou_num[c]
        iou_net_den += iou_den[c]
    iou_net = iou_net_num/iou_net_den
    print("Validation epoch {}, time elapsed {}".format(epoch, time.time() - ts))
    
    return total_loss, accuracy, iou_net, iou_net_class_wise

def colorMap(a, i):
    return idsColor[a][i]

def test(numFigures):
    #fcn_model.load_state_dict(torch.load(MYDIR + '/best_model'))
    #model = torch.load(MYDIR + '/best_model')
    #model.cuda()
    model.eval()
    dataiter = iter(test_loader)
    i = 0
    while(i < numFigures):
        i += 1
        X_test, img_full, Y_test = dataiter.next()
        if use_gpu:
            inputs_test = X_test.cuda()   # Move your inputs onto the gpu
            labels_test = Y_test.cuda() # Move your labels onto the gpu
            #img_full_test = img_full.cuda()
        else:
            inputs_test, labels_test = X_test, Y_test# Unpack variables into inputs and labels
            #img_full_test = img_full

        outputs = model(inputs_test)
        _, preds = torch.max(outputs, 1)
        inputs_test = inputs_test.permute(0, 2, 3, 1)
        #img_full_test = img_full_test.permute(0, 2, 3, 1)

        f = plt.figure(figsize=(16,8))
        ax = f.add_subplot(121)
        ax2 = f.add_subplot(122)

        #plt.subplot(1, 2, 1)
        ax.imshow(torchvision.utils.make_grid(-1*inputs_test.cpu()))
        


        preds = preds.cpu().numpy()

        preds0 = np.vectorize(colorMap)(preds, 0)
        preds1 = np.vectorize(colorMap)(preds, 1)
        preds2 = np.vectorize(colorMap)(preds, 2)

        preds = torch.from_numpy(np.transpose(np.concatenate((preds0, preds1, preds2), 0), (1,2,0)))

        #plt.subplot(1, 2, 2)
        ax2.imshow(torchvision.utils.make_grid(preds.cpu()))
        
        plt.savefig(MYDIR+'/test.png',dpi=300)
    
    
    
    
    
    
    
    
#     X_test = X_test.permute(0, 2, 3, 1)
#     plt.imshow(torchvision.utils.make_grid(-1*X_test))
#     outputs = model(X_test)
#     _, preds = torch.max(outputs, 1)
#     print(preds.Size())
    
        
        
        
    
    
              
    
if __name__ == "__main__":
    #test(10)
    try:
        checkpoint = torch.load(MYDIR+'/'+'model_params')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #optimizer.cuda()
        all_train_losses = checkpoint['data']['all_train_losses']
        all_val_losses = checkpoint['data']['all_val_losses']
        all_val_accuracies = checkpoint['data']['all_val_accuracies']
        all_val_iou = checkpoint['data']['all_val_iou']
        all_val_iou_class = checkpoint['data']['all_val_iou_class']
        startingValLoss = checkpoint['data']['starting_val_loss']
        curEpoch = checkpoint['data']['epoch'] + 1
        epochList = checkpoint['data']['epoch_list']
        print('starting with epoch : ', curEpoch)
    except:
        print('here')
        pass
    print(useful_ids)
    train()
    
    
    #train()


# In[ ]:




