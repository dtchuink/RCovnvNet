'''
Created on Oct 2, 2018

@author: danielle
'''

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from datasets import PartDataset
from lr_scheduler import ReduceLROnPlateau
from tensorflow.contrib.batching.ops.gen_batch_ops import batch

# Hyper Parameters
input_size = 2500
hidden_size = 1800
num_layers = 1
#num_classes = 10
batch_size = 32
test_batch_size=10
num_epochs = 5
learning_rate = 0.0001
hidden_size2 = 1048

blue = lambda x:'\033[94m' + x + '\033[0m'

train_dataset = PartDataset(root = '/home/danielle/pyDevelopment/lstm-rnn/src/ln_lstm/shapenetcore_partanno_segmentation_benchmark_v0', classification = False, class_choice = ['Chair'])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

test_dataset = PartDataset(root = '/home/danielle/pyDevelopment/lstm-rnn/src/ln_lstm/shapenetcore_partanno_segmentation_benchmark_v0', classification = False, train = False, class_choice = ['Chair'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

print(len(train_dataset), len(test_dataset))
num_classes = train_dataset.num_seg_classes
print('classes', num_classes)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_size2=hidden_size2

        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=0.05, batch_first=True)
        self.c1 = nn.Conv1d(3, 64, 1)
        self.c2 = nn.Conv1d(64, 128, 1)
        self.c3 = nn.Conv1d(128, 256, 1)
        self.c4 = nn.Conv1d(256, hidden_size2, 1)
        self.mp1 = nn.AdaptiveMaxPool1d(1)
        self.l1 = nn.Linear(hidden_size2, num_classes)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(hidden_size2)

    def forward(self, x, hidden):
        
        output, hidden = self.gru(x, hidden)
        x, hidden = self.gru2(output, hidden)
        x = F.relu(self.bn1(self.c1(x)))
        x = F.relu(self.bn2(self.c2(x)))
        x = F.relu(self.bn3(self.c3(x)))
        x = F.relu(self.bn4(self.c4(x)))
        x = self.mp1(x)
        x = x.view(-1, self.hidden_size2)
        x = self.l1(x)

#         iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batch_size,1)
#         if x.is_cuda:
#             iden = iden.cuda()
#         x = x + iden
#         x = x.view(-1, 3, 3)
        return F.log_softmax(x), hidden


class sRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, k=2):
        super(sRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_size2=hidden_size2
        self.k = k

        self.gru = nn.GRU(input_size, hidden_size, num_layers=n_layers, dropout=0.05, batch_first=True)
        self.c1 = nn.Conv1d(3, 64, 1)
        self.c2 = nn.Conv1d(64, 128, 1)
        self.c3 = nn.Conv1d(128, 256, 1)
        self.c4 = nn.Conv1d(256, hidden_size, 1)
        self.mp1 = nn.AdaptiveMaxPool1d(1)
        self.l1 = nn.Linear(hidden_size, num_classes)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(hidden_size)

    def forward(self, x, hidden):
        
        output, hidden = self.gru(x, hidden)
        x, hidden = self.gru2(output, hidden)
        x = F.relu(self.bn1(self.c1(x)))
        x = F.relu(self.bn2(self.c2(x)))
        x = F.relu(self.bn3(self.c3(x)))
        x = F.relu(self.bn4(self.c4(x)))
        
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1, num_classes))
        x = x.view(batch_size, self.num_points, num_classes)
        return x, hidden

#         iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batch_size,1)
#         if x.is_cuda:
#             iden = iden.cuda()
#         x = x + iden
#         x = x.view(-1, 3, 3)
#         return F.log_softmax(x), hidden

rnn = sRNN(input_size, hidden_size, output_size = num_classes, n_layers=num_layers)
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.0001, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'max') # set up scheduler
accuracy = 0
iteration = 0

for epoch in range(num_epochs):
    print("epoch=", epoch)
    for i, points in enumerate(train_loader, 0):
        data, target = points
        data, target = Variable(data), Variable(target[:,0])
        data = data.transpose(2,1)

        optimizer.zero_grad()
        pred, hidden = rnn(points, None)
        pred = pred.view(-1, num_classes)
        target = target.view(-1,1)[:,0] - 1
        #print(pred.size(), target.size())
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, batch_size, loss.data[0], correct/float(batch_size * 2500)))
        
        
#         print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
#                    %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
#         
        #Testing each epoch
        if i % 10 == 0:
            j, data = enumerate(test_loader, 0).__next__()
            points, target = data
            points, target = Variable(points), Variable(target)
            points = points.transpose(2,1) 
            pred, _ = rnn(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1,1)[:,0] - 1

            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            accuracy =  accuracy + correct/float(batch_size * 2500)
            iteration = iteration +1
            print('[%d: %d/%d] %s loss: %f accuracy: %f  avg_accuracy' %(epoch, i, batch_size, blue('test'), loss.data[0], correct/float(batch_size * 2500), accuracy/iteration))

# Test the Model
correct = 0
total = 0
i =0
for point_set, cls in test_loader:
    point_set = Variable(point_set)
    point_set = point_set.transpose(2,1)
    outputs, hidden = rnn(point_set, None)
    _, predicted = torch.max(outputs.data, 1)
    total += cls.size(0)
    correct += (predicted == cls[:,0]).sum()
    print("Test data %d correct %d, Total %d"% (i, correct, total))
 
print('Test Accuracy of the model on the test data: %d %%' % (100 * correct / total)) 




