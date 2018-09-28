import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN,self).__init__()
        self.hidden_size=hidden_size
        self.embed=nn.Embedding(vocab_size,embed_size)
        self.lstm=nn.LSTM(embed_size,hidden_size,num_layers,True)
        self.fc=nn.Linear(hidden_size,vocab_size)
    
    def forward(self, features, captions):
        batch_size=features.shape[0]
        captions_trim=captions[:,:-1]
        embed=self.embed(captions_trim)
        inputs=torch.cat([features.unsqueeze(1),embed],1)
        lstm_o,self.hidden=self.lstm(inputs)
        outputs=self.fc(lstm_o)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        t=[]
        for i in range(max_len):
            
            lstm_o,states=self.lstm(inputs,states)
            out=self.fc(lstm_o.squeeze(1))
            argmax=out.max(1)
            index=argmax[1].item()
            t.append(index)
            inputs=self.embed(argmax[1].long()).unsqueeeze(1)
            # 1 representative <end>
            if index==1:
                break
        return t