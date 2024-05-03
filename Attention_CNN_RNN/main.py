"""
Project: - ImageNarrative: From Conventional to Cutting Edge Captioning
Author: - Hussain Kanchwala
Start Date: - 04/1/24 End Date: - 04/22/24


Description: - 
This file contrains the CNN-RNN with Attention model and necessary steps to transform the raw data so as to able to learned by the model.
Added to that the file contains steps to analyze the trained model based on Rouge Scores and qualitatively assessing the predictions.
"""

#%% 
# IMPORT NECESSARY PACKAGES

import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset, random_split
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import  torch.optim as optim

import torchvision.models as models
import matplotlib.pyplot as plt
from rouge import Rouge
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import os
from PIL import Image
import pickle
#custom imports 
from data_formation import FlickrDataset,get_data_loader

#%%
# This section deals with vocabulary generation and creating dataloaders for train, test and validation datasets

image_location =  '/home/hussain/FInalize/Flicker/Images'
caption_file_location = '/home/hussain/FInalize/Flicker/captions.txt'
total_df = pd.read_csv(caption_file_location)

# Generate train, test and Validation data
grouped = total_df.groupby("image").agg(list).reset_index()
train,test_val=train_test_split(grouped,test_size=1000,random_state=42)
val, test = train_test_split(test_val,test_size=500,random_state=42)
main_test = test
main_train = train
main_val = val

train = train.explode("caption").reset_index(drop=True)
test = test.explode("caption").reset_index(drop=True)
val = val.explode("caption").reset_index(drop=True)

#setting the constants
BATCH_SIZE = 64
NUM_WORKER = 4

#defining the transform to be applied
transforms = T.Compose([
    T.Resize(226),                     
    T.RandomCrop(224),                 
    T.ToTensor(),                               
    T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])

# This is used for just vocabulary
dataset=FlickrDataset(image_location,total_df,transform=transforms,i_c_map=True)
#This is from where we will pull the values
mapping = dataset.vocab.dic

itos = dataset.vocab.itos
print("The vocabulary are: - ",itos,"\n")
stoi = dataset.vocab.stoi

# Train, test and validation
train_dataset=FlickrDataset(image_location,train,transform=transforms,i_c_map=True,itos=itos,stoi=stoi)
test_dataset=FlickrDataset(image_location,test,transform=transforms,i_c_map=True,itos=itos,stoi=stoi)
val_dataset=FlickrDataset(image_location,val,transform=transforms,i_c_map=True,itos=itos,stoi=stoi)

print(len(itos))
print(len(train_dataset.vocab.itos))
print(len(test_dataset.vocab.itos))
print(len(val_dataset.vocab.itos))

# Train dataloader
train_loader = get_data_loader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKER,
    shuffle=True,
    # batch_first=False
)

# Test dataloader
test_loader = get_data_loader(
    dataset=test_dataset,
    batch_size=32,
    num_workers=NUM_WORKER,
    shuffle=True,
    # batch_first=False
)

# Validation dataloader
val_loader = get_data_loader(
    dataset=val_dataset,
    batch_size=32,
    num_workers=NUM_WORKER,
    shuffle=True,
    # batch_first=False
)

#vocab_size
vocab_size = len(dataset.vocab)
print(vocab_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# %%
#show the tensor image
def show_image(img, title=None):
    """Imshow for Tensor."""
    
    #unnormalize 
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224 
    img[2] = img[2] * 0.225 
    img[0] += 0.485 
    img[1] += 0.456 
    img[2] += 0.406
    
    img = img.numpy().transpose((1, 2, 0))
    
    
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

#%%
# Encoder

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        

    def forward(self, images): #image of size (3,224,224)
        features = self.resnet(images)                                    #(batch_size,2048,7,7)
        features = features.permute(0, 2, 3, 1)                           #(batch_size,7,7,2048)
        features = features.view(features.size(0), -1, features.size(-1)) #(batch_size,49,2048)
        return features

#%%
# Attention
    
class Attention(nn.Module):
    def __init__(self,encoder_dim,decoder_dim,attention_dim):
        super(Attention,self).__init__()
        self.attentiion_dim = attention_dim
        self.W = nn.Linear(decoder_dim,attention_dim)
        self.U = nn.Linear(encoder_dim,attention_dim)
        self.A = nn.Linear(attention_dim,1)

    def forward(self,features,hidden_state):
        u_hs = self.U(features)     #(batch_size,num_layers,attention_dim)
        w_ah = self.W(hidden_state) #(batch_size,attention_dim)
        
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1)) #(batch_size,num_layers,attemtion_dim)
        
        attention_scores = self.A(combined_states)         #(batch_size,num_layers,1)
        attention_scores = attention_scores.squeeze(2)     #(batch_size,num_layers)
        
        
        alpha = F.softmax(attention_scores,dim=1)          #(batch_size,num_layers)
        
        attention_weights = features * alpha.unsqueeze(2)  #(batch_size,num_layers,features_dim)
        attention_weights = attention_weights.sum(dim=1)   #(batch_size,num_layers)
        
        return alpha,attention_weights
    
#%%
# Decoder RNN
    
class DecoderRNN(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,drop_prob=0.3):
        super().__init__()
        
        #save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.attention = Attention(encoder_dim,decoder_dim,attention_dim)
        
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        self.lstm_cell = nn.LSTMCell(embed_size+encoder_dim,decoder_dim,bias=True)
        self.f_beta = nn.Linear(decoder_dim,encoder_dim)
        self.fcn = nn.Linear(decoder_dim,vocab_size)
        self.drop = nn.Dropout(drop_prob)
    
    def forward(self, features, captions):
        
        #vectorize the caption
        embeds = self.embedding(captions)
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        #get the seq length to iterate
        seq_length = len(captions[0])-1 #Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)
        
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length,num_features).to(device)

        for s in range(seq_length):
            alpha,context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
                    
            output = self.fcn(self.drop(h))
            
            preds[:,s] = output
            alphas[:,s] = alpha  
        
        return preds, alphas
    
    def generate_caption(self,features,max_len=30,vocab=None):
        # Inference part
        # Given the image features generate the captions
        
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        alphas = []
        
        #starting input
        word = torch.tensor(vocab.stoi['<SOS>']).view(1,-1).to(device)
        embeds = self.embedding(word)

        #Addition
        captions = []


        for i in range(max_len):
            alpha,context = self.attention(features, h)
            
            
            #store the apla score
            alphas.append(alpha.cpu().detach().numpy())
            
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size,-1)
                        
            #select the word with most val
            predicted_word_idx = output.argmax(dim=1)
            
            #end if <EOS detected>
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break

            #save the generated word
            captions.append(predicted_word_idx.item())
            
            #send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))
        
        #covert the vocab idx to words and return sentence
        return [vocab.itos[idx] for idx in captions],alphas
    
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c
    
#%%
# Model
    
class EncoderDecoder(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,drop_prob=0.3):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size = vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim
        )
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

#%%
# Hyperparams
    
embed_size=300
vocab_size = len(dataset.vocab)
print(vocab_size)
attention_dim=256
encoder_dim=2048
decoder_dim=512
learning_rate = 3e-4

#%%
#init model

model = EncoderDecoder(
    embed_size=300,
    vocab_size = len(dataset.vocab),
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#%%
#helper function to save the model

def save_model(model,num_epochs):
    model_state = {
        'num_epochs':num_epochs,
        'embed_size':embed_size,
        'vocab_size':len(dataset.vocab),
        'attention_dim':attention_dim,
        'encoder_dim':encoder_dim,
        'decoder_dim':decoder_dim,
        'state_dict':model.state_dict()
    }

    torch.save(model_state,'checking_model_state.pth')

#%% 
# Rouge score Calculation function
    
def calculate_max_rouge_score(generated_caption, true_caption):
    rouge = Rouge()
    score = rouge.get_scores(generated_caption, true_caption)[0]['rouge-l']['f']  # Consider only the f-score of Rouge-L
    return score

#%%
# This section deals with training the model
# Training the model
num_epochs = 100
train_loss=[]
val_loss = []
val_rouge = []

for epoch in range(1,num_epochs+1):
    avg_train_loss = 0
    model.train()
    for idx, (image, captions) in enumerate(iter(train_loader)):
        image,captions = image.to(device),captions.to(device)

        # Zero the gradients.
        optimizer.zero_grad()

        # Feed forward
        outputs,attentions = model(image, captions)

        # Calculate the batch loss.
        targets = captions[:,1:]
        loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
        
        # Total loss across epoch
        avg_train_loss+=loss.item()

        # Backward pass.
        loss.backward()

        # Update the parameters in the optimizer.
        optimizer.step()
    
    # Train Loss        
    train_loss.append(avg_train_loss/len(train_loader))
    
    # Model in Evaluation Mode
    model.eval()
    with torch.no_grad():
        # Validation Loss Calculation
        total_val_loss = 0
        for idx,(img, captions) in enumerate(iter(val_loader)):
            img, captions = img.to(device), captions.to(device)
            outputs, _ = model(img, captions)
            targets = captions[:, 1:]
            loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
            total_val_loss += loss.item()
        val_loss.append(total_val_loss/len(val_loader))
        print("Validation calculation completed","\n")


        # Assuming that the batch size is at least 30 or adjust accordingly
        score = 0
        imgs, captions = next(iter(val_loader))
        for i in range(0,imgs.shape[0]): 
            img = imgs[i:i+1].to(device)
            feature = model.encoder(img)
            caps, alphas = model.decoder.generate_caption(feature, vocab=dataset.vocab)
            t_caption = ' '.join([dataset.vocab.itos[idx] for idx in captions[i].tolist() if dataset.vocab.itos[idx] not in ['<PAD>', '<SOS>', '<EOS>']])
            caption = ' '.join(caps)
            score = max(score,calculate_max_rouge_score(caption, t_caption))
            #print("True Caption:", t_caption)  # Debug: Check the actual true caption
            #print("Generated Caption:", caption)  # Debug: Check the generated caption            
        val_rouge.append(score)

        # Show a Image along with true and generated Caption
        t_caption = ' '.join([dataset.vocab.itos[idx] for idx in captions[0].tolist() if dataset.vocab.itos[idx] not in ['<PAD>', '<SOS>', '<EOS>']]) 
        print(t_caption)
        featue = model.encoder(imgs[0:1].to(device))
        caps,_ = model.decoder.generate_caption(featue,vocab=dataset.vocab)
        generated_caption = ' '.join(caps)
        show_image(imgs[0],generated_caption)

    print("Epoch: {} train_loss: {:.5f} val_loss: {:.5f} val_RougeF1: {:.2f}".format(epoch,train_loss[-1],val_loss[-1],val_rouge[-1]))
    #save the latest model
    save_model(model,epoch)
    model.train()


#%%

# SAVING THE LISTS
print(len(train_loss))
print(len(val_loss))
print(len(val_rouge))

# Save to a binary file in NumPy `.npy` format
np.save('/home/hussain/FInalize/media/train_loss.npy', train_loss)
np.save('/home/hussain/FInalize/media/val_loss.npy', val_loss)
np.save("/home/hussain/FInalize/media/val_rouge.npy",val_rouge)
main_train.to_csv('train.csv',index=False)
main_test.to_csv('test.csv',index=False)
main_val.to_csv('val.csv',index=False)

#%%
# Plotting function

def plot(epoches,y,title,y_label):
    plt.figure(figsize=(10,5))
    plt.plot(range(epoches),y)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()

#%%
# To load the arrays back
loaded_train_loss = np.load('/home/hussain/FInalize/media/train_loss.npy')
loaded_val_loss = np.load('/home/hussain/FInalize/media/val_loss.npy')
loaded_val_rouge = np.load('/home/hussain/FInalize/media/val_rouge.npy')

plot(100,loaded_train_loss,"Training Loss Over Epochs","Loss")
plot(100,loaded_val_loss,"Validation Loss Over Epoches","Loss")
plot(100,loaded_val_rouge,"Validation Rouge-L1 F1 score","Rouge-L1 score")

#%%
# Load the model in evaluation mode
model = EncoderDecoder(
    embed_size=300,
    vocab_size = len(dataset.vocab),
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512
).to(device)

model_path = 'checking_model_state.pth'
model_state = torch.load(model_path)
model.load_state_dict(model_state['state_dict'])
model.eval()

#%%
# Test Loss
model.eval()
total_test_loss = 0
for idx,(img, captions) in enumerate(iter(test_loader)):
    img, captions = img.to(device), captions.to(device)
    outputs, _ = model(img, captions)
    targets = captions[:, 1:]
    loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
    total_test_loss += loss.item()
test_loss = total_test_loss/len(test_loader)
print("The test loss is: - ",test_loss,"\n")

#%%

def calculate_max_rouge_score(generated_caption, true_captions):
    rouge = Rouge()
    max_rouge_score = 0
    for true_caption in true_captions:
        caption_vec = []
        caption_vec += dataset.vocab.numericalize(true_caption)
        true_caption = ' '.join([dataset.vocab.itos[idx] for idx in caption_vec if dataset.vocab.itos[idx] not in ['<PAD>', '<SOS>', '<EOS>']])
        scores = rouge.get_scores(generated_caption, true_caption)[0]['rouge-l']['f']  # Consider only the f-score of Rouge-L
        print("The true caption is: ",true_caption,"\n")
        print("The generated caption  is: ", generated_caption,"\n")
        if scores > max_rouge_score:
            max_rouge_score = scores
    return max_rouge_score

#%%
# Testing Rouge Score
te_rouge_lst = []
te_rouge_info = []
for index in range(len(main_test)):
    test_data=main_test.iloc[index]
    test_img_name = test_data[0]    
    test_true_caption = mapping[test_img_name]                
    test_img_location = os.path.join("/home/hussain/FInalize/Flicker/Images",test_img_name)
    test_img = Image.open(test_img_location).convert("RGB")                
    test_img = transforms(test_img)
    test_img = test_img.unsqueeze(0)
    test_features = model.encoder(test_img.to(device))
    test_caps,test_alphas = model.decoder.generate_caption(test_features,vocab=dataset.vocab)
    test_caption = ' '.join(test_caps)
    rouge_score=calculate_max_rouge_score(test_caption,test_true_caption)
    te_rouge_lst.append(rouge_score)
    te_rouge_info.append((rouge_score, test_img_name, test_caption))

#%%
# Testing Rouge score
plot(500,te_rouge_lst,"Testing Rouge-L1 F1 over Epochs","Rouge-L1 F1")

#%%
# Top 10 Predictions
te_rouge_info.sort(reverse=True, key=lambda x: x[0])
top_10_images = te_rouge_info[:10]
for score,img_name,caption in top_10_images:
    test_img_location = os.path.join("/home/hussain/FInalize/Flicker/Images",img_name)
    img = Image.open(test_img_location).convert("RGB")
    title = f"Prediction: - {caption} | ROUGE Score: - {score:.2f}"
    plt.imshow(img)
    plt.title(title)
    plt.show()

#%%
# Worst 10 Predictions
bottom_10_images = te_rouge_info[-10:]
for score,img_name,caption in bottom_10_images:
    test_img_location = os.path.join("/home/hussain/FInalize/Flicker/Images",img_name)
    img = Image.open(test_img_location).convert("RGB")
    title = f"Prediction: - {caption} | ROUGE Score: - {score:.2f}"
    plt.imshow(img)
    plt.title(title)
    plt.show()

#%%
# ITOS AND STOI SAVE FOR FUTURE
import pickle
with open('/home/hussain/FInalize/media/itos.pkl','wb') as f:
    pickle.dump(itos,f)
with open('/home/hussain/FInalize/media/stoi.pkl','wb') as f:
    pickle.dump(stoi,f)

#%%
# Trail load for check
with open('/home/hussain/FInalize/media/itos.pkl','rb') as f:
    try_load = pickle.load(f)
print(len(try_load))
print(try_load[0])


#%%
from collections import Counter
import spacy
spacy_english = spacy.load("en_core_web_sm")
# Function to plot frequent words
def plot_top_frequent_words(caption_list,title,top_count=50):
    words=[]
    for caption in caption_list:
        caption_tokenized=[token.text.lower() for token in spacy_english.tokenizer(caption)]
        words.extend(caption_tokenized)
    total_freq=Counter(words)
    total_freq=total_freq.most_common(top_count)
    words, frequencies = zip(*total_freq)
    plt.figure(figsize=(12, 6))
    plt.bar(words, frequencies)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.show()

def calculate_rouge_score(image_captions,predicted_caption):
    """
    Function which calculates the best the rouge-l f-score  for a given list of image captions against the predicted caption
    """

    rouge=Rouge()
    caption_score_map={}
    for caption in image_captions:
        caption_score_map[caption]=rouge.get_scores(predicted_caption, caption)[0]['rouge-l']['f']
    best_match=max(caption_score_map,key=caption_score_map.get)
    best_score=round(caption_score_map[best_match],3)
    return (best_match,best_score)

#%%
# Plotting top 50 frequent words in test datat and predictions

test = pd.read_csv('/home/hussain/FInalize/media/test.csv')
test = test.explode("caption").reset_index(drop=True)
image_location =  '/home/hussain/FInalize/Flicker/Images'
transforms = T.Compose([
    T.Resize(226),                     
    T.RandomCrop(224),                 
    T.ToTensor(),                               
    T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])
with open('/home/hussain/FInalize/media/itos.pkl','rb') as f:
    itos = pickle.load(f)
with open('/home/hussain/FInalize/media/stoi.pkl','rb') as f:
    stoi = pickle.load(f)
test_dataset=FlickrDataset(image_location,test,transform=transforms,i_c_map=True,itos=itos,stoi=stoi)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EncoderDecoder(
    embed_size=300,
    vocab_size = len(itos),
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512
).to(device)
model_path = 'checking_model_state.pth'
model_state = torch.load(model_path)
model.load_state_dict(model_state['state_dict'])
model.eval()

test_idx=0
test_images=test_dataset.imgs
test_captions=test_dataset.captions
rouge_score_map={}
while test_idx <len(test_captions):
    image_captions=test_captions[test_idx:test_idx+5]
    test_img = Image.open(os.path.join(image_location,test_images[test_idx])).convert("RGB")                
    test_img = transforms(test_img)
    transform_image = test_img.unsqueeze(0)
    test_features = model.encoder(transform_image.to(device))
    test_caps,test_alphas = model.decoder.generate_caption(test_features,vocab=test_dataset.vocab)
    test_caption = ' '.join(test_caps)
    best_caption,best_score=calculate_rouge_score(image_captions,test_caption)
    rouge_score_map[test_images[test_idx]]=(best_caption,test_caption,best_score)
    test_idx+=5
print(len(rouge_score_map))

predicted_captions=[score_map[1] for score_map in rouge_score_map.values()]
prediction_title="Top 50 Frequent words in the predicted captions"
plot_top_frequent_words(predicted_captions,prediction_title)

test_captions=[score_map[0] for score_map in rouge_score_map.values()]
test_title="Top 50 Frequent words in the best matching test captions"
plot_top_frequent_words(test_captions,test_title)
