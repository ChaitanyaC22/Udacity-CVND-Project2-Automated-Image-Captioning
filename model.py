import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size) 

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)   
        ## features.size(0) is the batch_size. So the output of each image is a flattened vector 
        ## In this case, for a batch of images, (batch_size, size of resnet.fc.in_feautures)
        
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        """
        embed_size - dimensionality of the image and word embeddings
        hidden_size -  number of features in the hidden state of the RNN decoder.
        vocab_size - vocabulary size
        num_layers - number of recurrent layers
        """
        
        # Different sizes
        self.embed_size= embed_size
        self.hidden_size= hidden_size
        self.vocab_size= vocab_size
        self.num_layers= num_layers
        
        # 1. Embedding Layer 
        # Creating an embedding i.e. A simple lookup table that stores embeddings of a fixed dictionary and size.
        """
        Input: num_embeddings- size of the dictionary of embeddings and 
        Output: embedding_dim: size of embedding vector
        """
        self.embed= nn.Embedding(num_embeddings= self.vocab_size, embedding_dim= self.embed_size)
        
        # 2. Create a LSTM layer: Passing the output of embedding layer to the input of LSTM unit(s)
        self.lstm = nn.LSTM(input_size= self.embed_size, 
                            hidden_size= self.hidden_size, 
                            batch_first = True,    ## Inputs and outputs in (batch, sequence/caption, feature/embed_size) format
                            num_layers= self.num_layers)
      
        # 3. Final Fully_Connected Layer (Linear)- Input: hidden_state output from LSTM, Output: vocab_size
        self.fc_output = nn.Linear(in_features= self.hidden_size, out_features= self.vocab_size) 
        
      
    def forward(self, features, captions):
        """
        Helps in decoding the embedded image feature vectors and pre-processed captions, to generate/predict the next word in 
        captions
        """
        # To avoid mismatch in the shape after concatenation (as last column would be an array of 1s,
        # after concatenation)
        captions= captions[:,:-1] 
        
        # 1. Embedding the captions
        """
        Pass the captions through the embedding layer so that the model can find the relationships between the word tokens
        better
        """
        ## captions_embedding will be a tensor of size([batch_size, (seq - 1), embed_size])
        captions_embedding= self.embed(captions)   
        
        # 2. Alter features size for concatenation (Unsqueeze at dim=1)-->Concat features with captions_embedding-->LSTM_inputs
        features= features.unsqueeze(1)  #
        ## Same size as that of captions_embedding (to concat we need tensors of same size).
        # Features tensor format: size([batch_size, 1, embed_size])  

        
        # 3. Concat embedded image features with embedded captions (Input to LSTM)
        LSTM_inputs = torch.cat((features, captions_embedding), dim=1)  # concat the tensors in dimension 1. 
        # Inputs to LSTM: tensor size([batch_size, seq, embed_size])
        
        
        ## 4. Initializing the hidden and cell state of LSTM
        # setup the device
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        batch_size= features.size(0)  # batch_size: Using .size(), since it is a tensor
        # Initializing the hidden and cell states to zeros
        self.hidden_state = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(device)
        self.cell_state = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(device)
        

        # 5. Passing the LSTM_inputs to LSTM unit(s) along with init hidden_state and cell_state
        outputs, (self.hidden_state, self.cell_state) = self.lstm(LSTM_inputs, (self.hidden_state, self.cell_state))
        
        # 6. Passing the output of LSTM to Linear FC_layer to get the final outputs
        outputs= self.fc_output(outputs) 
        
        return outputs
    
        
        
    def sample(self, features, states=None, max_len=20):
        """" 
        Accepts a single pre-processed image tensor (input) and returns predicted indices of word tokens to generate
        captions for the given image (list of tensor ids upto a max length of 'max_len'). Aborts the generation of subsequent
        word indices within a caption when word index corresponding to <end> token (i.e. 1) impyling end of sentence is
        reached, or when a maximum length of 'max_len' value is reached (whichever occurs earlier).
        """
        
        predicted_idx_list = []
        sample_LSTM_inputs = features

        for word in range(max_len):  
            # 1. LSTM layer 
            # Inputs to LSTM word-by-word: tensor size([batch_size=1 image, seq=1 word, embed_size])  
            outputs, states = self.lstm(sample_LSTM_inputs, states)   
            
            # 2. LSTM layer to Fully-Connected Layer
            outputs = self.fc_output(outputs)  ## Outputs generated: size([batch_size=1 image, seq=1 word, embed_size]) 
            
            ## max value within dim=2 i.e. within the embed_size of 500; finding (max_value, respective_index)
            max_value_tensor, predicted_argmax_idx_tensor = outputs.max(2) 
            
            ## 3. Predicted word idx: Appending argmax index to predicted_idx_list
            predicted_idx_list.append(predicted_argmax_idx_tensor.item())
            if predicted_argmax_idx_tensor.item()==1:  ## Terminate the sentence when idx 1 mapped to <end> token is detected
                break;    
            
            ## 4. LSTM output index (word idx prediction) becomes the input for the next LSTM cell to generate next word idx
            sample_LSTM_inputs = self.embed(predicted_argmax_idx_tensor)  
            # The loop continues till the entire caption is generated.

        return predicted_idx_list    ##  returns predicted indices list (predicted indices of word tokens)  
      
    
   