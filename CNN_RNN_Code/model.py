# Imports
import torch
import torch.nn as nn
import torchvision.models as models

# Constants
END_TOKEN = "</s>"


# Encoder Architecture using Incption V3 Model and replacing the last layer
class EncoderCNN(nn.Module):

    def __init__(self, embedding_size, is_training=True):
        super(EncoderCNN, self).__init__()
        # keep track if is training or evaluation stage
        self.is_training = is_training

        # Get pretrained network
        self.inception_model = models.inception_v3(
            weights="Inception_V3_Weights.DEFAULT", aux_logits=True
        )
        # Change the last layer to collect image embedding
        self.inception_model.fc = nn.Linear(
            self.inception_model.fc.in_features, embedding_size
        )
        self.relu_activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(0.5)

    def forward(self, images):
        image_features = self.inception_model(images)
        # If it is trianing, we need the main component i.e. logits
        # When we set aux_logits to true, we get the main component from logits
        # We don't need to do this during evaluation
        if self.is_training:
            image_features = image_features.logits
        return self.dropout_layer(self.relu_activation(image_features))


# LSTM based decoder to get the outputs
# Here we are using a Teacher-Force ratio of 100 i.e. we pass the correct token to the model everytime
class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, layers):
        # Create embedding, lstm layer and develop this architecture
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout_layer = nn.Dropout(0.5)

    def forward(self, image_features, captions):
        # Get the embeddings from the caption
        embeddings = self.dropout_layer(self.embedding(captions))
        # Cocatenate the image features and the caption embeddings
        embeddings = torch.cat((image_features.unsqueeze(0), embeddings), dim=0)
        hidden_units, _ = self.lstm(embeddings)
        # Return outputs after the forward propagation
        outputs = self.linear(hidden_units)
        return outputs


# Main CNN-RNN network to handle the encoder decoder
class CNN_RNN_network(nn.Module):
    # Pass the hyperparameters
    def __init__(
        self, embedding_size, hidden_size, vocab_size, layers, is_training=True
    ):
        super(CNN_RNN_network, self).__init__()
        # Pass the target parameters to the appropriate network
        self.encoderCNN = EncoderCNN(embedding_size, is_training)
        self.decoderRNN = DecoderRNN(embedding_size, hidden_size, vocab_size, layers)

    def forward(self, images, captions):
        # Call the models
        image_features = self.encoderCNN(images)
        outputs = self.decoderRNN(image_features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=40):
        # Perform image captioning in the evaluation stage
        # Max caption length is 38, so we are setting it to 40 here
        predicted_caption = []
        # Evaluation
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            # Keep track of states and pass them to the LSTM network to get the next token
            curr_states = None

            for i in range(max_length):
                hidden_units, curr_states = self.decoderRNN.lstm(x, curr_states)
                output = self.decoderRNN.linear(hidden_units.squeeze(0))
                # Get the highest probability token from the vocabulary
                predicted_token = output.argmax(1)
                predicted_caption.append(predicted_token.item())
                # Pass in the predicted token to the network to get the next prediciton
                x = self.decoderRNN.embedding(predicted_token).unsqueeze(0)

                # If we encounter the END_TOKEN, it means that the caption has been generated
                if vocabulary.idx_word_map[predicted_token.item()] == END_TOKEN:
                    break

        # Decode it based on the indexing in vocabulary and provide the final caption
        predicted_caption = [
            vocabulary.idx_word_map[caption] for caption in predicted_caption
        ]
        return predicted_caption
