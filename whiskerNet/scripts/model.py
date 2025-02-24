import torch
from torch import nn

class ContactTransformer(nn.Module):
    def __init__(self, sensor_dim, max_len,**kwargs):
        """
        encoder-decoder transformer model for contact prediction
        """
        super().__init__()
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
            )
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=128, nhead=8, dim_feedforward=512)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
        self.position_encoder = PositionalEncoding(d_model=128, max_len=max_len)
        self.mlp = nn.Linear(128, 3)
        
    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))  


    def forward(self, signal, **kwargs):
        """
        Args:
            signal (tensor): input data of shape (batch_size, sensor_dim, time_steps), 
                            which is the sensor signal in y, z directrion in previous time_steps
            property (tensor): input data of shape (batch_size, data_length, data_dim)
        """
        
        sensor_feat = self.sensor_encoder(signal) # (batch_size, time_steps, 64)
        feat = sensor_feat # (batch_size, time_steps, 128)
        feat = feat.permute(1, 0, 2) # (time_steps, batch_size, 256)
        feat = self.position_encoder(feat)  # apply position encoding
        mask = self._generate_square_subsequent_mask(feat.shape[0]).to(feat.device)
        decoder_output = self.transformer_decoder(feat, feat, mask, mask, tgt_is_causal=True, memory_is_causal=True) # conditioned on current observation
        pred_contact = self.mlp(decoder_output.permute(1, 0, 2))
        return pred_contact

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
       # Initialize the embeddings for each position
        self.position_embeddings = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)
        # Registering position ids
        self.register_buffer("position_ids", torch.arange(max_len).expand((1, -1)))

    def forward(self, x):
        # Get the position embeddings
        position_embeddings = self.position_embeddings(self.position_ids).permute(1, 0, 2)[:x.shape[0]]
        
        # Expand the position embeddings to match the batch size of x
        position_embeddings = position_embeddings.expand_as(x)
        
        # Add the position embeddings to the input tensor
        return x + position_embeddings