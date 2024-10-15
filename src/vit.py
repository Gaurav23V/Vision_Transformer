class ViT(nn.Module):
  """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
  # 1. Initialize the class with hyperparameters
  def __init__(self,
               img_size:int = 224, # Training resolution in ViT paper
               in_channels:int = 3, # Number of channels in input image
               patch_size:int = 16, # Path size
               num_transformer_layers:int = 12, # Layers for ViT base
               embedding_dim:int = 768, # Hidden size D for ViT base
               mlp_size:int = 3072, # MLP size for ViT base
               num_heads: int = 12, # Heads for ViT base
               attn_dropout:float = 0, # Dropout for attention projection
               mlp_dropout:float = 0.1, # Dropout for dense/MLP layers
               embedding_dropout:float = 0.1, # Dropout for patch and position embeddings
               num_classes:int = 1000): # Default for the imageNet but can customize this
    super().__init__()

    # 2. Make sure the image size is divisible by the patch size
    assert img_size % patch_size == 0, f"Image size ({img_size}) must be divisible by patch size ({patch_size})"

    # 3. Calcualte the number of patches
    self.num_patches = (img_size * img_size) // patch_size**2

    # 4. Create leanable class embeddings (these need to go at the front of the sequence of patch embeddings)
    self.class_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                        requires_grad = True)

    # 5. Create leanable position embedding
    self.position_embedding = nn.Parameter(data = torch.randn(1, self.num_patches + 1, embedding_dim),
                                           requires_grad = True)

    # 6. Create embedding dropout value
    self.embedding_dropout = nn.Dropout(embedding_dropout)

    # 7. Create patch embedding layer
    self.patch_embedding = PatchEmbedding(in_channels = in_channels,
                                                patch_size = patch_size,
                                                embedding_dim = embedding_dim)

    # 8. Create transformer encoder block (we will stack these blocks using nn.Sequential())
    # Note: The "*" means "all"
    self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim = embedding_dim,
                                                                       num_heads = num_heads,
                                                                       mlp_size = mlp_size,
                                                                       mlp_dropout = mlp_dropout) for _ in range(num_transformer_layers)])

    # 9. Create classifier head
    self.classifier = nn.Sequential(
        nn.LayerNorm(normalized_shape = embedding_dim),
        nn.Linear(in_features = embedding_dim,
                  out_features = num_classes)
    )

  # 10. Create a forward() method
  def forward(self, x):
    # 11. Get the batch size
    batch_size = x.shape[0]

    # 12. Create class token embeddings and expand it to match the batch size
    class_token = self.class_embedding.expand(batch_size, -1, -1)

    # 13. Create patch embedding
    x = self.patch_embedding(x)

    # 14. Concat class embedding and patch embedding
    x = torch.cat((class_token, x), dim = 1)

    # 15. Adding position embeddings
    x = self.position_embedding + x

    # 16. run embedding dropout
    x = self.embedding_dropout(x)

    # 17. Pass patch, position and class embedding through the transformer encoder layer
    x = self.transformer_encoder(x)

    # 18. Put 0 index logits through classifier
    x = self.classifier(x[:, 0])

    return x