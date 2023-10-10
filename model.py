import torch
from torch import nn


class NewsEncoder(nn.Module):
    def __init__(
        self, 
        vector_dim, 
        news_dim, 
        window_size, 
        vocab, 
        word_vectors = None,
    ):
        super(NewsEncoder, self).__init__()
        self.vocab = vocab
        self.word_vectors = word_vectors
        self.vector_dim = vector_dim
        self.news_dim = news_dim
        self.window_size = window_size

        self.vocab_size = len(vocab)

        # word vector at [vocab_size + 1] for unknown words.
        self.word_embeddings = nn.Embedding(
            num_embeddings=self.vocab_size + 2,
            embedding_dim=self.vector_dim,
            padding_idx=0,
        )
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.vector_dim,
                out_channels=self.news_dim,
                kernel_size=3,
                padding=self.window_size,
            ),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1),
        )
        
        if word_vectors is not None:
            self.init_embed()

    def init_embed(self):
        for word, vec in self.word_vectors.items():
            idx = self.vocab[word]
            self.word_embeddings.weight.data[idx] = torch.tensor(vec)

    def forward(self, input_ids):
        embedding_output = self.word_embeddings(input_ids)
        embedding_output = embedding_output.permute(0, 2, 1)
        encoder_output = self.encoder(embedding_output)
        encoder_output = encoder_output[..., 0]
        return encoder_output


class UserEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(UserEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, history_vecs):
        history_vecs = history_vecs.mean(dim=0, keepdim=True)
        encoder_output = self.encoder(history_vecs)
        return encoder_output


class NewsRecBaseModel(nn.Module):
    def __init__(
        self,
        vector_dim,
        news_dim,
        window_size,
        vocab,
        word_vectors = None,
    ):
        super(NewsRecBaseModel, self).__init__()
        self.news_encoder = NewsEncoder(
            vector_dim=vector_dim,
            news_dim=news_dim,
            window_size=window_size,
            vocab=vocab,
            word_vectors=word_vectors,
        )
        self.user_encoder = UserEncoder(news_dim)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, batch_history, batch_imp, batch_label = None):
        user_vecs = []
        for history in batch_history:
            history_vecs = self.news_encoder(history)
            user_vecs.append(self.user_encoder(history_vecs))

        user_vecs = torch.cat(user_vecs, dim=0)        
        news_vecs = self.news_encoder(batch_imp)
        score = torch.mul(user_vecs, news_vecs).sum(dim=1)

        if batch_label is None:
            return score
        
        loss = self.loss_fn(score, batch_label.float())
        return loss, score
