import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, BertModel
import networkx as nx
import bz2
import xml.etree.ElementTree as ET
from collections import namedtuple
import random
from wikidump_reader_2 import WikipediaDumpReader

# --- Model and Training Classes ---

class WaveletKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, wavelet='db4', scales=5):
        super().__init__()
        self.wavelet = wavelet
        self.scales = scales
        self.basis = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim // (2**s), output_dim))
            for s in range(scales)
        ])
        
    def forward(self, x):
        coeffs = pywt.wavedec(x.cpu().numpy(), self.wavelet, level=self.scales-1, axis=-1)
        transformed = []
        for s in range(self.scales):
            level = coeffs[s].shape[-1]
            basis = self.basis[s][:level]
            transformed.append(torch.matmul(torch.tensor(coeffs[s], device=x.device), basis))
        return torch.sum(torch.stack(transformed), dim=0)

class MengerWavKAN(nn.Module):
    def __init__(self, bert_dim=768, latent_dim=256, branches=20):
        super().__init__()
        self.encoder = nn.Sequential(
            WaveletKANLayer(bert_dim, 512),
            nn.LayerNorm(512),
            DendriticBlock(512, 256, branches)
        )
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            DendriticBlock(latent_dim, 512, branches),
            WaveletKANLayer(512, bert_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, adj):
        x = torch.sparse.mm(adj, x)
        encoded = self.encoder(x)
        mu, logvar = self.mu(encoded), self.logvar(encoded)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class DendriticBlock(nn.Module):
    def __init__(self, in_dim, out_dim, branches):
        super().__init__()
        self.branches = branches
        self.dendrites = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU())
            for _ in range(branches)
        ])
        
    def forward(self, x):
        branch_outputs = [d(x) for d in self.dendrites]
        return torch.max(torch.stack(branch_outputs), dim=0)[0]

class WikipediaKG:
    def __init__(self, index_path, dump_path):
        self.wiki_reader = WikipediaDumpReader(index_path, dump_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.scaler = MinMaxScaler()
        
    def build_graph(self, n_articles=10000, k=5):
        """Construct semantic graph from Wikipedia articles"""
        articles = self.wiki_reader.get_random_sample(n_articles)
        texts = [a['text'][:512] for a in articles if a and 'text' in a and a['text']]
        
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        
        self.embeddings = self.scaler.fit_transform(embeddings)
        adj = kneighbors_graph(self.embeddings, n_neighbors=k, mode='connectivity')
        self.graph = nx.from_scipy_sparse_array(adj)
        return self.embeddings, adj

class LoSwarm:
    def __init__(self, model_creator, n_particles=20):
        self.particles = [model_creator() for _ in range(n_particles)]
        self.optimizers = [torch.optim.Adam(p.parameters(), lr=1e-3) for p in self.particles]
        self.best_loss = float('inf')
        self.best_model = None
        
    def train_epoch(self, data, adj):
        epoch_losses = []
        for model, opt in zip(self.particles, self.optimizers):
            model.train()
            opt.zero_grad()
            
            recon, mu, logvar = model(data, adj)
            recon_loss = F.mse_loss(recon, data)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.1 * kld
            
            loss.backward()
            opt.step()
            
            loss_item = loss.item()
            epoch_losses.append(loss_item)
            
            if loss_item < self.best_loss:
                self.best_loss = loss_item
                self.best_model = model
                
        return np.mean(epoch_losses)

class KnowledgeGenerator:
    def __init__(self, model, wiki_kg):
        self.model = model
        self.wiki_kg = wiki_kg
        self.bert = wiki_kg.bert
        self.tokenizer = wiki_kg.tokenizer
        
    def generate(self, prompt=None, max_length=50, temp=0.7):
        self.model.eval()
        with torch.no_grad():
            if prompt:
                inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
                outputs = self.bert(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).numpy()
                emb = self.wiki_kg.scaler.transform(emb)
                encoded = self.model.encoder(torch.FloatTensor(emb))
                mu, _ = self.model.mu(encoded), self.model.logvar(encoded)
                z = mu
            else:
                z = torch.randn(1, self.model.mu.out_features)
                
            gen_emb = self.model.decoder(z)
            gen_emb = self.wiki_kg.scaler.inverse_transform(gen_emb.numpy())
            
            sim = torch.cosine_similarity(
                torch.FloatTensor(gen_emb),
                self.bert.embeddings.word_embeddings.weight,
                dim=-1
            )
            tokens = []
            for _ in range(max_length):
                probs = F.softmax(sim / temp, -1)
                token_id = torch.multinomial(probs, 1).item()
                tokens.append(self.tokenizer.decode([token_id]))
                # Update sim for next token - this is a simplified generation
                # A more advanced method would feed the generated token back into the model
                next_emb = self.bert.embeddings.word_embeddings.weight[token_id].unsqueeze(0)
                sim = torch.cosine_similarity(next_emb, self.bert.embeddings.word_embeddings.weight, dim=-1)

            return ' '.join(tokens).replace(' ##', '')

# --- Usage Example ---
if __name__ == "__main__":
    # 1. Initialize Wikipedia reader and build graph
    print("Initializing Wikipedia Knowledge Graph...")
    wiki = WikipediaKG(
        index_path="enwiki-20241201-pages-articles-multistream-index.txt.bz2",
        dump_path="enwiki-20241201-pages-articles-multistream.xml.bz2"
    )
    embeddings, adj_matrix = wiki.build_graph(n_articles=10000, k=5)
    
    # 2. Convert data to PyTorch tensors
    print("Converting data to tensors...")
    adj_coo = adj_matrix.tocoo()
    adj = torch.sparse_coo_tensor(
        np.vstack([adj_coo.row, adj_coo.col]),
        adj_coo.data.astype(np.float32),
        adj_coo.shape
    )
    data = torch.FloatTensor(embeddings)
    
    # 3. Initialize swarm of models
    print("Initializing model swarm...")
    def create_model():
        return MengerWavKAN(bert_dim=768, latent_dim=256, branches=20)
    
    swarm = LoSwarm(create_model, n_particles=10)
    
    # 4. Training loop
    print("Starting training...")
    for epoch in range(100):
        loss = swarm.train_epoch(data, adj)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Best Loss: {swarm.best_loss:.4f}")
    
    # 5. Generate from the best model
    print("\n--- Generating Knowledge ---")
    if swarm.best_model:
        generator = KnowledgeGenerator(swarm.best_model, wiki)
        
        # Prompt-based generation
        print("\n--- Prompt-based Generation ---")
        prompt = "Quantum entanglement in physics"
        print(f"Prompt: '{prompt}'")
        print(generator.generate(prompt, max_length=100))
        
        # Random generation
        print("\n--- Random Generation ---")
        print(generator.generate(max_length=100))
    else:
        print("Training did not result in a best model. Cannot generate text.")